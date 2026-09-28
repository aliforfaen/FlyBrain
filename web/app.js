/* ============================================================================
 * FlyBrain — live 3D connectome dashboard
 * ----------------------------------------------------------------------------
 * Renders FlyWire v783 (138,639 neurons) as a single THREE.Points cloud whose
 * per-neuron intensity arrives as a 1-byte-per-neuron binary WebSocket frame.
 * All colour mapping happens in the vertex/fragment shader; JavaScript never
 * touches per-point colour and never reallocates buffers in the frame loop.
 * ========================================================================= */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';


/* ------------------------------------------------------------------ config */

const FALLBACK_MAGIC = 0x4642524E;   // "FBRN"
/* Six 4-byte header fields (magic, seq, n, sim_ms, total, active) => 24.
 * /api/config.wire.header_bytes is authoritative; this is only the fallback. */
const FALLBACK_HEADER_BYTES = 24;
const MAX_POINT_CHUNK = 64;          // gl_PointSize clamp

const DEFAULT_CAM = { pos: [1.52, 0.53, 0.21], target: [0, 0, -0.12] };

const MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';


/* ------------------------------------------------------------------- utils */

const $ = (sel) => document.querySelector(sel);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined && text !== null) n.textContent = text;
  return n;
}

/* Set an element's text, if it exists. Deliberately not named `set`: that name is local to
 * buildConnectUI and writes input *values*, so reusing it here would silently write to
 * .value on a div and appear to do nothing at all. */
function setText(sel, text) {
  const node = $(sel);
  if (node) node.textContent = text == null ? '\u2014' : String(text);
}

const fmtInt = (n) => (Number.isFinite(n) ? Math.round(n).toLocaleString('en-US') : '—');

function fmtCount(n) {
  if (!Number.isFinite(n)) return '—';
  const v = Math.round(n);
  const a = Math.abs(v);
  if (a >= 1e9) return (v / 1e9).toFixed(2) + 'B';
  if (a >= 1e6) return (v / 1e6).toFixed(2) + 'M';
  if (a >= 1e4) return (v / 1e3).toFixed(1) + 'k';
  return v.toLocaleString('en-US');
}

function niceCeil(v) {
  if (!(v > 0)) return 1;
  const e = Math.pow(10, Math.floor(Math.log10(v)));
  const n = v / e;
  const m = n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10;
  return m * e;
}

let noticeTimer = null;
function showNotice(text, ms = 6000) {
  const box = $('#notice');
  box.textContent = text;
  box.hidden = false;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => { box.hidden = true; }, ms);
}

function bootMsg(text) {
  const m = $('#boot-msg');
  if (m) m.textContent = text;
}

function hideBoot() {
  const b = $('#boot');
  if (b && !b.classList.contains('gone')) {
    b.classList.add('gone');
    setTimeout(() => { b.style.display = 'none'; }, 500);
  }
}

function showFatal(title, msg, detail) {
  $('#fatal-title').textContent = title;
  $('#fatal-msg').textContent = msg;
  $('#fatal-detail').textContent = detail || '';
  $('#fatal').hidden = false;
  const b = $('#boot');
  if (b) b.style.display = 'none';
}

async function getJSON(url) {
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) {
    let body = '';
    try { body = (await res.text()).slice(0, 300); } catch (_) { /* ignore */ }
    const err = new Error(`HTTP ${res.status} ${res.statusText}${body ? ' — ' + body : ''}`);
    err.status = res.status;
    throw err;
  }
  return res.json();
}


/* ------------------------------------------------------------- preferences */

/* One tiny store for everything the *viewer* chooses: which panels are open, the view mode,
 * auto-orbit, and the two view toggles. Reads are guarded because a corrupt or unavailable
 * localStorage must never stop the dashboard booting — a privacy setting that blocks storage
 * would otherwise present as a blank page. */

const STORE_KEY = 'flybrain.prefs.v1';

function loadPrefs() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    const parsed = raw ? JSON.parse(raw) : null;
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch (err) {
    console.warn('could not read saved preferences; using defaults', err);
    return {};
  }
}

/* Panels that start collapsed: the diagnostics, the settings block, and the walkthrough. The
   point is to leave the dramatic things visible and put the reference material one click away. */
const DEFAULT_CLOSED = [
  'panel-pacing', 'panel-senses', 'panel-guides', 'panel-layers',
  'panel-trust', 'panel-journal', 'panel-connect', 'panel-usage',
];

const prefs = loadPrefs();
const collapsedPanels = new Set(
  Array.isArray(prefs.collapsed) ? prefs.collapsed : DEFAULT_CLOSED,
);

function savePrefs() {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify({
      collapsed: [...collapsedPanels],
      viewMode: state.viewMode,
      autoRotate: state.autoRotate,
      afterimage: state.afterimage,
      markers: state.markers,
    }));
  } catch (err) {
    // Storage blocked or full. The dashboard is not worse for it, so this is not an error.
    console.warn('could not save preferences', err);
  }
}

/* ------------------------------------------------------------- app state */

const state = {
  cfg: null,
  nNeurons: 0,
  magic: FALLBACK_MAGIC,
  headerBytes: FALLBACK_HEADER_BYTES,
  dtMs: 0.1,
  paused: false,
  autoRotate: prefs.autoRotate === true,
  lastFrameAt: 0,
  spikeRate: 0,
  silentSince: 0,
  silenceWarned: false,
  frameDtMs: 0,
  loop: null,          // last /api/loop or websocket "loop" payload
  //: Which question the cloud is answering. "activity" is the dramatic live view and stays the
  //: default; the other two are deliberate ways to explore rather than things to stumble into.
  viewMode: ['activity', 'families', 'spotlight'].includes(prefs.viewMode) ? prefs.viewMode : 'activity',
  //: A spike lingers for a moment after it fires. Off under a reduced-motion preference.
  afterimage: prefs.afterimage !== false && !prefersReducedMotion(),
  //: Front/back/left/right markers, derived from the annotations rather than guessed.
  markers: prefs.markers !== false,
  //: The spotlight selection: a family id and/or a sense id, -1 meaning "not selected".
  spotFamily: -1,
  spotSense: -1,
  groups: null,        // /api/groups metadata
  familyIds: null,     // Uint8Array, one per neuron
  senseIds: null,      // Uint8Array, one per neuron
  familyActivity: null, // id -> spikes, from /api/regions
  activeGuide: null,
  trail: null,          // last /api/timeline payload, so expanding needs no refetch
  trailExpanded: false,
  //: The last chart payload and the axes used to draw it, so a click can be turned back into the
  //: decision that produced the point under the cursor.
  chartRows: [],
  chartMap: null,
  selectedSeq: null,
  //: The last status blocks, so the Jev switch can repaint the badge without a refetch.
  lastTrust: null,
  lastJev: null,
};

function prefersReducedMotion() {
  return !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
}

/* ------------------------------------------------------------ collapsible panels */

/* Panels are made collapsible here rather than in the markup. There are fifteen of them, and the
 * body must stay in the DOM while collapsed — every paint function addresses its nodes by id, so
 * removing a body would silently stop it updating and the panel would show stale numbers when
 * reopened. Doing it in one place also means a new panel cannot forget to be collapsible. */

function setSummary(key, text) {
  setText('#sum-' + key, text);
}

function applyPanel(panel, open) {
  panel.dataset.collapsed = open ? '0' : '1';
  const btn = panel.querySelector('.panel-head');
  if (btn) btn.setAttribute('aria-expanded', String(open));
}

function togglePanel(id) {
  const panel = document.getElementById(id);
  if (!panel) return;
  const open = panel.dataset.collapsed === '1';
  applyPanel(panel, open);
  if (open) collapsedPanels.delete(id); else collapsedPanels.add(id);
  savePrefs();
}

function buildPanels() {
  for (const panel of document.querySelectorAll('section.panel')) {
    const h2 = panel.querySelector(':scope > h2');
    const body = panel.querySelector(':scope > .body');
    if (!h2 || !body || panel.querySelector('.panel-head')) continue;
    const key = panel.id.replace(/^panel-/, '');
    if (!body.id) body.id = 'body-' + key;

    const btn = el('button', 'panel-head');
    btn.type = 'button';
    btn.setAttribute('aria-controls', body.id);
    // Move the existing heading content into the button rather than rebuilding it, so the tags
    // the paint functions write to (`#pace-mode`, `#brain-tag`, …) keep their ids and handlers.
    while (h2.firstChild) btn.appendChild(h2.firstChild);
    const sum = el('span', 'sum');
    sum.id = 'sum-' + key;
    btn.appendChild(sum);
    h2.appendChild(btn);

    applyPanel(panel, !collapsedPanels.has(panel.id));
    btn.addEventListener('click', () => togglePanel(panel.id));
  }
}


/* Temperature and colour history is owned by the server (it is the thing that actually
 * made the decisions), so the chart cannot disagree with the numbers beside it. */

/* ------------------------------------------------------------- light colour */

/* Kelvin -> sRGB, Tanner Helland's approximation. Used only to paint the swatch, so an
 * approximation is exactly right: it is the perceived colour of the light, not a
 * colorimetric measurement. */
function kelvinToRgb(kelvin) {
  const t = Math.min(40000, Math.max(1000, Number(kelvin) || 0)) / 100;
  let r, g, b;
  if (t <= 66) {
    r = 255;
    g = 99.4708025861 * Math.log(t) - 161.1195681661;
  } else {
    r = 329.698727446 * Math.pow(t - 60, -0.1332047592);
    g = 288.1221695283 * Math.pow(t - 60, -0.0755148492);
  }
  if (t >= 66) b = 255;
  else if (t <= 19) b = 0;
  else b = 138.5177312231 * Math.log(t - 10) - 305.0447927307;
  const cl = (v) => Math.max(0, Math.min(255, Math.round(v)));
  return [cl(r), cl(g), cl(b)];
}
const kelvinToCss = (k) => `rgb(${kelvinToRgb(k).join(', ')})`;

/* --------------------------------------------------------- brain activity */

function paintBrainActivity(activeNeurons, rateHz) {
  const st = plainBrainWord(
    Number.isFinite(rateHz) ? rateHz / Math.max(state.nNeurons, 1) : NaN,
  );
  const tag = $('#brain-tag');
  if (tag) {
    tag.textContent = state.paused ? 'paused' : st.word;
    tag.classList.toggle('on', st.cls === 'on' && !state.paused);
  }
  const plain = $('#brain-plain');
  if (plain) {
    plain.textContent = state.paused
      ? 'Paused. The brain is not stepping, so the GPU is idle (~19 W instead of ~165 W). '
        + 'Press space or Resume to start it again.'
      : `About ${fmtCount(activeNeurons)} of the fly's ${fmtInt(state.nNeurons)} neurons `
        + `are firing right now — the brain is ${st.word}.`;
  }
  setSummary('brain', state.paused
    ? 'paused'
    : `${fmtCount(activeNeurons)} active · ${st.word}`);
}

/* A plain description of what the light looks like. Derived from the Kelvin value, not
 * from the band key, so the words always match the swatch beside them. */
function describeKelvin(k) {
  if (!Number.isFinite(k)) return '\u2014';
  if (k < 3000) return 'very warm, candle-like';
  if (k < 4500) return 'warm white';
  if (k < 5500) return 'neutral white';
  if (k < 7000) return 'cool white';
  return 'daylight blue';
}

/* ------------------------------------------------------------ loop panel */

function plainBrainWord(rateHz) {
  if (!Number.isFinite(rateHz)) return { word: 'idle', cls: '' };
  if (rateHz < 0.05) return { word: 'quiet', cls: '' };
  if (rateHz < 0.4) return { word: 'humming', cls: 'on' };
  if (rateHz < 1.5) return { word: 'busy', cls: 'on' };
  return { word: 'very busy', cls: 'on' };
}

function paintLoop(loop) {
  if (!loop) return;
  state.loop = loop;
  paintSenses(loop);
  if (typeof loop.paused === 'boolean' && loop.paused !== state.paused) setPaused(loop.paused);

  const modeEl = $('#loop-mode');
  if (modeEl) {
    modeEl.textContent = loop.available === false
      ? 'unavailable'
      : (loop.mode === 'mock' ? 'mock home' : (loop.dry_run ? 'dry run' : 'live'));
    modeEl.classList.toggle('warn', !!loop.dry_run && loop.mode !== 'mock');
  }

  if (loop.available === false) {
    const note = $('#loop-note');
    if (note) note.textContent = loop.reason || 'The control loop is not running.';
    return;
  }

  const tempEl = $('#loop-temp');
  const kelvinEl = $('#loop-kelvin');
  const hasReading = Number.isFinite(loop.temperature_c) && Number.isFinite(loop.kelvin);
  setSummary('loop', hasReading
    ? `${Number(loop.temperature_c).toFixed(1)} °C → ${Math.round(loop.kelvin)} K`
    : (loop.available === false ? 'unavailable' : 'waiting'));

  if (tempEl) {
    tempEl.innerHTML = hasReading
      ? `${loop.temperature_c.toFixed(1)}<small>\u00b0C</small>`
      : `\u2014<small>\u00b0C</small>`;
  }
  if (kelvinEl) {
    kelvinEl.innerHTML = hasReading
      ? `${fmtInt(loop.kelvin)}<small>K</small>`
      : `\u2014<small>K</small>`;
  }
  const entityEl = $('#loop-entity');
  if (entityEl) entityEl.textContent = loop.temperature_entity || '\u2014';
  const bandEl = $('#loop-band');
  if (bandEl) bandEl.textContent = describeKelvin(Number(loop.kelvin));
  const windowEl = $('#loop-window');
  if (windowEl) windowEl.textContent = `${Number(loop.window_ms).toFixed(0)} ms`;

  const chip = $('#loop-swatch');
  const title = $('#loop-swatch-title');
  const sub = $('#loop-swatch-sub');
  if (hasReading) {
    if (chip) {
      chip.style.background = kelvinToCss(loop.kelvin);
      chip.classList.add('lit');
    }
    if (title) title.textContent = describeKelvin(Number(loop.kelvin));
    const err = Number(loop.error_k);
    if (sub) {
      sub.textContent = Number.isFinite(err)
        ? `${err === 0 ? 'exactly the ideal colour' : (err > 0 ? err + ' K warmer' : Math.abs(err) + ' K cooler') + ' than the ideal'}`
        : '\u2014';
    }
  } else if (title) {
    title.textContent = 'waiting for the first reading\u2026';
  }

  const idealEl = $('#loop-ideal');
  if (idealEl) {
    idealEl.textContent = Number.isFinite(loop.ideal_kelvin)
      ? `${fmtInt(loop.ideal_kelvin)} K`
      : '\u2014';
  }
  const driveEl = $('#loop-drive');
  if (driveEl) {
    driveEl.textContent = `${fmtInt(loop.driven_neurons)} \u00b7 ${fmtInt(loop.readout_neurons)}`;
  }

  const note = $('#loop-note');
  if (note) {
    const act = loop.last_action;
    if (act) {
      const verb = act.sent ? 'sent' : 'would send';
      note.innerHTML =
        `<b>${verb}</b> <code>${act.entity_id} ${act.service}</code> ` +
        `<code>${act.data.color_temp_kelvin} K</code> \u00b7 ${fmtInt(loop.decisions)} decisions` +
        (loop.mode === 'mock'
          ? ' \u00b7 simulated home, so this really changed the fake light.'
          : (loop.dry_run ? ' \u00b7 <b>dry run, nothing real was touched</b>' : ''));
    } else {
      note.textContent = 'Waiting for the control loop to produce its first decision.';
    }
  }

  paintConnect(loop);
}

/* --------------------------------------------------------- history chart */

const chartCanvas = $('#chart');
const cctx = chartCanvas ? chartCanvas.getContext('2d') : null;
let chartW = 0, chartH = 0, chartDpr = 1, lastSpanLabel = '';

function sizeChart() {
  if (!chartCanvas) return;
  const rect = chartCanvas.getBoundingClientRect();
  chartDpr = Math.min(window.devicePixelRatio || 1, 2);
  chartW = Math.max(80, Math.round(rect.width));
  chartH = Math.max(48, Math.round(rect.height));
  chartCanvas.width = Math.round(chartW * chartDpr);
  chartCanvas.height = Math.round(chartH * chartDpr);
}

/* An X-Y plot rather than a time series: the x axis is the room temperature and the y
 * axis is the colour the loop chose for it, so each dot is one decision and the cloud of
 * dots *is* the mapping. The dashed line is what a perfect mapping would look like.
 * Overlapping two time series would have hidden exactly the thing worth seeing. */
function drawChart() {
  if (!chartW || !cctx) return;
  const loop = state.loop;
  const hist = (loop && Array.isArray(loop.history)) ? loop.history : [];
  const c = cctx;
  state.chartRows = hist;
  const lastK = hist.length ? hist[hist.length - 1].kelvin : null;
  setSummary('history', hist.length
    ? `${hist.length} decision(s)${lastK != null ? ` · last ${Math.round(lastK)} K` : ''}`
    : 'collecting…');
  c.setTransform(chartDpr, 0, 0, chartDpr, 0, 0);
  c.clearRect(0, 0, chartW, chartH);

  if (hist.length < 2) {
    c.fillStyle = 'rgba(150,170,190,0.55)';
    c.font = `11px ${MONO}`;
    c.textAlign = 'center';
    c.fillText('collecting readings\u2026', chartW / 2, chartH / 2 + 4);
    return;
  }

  const range = (loop && loop.temp_range_c) || [10, 35];
  const centres = (loop && loop.band_centres_k) || {};
  const bandVals = Object.values(centres);
  const tLo = Number(range[0]), tHi = Number(range[1]);
  const kLo = bandVals.length ? Math.min(...bandVals) : 4000;
  const kHi = bandVals.length ? Math.max(...bandVals) : 6500;

  const padL = 42, padR = 40, padT = 9, padB = 16;
  const w = chartW - padL - padR;
  const h = chartH - padT - padB;
  const pad = (lo, hi, f) => { const m = (hi - lo) * f; return [lo - m, hi + m]; };
  const [xLo, xHi] = pad(tLo, tHi, 0.06);
  const [yLo, yHi] = pad(kLo, kHi, 0.08);
  const X = (t) => padL + ((t - xLo) / (xHi - xLo)) * w;
  const Y = (k) => padT + h - ((k - yLo) / (yHi - yLo)) * h;
  // Kept so a click can be turned back into a decision. Recomputing the axes in the click
  // handler is how a hit test silently drifts out of step with the thing it is testing.
  state.chartMap = { X, Y };

  // grid
  c.strokeStyle = 'rgba(120,150,180,0.12)';
  c.lineWidth = 1;
  for (let g = 0; g <= 3; g++) {
    const y = padT + (h * g) / 3;
    c.beginPath(); c.moveTo(padL, y); c.lineTo(padL + w, y); c.stroke();
  }

  // the ideal mapping
  c.setLineDash([4, 4]);
  c.strokeStyle = 'rgba(180,205,230,0.45)';
  c.beginPath();
  c.moveTo(X(tLo), Y(kLo));
  c.lineTo(X(tHi), Y(kHi));
  c.stroke();
  c.setLineDash([]);

  // the decisions
  c.beginPath();
  hist.forEach((d, i) => {
    const px = X(Number(d.temperature_c));
    const py = Y(Number(d.kelvin));
    i ? c.lineTo(px, py) : c.moveTo(px, py);
  });
  c.strokeStyle = 'rgba(143,208,255,0.5)';
  c.lineWidth = 1.2;
  c.stroke();

  // newest decision last, so it sits on top and reads as "now"
  const last = hist[hist.length - 1];
  const lx = X(Number(last.temperature_c));
  const ly = Y(Number(last.kelvin));
  const [r, g, b] = kelvinToRgb(last.kelvin);
  c.beginPath();
  c.arc(lx, ly, 4, 0, Math.PI * 2);
  c.fillStyle = `rgb(${r}, ${g}, ${b})`;
  c.fill();
  c.lineWidth = 1.5;
  c.strokeStyle = 'rgba(255,255,255,0.85)';
  c.stroke();

  // The decision the inspector is currently showing, ringed so the verdict below the chart is
  // visibly attached to one point rather than floating free.
  if (state.selectedSeq != null) {
    const picked = hist.find((d) => Number(d.seq) === Number(state.selectedSeq));
    if (picked) {
      const px = X(Number(picked.temperature_c));
      const py = Y(Number(picked.kelvin));
      c.beginPath();
      c.arc(px, py, 7, 0, Math.PI * 2);
      c.strokeStyle = 'rgba(244,168,63,0.95)';
      c.lineWidth = 1.6;
      c.stroke();
      c.beginPath();
      c.moveTo(px + 7, py); c.lineTo(px + 13, py);
      c.stroke();
    }
  }

  // axis labels
  c.font = `10px ${MONO}`;
  c.fillStyle = 'rgba(190,210,230,0.95)';
  c.textAlign = 'left';
  c.fillText(`${Math.round(kHi)}K`, 2, padT + 8);
  c.fillText(`${Math.round(kLo)}K`, 2, padT + h);
  c.textAlign = 'center';
  c.fillText(`${tLo.toFixed(0)}\u00b0C`, padL, chartH - 3);
  c.fillText(`${tHi.toFixed(0)}\u00b0C`, padL + w, chartH - 3);
  c.fillStyle = 'rgba(150,170,190,0.75)';
  c.fillText('room temperature', padL + w / 2, chartH - 3);

  const spanLabel = `${hist.length} decisions`;
  if (spanLabel !== lastSpanLabel) {
    lastSpanLabel = spanLabel;
    const el = $('#history-span');
    if (el) el.textContent = spanLabel;
  }
}

/* ================================================================= THREE */

const canvas = $('#gl');
let renderer = null;
let scene = null;
let camera = null;
let controls = null;
let points = null;
let material = null;
let intensityAttr = null;
let intensityArray = null;
/* Afterimage buffers. `frameBase` is what the server said, `glowValues` is the decaying copy,
 * and `recentIdx` lists only the neurons currently fading. */
let frameBase = null;
let glowValues = null;
let recentIdx = null;
let inRecent = null;
let recentCount = 0;
//: Seconds for a spike to fade to about a tenth. Short on purpose: this is a trace of something
//: that just happened, not a second, invented activity signal.
const GLOW_TAU_S = 0.35;
//: Markers placed from the annotation table, so the cloud reads as a head with a front and a back.
let markerGroup = null;
//: The connection trace drawn for the active guide, if any.
let traceGroup = null;

/* Which numeric code the shader expects for the current view mode. */
function viewModeCode() {
  return { activity: 0, families: 1, spotlight: 2 }[state.viewMode] ?? 0;
}

/* A uint8 id buffer as a float attribute. The shader compares ids, so a float is what it wants;
 * sharing one helper keeps the two id attributes the same shape. */
function familyAttr(bytes) {
  const arr = new Float32Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) arr[i] = bytes[i];
  const attr = new THREE.BufferAttribute(arr, 1);
  attr.setUsage(THREE.StaticDrawUsage);
  return attr;
}
let fpsEma = 0;
let lastTick = 0;
let lastFpsPaint = 0;
let lastChartPaint = 0;

/* Render on demand. The server's frame rate is compute-bound (often ~2/s during a burst, zero
 * while waiting for the heartbeat), so redrawing 138k points at 60 fps draws the same frame
 * ~30 times over and burns laptop battery for nothing on a dashboard meant to be left open
 * for days. The view renders only when something changed: a frame arrived, the camera moved
 * (OrbitControls fires 'change' for drags, damping and auto-orbit alike), the afterimage is
 * still fading, or a control touched the scene — everything routes through markViewDirty(). */
let viewDirty = true;

function markViewDirty() {
  viewDirty = true;
}

function initThree() {
  renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: false,
    alpha: false,
    powerPreference: 'high-performance',
    stencil: false,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x04060b, 1);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(42, 1, 0.01, 60);
  camera.position.set(DEFAULT_CAM.pos[0], DEFAULT_CAM.pos[1], DEFAULT_CAM.pos[2]);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(DEFAULT_CAM.target[0], DEFAULT_CAM.target[1], DEFAULT_CAM.target[2]);
  controls.enableDamping = true;
  controls.dampingFactor = 0.065;
  controls.rotateSpeed = 0.6;
  controls.zoomSpeed = 0.85;
  controls.panSpeed = 0.7;
  controls.minDistance = 0.35;
  controls.maxDistance = 9;
  controls.autoRotate = state.autoRotate;
  controls.autoRotateSpeed = 0.42;
  controls.update();
  controls.addEventListener('change', markViewDirty);

  resizeThree();
  window.addEventListener('resize', resizeThree);
  if (window.ResizeObserver) new ResizeObserver(resizeThree).observe($('#stage'));
}

function resizeThree() {
  if (!renderer) return;
  const w = Math.max(1, window.innerWidth);
  const h = Math.max(1, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  if (material) material.uniforms.uPixelRatio.value = renderer.getPixelRatio();
  markViewDirty();
  sizeChart();
}

function buildCloud(positions) {
  const n = state.nNeurons;

  const geom = new THREE.BufferGeometry();
  const posAttr = new THREE.BufferAttribute(positions.subarray(0, n * 3), 3);
  posAttr.setUsage(THREE.StaticDrawUsage);
  geom.setAttribute('position', posAttr);

  intensityArray = new Uint8Array(n);
  intensityAttr = new THREE.BufferAttribute(intensityArray, 1);
  intensityAttr.setUsage(THREE.DynamicDrawUsage);
  geom.setAttribute('aIntensity', intensityAttr);

  // Server truth, kept separate from the attribute the shader reads: the attribute carries the
  // afterimage blend, so overwriting it with the raw frame would erase the fade every time a
  // frame arrived. `glow` is the decaying half and `recentIdx` is the sparse list of neurons it
  // currently applies to — sparse because decaying 138,639 entries every animation frame would
  // be 8M writes a second to fade a few thousand cells.
  frameBase = new Uint8Array(n);
  glowValues = new Float32Array(n);
  recentIdx = new Int32Array(n);
  inRecent = new Uint8Array(n);
  recentCount = 0;

  // Family and sense ids, one byte each, straight from /api/groups/ids. Absent when the server
  // has no annotation table, in which case the family views fall back to the activity ramp.
  if (state.familyIds) geom.setAttribute('aFamily', familyAttr(state.familyIds));
  if (state.senseIds) geom.setAttribute('aSense', familyAttr(state.senseIds));

  geom.computeBoundingBox();

  const source = shaderSource(state.groups);
  material = new THREE.ShaderMaterial({
    uniforms: {
      // A touch smaller than the 1.9 this used to be: at 1.9 the resting cloud read as a solid
      // mass and individual firing cells had nowhere to stand out.
      uPointSize: { value: 1.45 },
      uPixelRatio: { value: renderer.getPixelRatio() },
      uExposure: { value: 1.35 },
      uFloor: { value: 0.05 },
      uViewMode: { value: viewModeCode() },
      uSpotFamily: { value: state.spotFamily },
      uSpotSense: { value: state.spotSense },
      uSpotDim: { value: 0.22 },
    },
    vertexShader: source.vert,
    fragmentShader: source.frag,
    transparent: true,
    depthTest: true,
    depthWrite: true,
    blending: THREE.NormalBlending,
  });

  points = new THREE.Points(geom, material);
  points.frustumCulled = false;
  scene.add(points);

  // A very faint bounding box gives the cloud a sense of scale.
  const box = geom.boundingBox.clone().expandByScalar(0.015);
  const helper = new THREE.Box3Helper(box, new THREE.Color(0x14304a));
  helper.material.transparent = true;
  helper.material.opacity = 0.30;
  helper.material.depthWrite = false;
  scene.add(helper);
  markViewDirty();
}

/* The family palette comes from /api/groups, so there is exactly one place that decides what
 * colour "optic" is — the Python vocabulary the tests check against the annotation table. The
 * shader is generated from it rather than duplicating eleven hex codes here. */
function hexToVec3(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(String(hex || ''));
  if (!m) return 'vec3(0.35, 0.39, 0.45)';
  const v = parseInt(m[1], 16);
  const r = ((v >> 16) & 255) / 255, g = ((v >> 8) & 255) / 255, b = (v & 255) / 255;
  return `vec3(${r.toFixed(3)}, ${g.toFixed(3)}, ${b.toFixed(3)})`;
}

function familyHueGLSL(groups) {
  const families = (groups && Array.isArray(groups.families)) ? groups.families : [];
  if (!families.length) {
    // No annotation table: every dot is the same grey, and the view modes that need a family
    // simply look like the activity view rather than breaking.
    return 'vec3 familyHue(float id) { return vec3(0.35, 0.39, 0.45); }';
  }
  // The branches are upper bounds on an ascending id, so **every** id needs its own bound,
  // including 0. Without an explicit `id < 0.5` first, unlabelled cells satisfy `id < 1.5` and
  // silently take the first real family's colour.
  const ordered = [...families].sort((a, b) => a.id - b.id);
  const lines = ordered.map((f) => `    if (id < ${(f.id + 0.5).toFixed(1)}) return ${hexToVec3(f.colour)};`);
  return [
    '  // Family id -> hue. An if/else chain rather than a uniform array: GLSL ES 1.00 restricts',
    '  // dynamic indexing of uniform arrays, and this has to compile everywhere three does.',
    '  vec3 familyHue(float id) {',
    ...lines,
    '    return vec3(0.35, 0.39, 0.45);',
    '  }',
  ].join('\n');
}

function shaderSource(groups) {
  const vert = `
  uniform float uPointSize;
  uniform float uPixelRatio;
  uniform int uViewMode;
  uniform int uSpotFamily;
  uniform int uSpotSense;
  uniform float uSpotDim;

  attribute float aIntensity;
  attribute float aFamily;
  attribute float aSense;

  varying float vValue;
  varying vec3 vFamilyHue;
  varying float vDim;
  varying float vSel;
  varying float vFog;

${familyHueGLSL(groups)}

  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    float inten = clamp(aIntensity / 255.0, 0.0, 1.0);
    vValue = inten;
    vFog = clamp((-mv.z - 1.15) / 1.75, 0.0, 1.0);
    vFamilyHue = familyHue(aFamily);

    // Spotlight fades everything that is not selected, rather than hiding it: the shape of the
    // brain is part of what makes a highlighted pathway legible.
    float dim = 1.0;
    float sel = 0.0;
    if (uViewMode == 2 && (uSpotFamily >= 0 || uSpotSense >= 0)) {
      bool hit = false;
      if (uSpotFamily >= 0 && abs(aFamily - float(uSpotFamily)) < 0.5) hit = true;
      if (uSpotSense >= 0 && abs(aSense - float(uSpotSense)) < 0.5) hit = true;
      dim = hit ? 1.0 : uSpotDim;
      sel = hit ? 1.0 : 0.0;
    }
    vDim = dim;
    vSel = sel;

    // Firing neurons swell slightly, which reads as emphasis at 1-3 px.
    float boost = 1.0 + 1.15 * inten;
    // The selected pathway is *lifted* as well as the rest being pushed down. Dimming alone is
    // not enough to find 29 neurons inside 138,639: the whole point of the spotlight is that the
    // small thing becomes the visible thing.
    float emphasis = mix(1.0, 2.8, sel);
    float size = uPointSize * uPixelRatio * boost * (2.4 / max(0.30, -mv.z)) * emphasis;
    gl_PointSize = clamp(size * mix(1.0, 0.82, 1.0 - dim), 1.0, ${MAX_POINT_CHUNK}.0);
    gl_Position = projectionMatrix * mv;
  }
`;

  const frag = `
  uniform float uExposure;
  uniform float uFloor;
  uniform int uViewMode;

  varying float vValue;
  varying vec3 vFamilyHue;
  varying float vDim;
  varying float vSel;
  varying float vFog;

  // Ordered ramp: near-black blue -> deep blue -> cyan -> amber -> white.
  vec3 ramp(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.010, 0.020, 0.090);
    vec3 c1 = vec3(0.060, 0.180, 0.680);
    vec3 c2 = vec3(0.100, 0.770, 0.950);
    vec3 c3 = vec3(0.980, 0.830, 0.290);
    vec3 c4 = vec3(1.000, 1.000, 0.960);
    vec3 c = mix(c0, c1, smoothstep(0.00, 0.30, t));
    c = mix(c, c2, smoothstep(0.26, 0.52, t));
    c = mix(c, c3, smoothstep(0.50, 0.74, t));
    c = mix(c, c4, smoothstep(0.76, 1.00, t));
    return c;
  }

  void main() {
    float d = length(gl_PointCoord - vec2(0.5));
    if (d > 0.5) discard;                       // round the square point into a disc
    float edge = smoothstep(0.5, 0.34, d);

    float t = clamp(vValue * uExposure, 0.0, 1.0);

    vec3 col;
    if (uViewMode == 0) {
      // Activity: resting connectome reads as dim steel blue; spiking neurons climb the ramp.
      vec3 rest = vec3(0.100, 0.175, 0.310);
      float k = smoothstep(uFloor, 0.70, t);
      col = mix(rest, ramp(0.30 + 0.70 * t), k);
    } else {
      // Families and spotlight: the hue is *what kind* of cell, the brightness is *whether it is
      // firing*. Two facts, two channels, so both stay readable at once.
      //
      // The resting floor is deliberately high (0.55). At a low floor the hues were technically
      // distinct but practically invisible: the optic lobe alone is 56% of the brain, so a cloud
      // of dark blue cells just reads as blue.
      vec3 base = vFamilyHue * (0.55 + 0.85 * t);
      // Only a *small* wash towards white, and only at the very top. Bleaching a firing cell
      // destroyed the family information exactly where there was activity to be interested in:
      // the central brain is the busiest family in a typical window, and at a 0.85 wash its
      // violet never appeared at all. Brightness now carries "firing" without erasing "kind".
      col = mix(base, vec3(1.0, 1.0, 0.97), smoothstep(0.80, 1.0, t) * 0.35);
    }
    // Depth cue: the near surface of the cloud stays bright, the far side sinks.
    col *= mix(1.10, 0.55, vFog);
    col *= vDim;
    // Selection adds light rather than changing hue, so the family colour and the firing
    // brightness both survive being picked out.
    col = mix(col, min(col * 1.9 + 0.18, vec3(1.0)), vSel);
    gl_FragColor = vec4(col, edge);
  }
`;

  return { vert, frag };
}

/* ------------------------------------------------------------- frame loop */

function animate(now) {
  requestAnimationFrame(animate);
  const dt = lastTick ? now - lastTick : 16.7;
  lastTick = now;
  if (dt > 0 && dt < 1000) fpsEma = fpsEma ? fpsEma * 0.92 + (1000 / dt) * 0.08 : 1000 / dt;

  if (!document.hidden) {
    if (controls) controls.update();
    // The afterimage keeps its own render cadence: while any cell is still fading the scene
    // changes every tick, so render until the last glow dies out.
    const glowing = state.afterimage && recentCount > 0;
    if (glowing) decayGlow(dt);
    if (viewDirty || glowing) {
      if (renderer && scene && camera) renderer.render(scene, camera);
      viewDirty = false;
    }

    if (now - lastChartPaint > 55) {
      lastChartPaint = now;
      drawChart();
    }
    if (now - lastFpsPaint > 240) {
      lastFpsPaint = now;
      const fpsEl = $('#ts-fps');
      if (fpsEl) fpsEl.textContent = fpsEma ? fpsEma.toFixed(0) : '—';
      paintStaleness(now);
    }
  }
}

function paintStaleness(now) {
  const tag = $('#brain-tag');
  if (!tag) return;
  if (state.paused) { tag.textContent = 'paused'; setSummary('view', 'paused'); return; }
  if (!state.lastFrameAt) { tag.textContent = 'idle'; return; }
  const age = now - state.lastFrameAt;
  // Pacing means the brain deliberately steps in bursts and then waits. Frames *should* be
  // absent between decisions, so reporting "stalled" would cry wolf about the configured
  // behaviour. Only warn once the gap exceeds what the pacing actually asks for.
  const interval = Number(state.loop && state.loop.settings && state.loop.settings.interval_s) || 0;
  const budget = 3000 + interval * 1000 * 1.5;
  if (age <= budget) {
    // "waiting" is the heartbeat and "bursting" is the house doing something. Both are
    // configured behaviour, so neither should read as a fault.
    const mode = state.loop && state.loop.pacing && state.loop.pacing.mode;
    if (mode === 'burst') { tag.textContent = 'bursting'; return; }
    tag.textContent = interval > 0 ? 'waiting' : 'live';
    setSummary('view', `${state.viewMode} · ${fpsEma ? fpsEma.toFixed(0) : '—'} fps`);
    return;
  }
  tag.textContent = `stalled ${(age / 1000).toFixed(1)}s`;
  setSummary('view', `stalled ${(age / 1000).toFixed(1)}s`);
}

/* ------------------------------------------------------------- afterimage */

/* A neuron that spikes stays visible for a moment after the frame that showed it. This is
 * *decoration on a real measurement*: the truth is `frameBase`, the fade is ours, and the note in
 * the view panel says so. It is also only ever visible within a burst — with a 60 s heartbeat the
 * brain produces frames seconds apart, so a fade of a third of a second cannot make an idle brain
 * look busy, and it is not meant to. The toggle exists so exact values can be read. */

function registerGlow(incoming, n) {
  if (!state.afterimage || !glowValues) return;
  for (let i = 0; i < n; i++) {
    const v = incoming[i];
    if (!v) continue;
    const g = v / 255;
    if (g > glowValues[i]) glowValues[i] = g;
    if (!inRecent[i]) {
      inRecent[i] = 1;
      recentIdx[recentCount++] = i;
    }
  }
  // Write the blend once, so a spiking cell is bright immediately rather than next tick.
  for (let k = 0; k < recentCount; k++) {
    const i = recentIdx[k];
    intensityArray[i] = Math.max(frameBase[i], Math.round(glowValues[i] * 255));
  }
}

function decayGlow(dtMs) {
  if (!state.afterimage || !glowValues || !recentCount) return;
  const decay = Math.exp(-Math.max(dtMs, 0) / 1000 / GLOW_TAU_S);
  let k = 0;
  while (k < recentCount) {
    const i = recentIdx[k];
    glowValues[i] *= decay;
    if (glowValues[i] < 0.02) {
      // Swap-remove: order does not matter, and compaction keeps the loop proportional to the
      // number of cells actually fading rather than to the size of the brain.
      inRecent[i] = 0;
      glowValues[i] = 0;
      intensityArray[i] = frameBase[i];
      recentIdx[k] = recentIdx[--recentCount];
      continue;
    }
    intensityArray[i] = Math.max(frameBase[i], Math.round(glowValues[i] * 255));
    k += 1;
  }
  intensityAttr.needsUpdate = true;
}

function clearGlow() {
  if (!glowValues) return;
  for (let k = 0; k < recentCount; k++) {
    const i = recentIdx[k];
    inRecent[i] = 0;
    glowValues[i] = 0;
    intensityArray[i] = frameBase[i];
  }
  recentCount = 0;
  intensityAttr.needsUpdate = true;
  markViewDirty();
}

/* =========================================================== cell families */

/* The legend, the spotlight and the two exploration views. The names, colours and group
 * memberships all come from /api/groups, which is built from the published cell annotations —
 * nothing here invents a grouping or a hue, and the legend always shows the name and the count
 * so colour is never the only thing carrying a meaning. */

async function loadGroups() {
  try {
    const [meta, idBuf] = await Promise.all([
      getJSON('/api/groups'),
      fetch('/api/groups/ids', { cache: 'no-store' }).then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status} ${r.statusText}`);
        return r.arrayBuffer();
      }),
    ]);
    const want = state.nNeurons * 2;
    if (idBuf.byteLength < want) {
      showNotice(`Group ids payload is ${fmtInt(idBuf.byteLength)} B, expected ${fmtInt(want)} B.`, 12000);
    }
    const raw = new Uint8Array(idBuf, 0, Math.min(idBuf.byteLength, want));
    const fam = new Uint8Array(state.nNeurons);
    const sen = new Uint8Array(state.nNeurons);
    for (let i = 0; i < fam.length; i++) {
      fam[i] = raw[i * 2] || 0;
      sen[i] = raw[i * 2 + 1] || 0;
    }
    state.groups = meta;
    state.familyIds = fam;
    state.senseIds = sen;
    return true;
  } catch (err) {
    // Not fatal. The cloud still renders and the activity view still works; only the family
    // colouring and the spotlight have nothing to go on, and the legend says so.
    showNotice(`Cell families unavailable (${err.message}). The brain still renders.`, 12000);
    return false;
  }
}

function buildFamiliesUI() {
  const host = $('#fam-rows');
  if (!host) return;
  const families = (state.groups && state.groups.families) || [];
  host.textContent = '';
  if (!families.length) {
    host.appendChild(el('div', 'empty', 'no cell annotations available'));
    setSummary('families', 'unavailable');
    return;
  }
  const tag = $('#families-tag');
  if (tag) tag.textContent = `${families.length} families`;
  setSummary('families', `${families.length} families`);

  // Biggest first: the legend is a table of contents for the brain, and the optic lobe is a
  // quarter of it. Ids stay stable; only the display order changes.
  for (const f of [...families].sort((a, b) => b.neurons - a.neurons)) {
    const row = el('button', 'fam-row');
    row.type = 'button';
    row.dataset.family = String(f.id);
    row.dataset.neurons = String(f.neurons);
    row.setAttribute('aria-pressed', String(state.spotFamily === f.id));
    const sw = el('span', 'sw');
    sw.style.background = f.colour;
    row.appendChild(sw);
    row.appendChild(el('span', 'nm', f.label));
    row.appendChild(el('span', 'ct', fmtInt(f.neurons)));
    row.title = `${f.label} — ${f.blurb}`;
    row.addEventListener('click', () => selectFamily(f.id));
    host.appendChild(row);
  }

  buildSenseChips();
}

function buildSenseChips() {
  const senses = (state.groups && state.groups.senses) || [];
  if (!senses.length) return;
  const anchor = $('#fam-detail');
  if (!anchor || $('#sense-chips')) return;
  const wrap = el('div', 'sense-chips');
  wrap.id = 'sense-chips';
  for (const s of senses) {
    const b = el('button', 'sense-chip');
    b.type = 'button';
    b.dataset.sense = String(s.id);
    b.dataset.wired = s.wired ? '1' : '0';
    b.setAttribute('aria-pressed', 'false');
    // A pathway with no sensor is still a real set of cells, so it is offered and labelled
    // rather than hidden — the panel would otherwise disagree with the brain on screen.
    b.appendChild(el('span', '', s.wired ? s.label : `${s.label} (no sensor)`));
    b.title = `${s.label} — ${s.blurb}`;
    b.addEventListener('click', () => selectSense(s.id));
    wrap.appendChild(b);
  }
  anchor.parentNode.insertBefore(wrap, anchor);
}

function paintFamilyActivity(families) {
  state.familyActivity = families || null;
  const byId = new Map((families || []).map((f) => [f.id, f]));
  let busiest = null;
  for (const row of document.querySelectorAll('.fam-row[data-family]')) {
    const id = Number(row.dataset.family);
    const f = byId.get(id);
    const ct = row.querySelector('.ct');
    if (!ct) continue;
    const neurons = Number(row.dataset.neurons) || 0;
    if (f && f.spikes > 0) {
      // Name and count always present; the spike number is the only thing that changes colour.
      ct.textContent = '';
      ct.appendChild(document.createTextNode(`${fmtInt(neurons)} `));
      ct.appendChild(el('b', '', fmtCount(f.spikes)));
      row.dataset.hot = '1';
      if (!busiest || f.spikes > busiest.spikes) busiest = f;
    } else {
      ct.textContent = fmtInt(neurons);
      row.dataset.hot = '0';
    }
  }
  if (busiest) {
    const meta = ((state.groups && state.groups.families) || []).find((f) => f.id === busiest.id);
    setSummary('families', `${meta ? meta.label : busiest.key} busiest`);
  }
}

function applySpotlight() {
  if (!material) return;
  material.uniforms.uSpotFamily.value = state.spotFamily;
  material.uniforms.uSpotSense.value = state.spotSense;
  material.uniforms.uViewMode.value = viewModeCode();
  markViewDirty();
  const active = state.spotFamily >= 0 || state.spotSense >= 0;
  for (const b of document.querySelectorAll('#sense-chips .sense-chip')) {
    b.setAttribute('aria-pressed', String(Number(b.dataset.sense) === state.spotSense));
  }
  if (state.viewMode === 'spotlight' && !active) {
    setText('#view-note', 'Spotlight is on but nothing is selected — pick a family or a sense, ' +
      'and the rest of the brain fades back.');
  }
}

function selectFamily(id) {
  state.spotFamily = state.spotFamily === id ? -1 : id;
  state.spotSense = -1;
  showFamilyDetail(state.spotFamily >= 0 ? state.spotFamily : null);
  if (state.spotFamily >= 0) setViewMode('spotlight');
  syncFamilyRows();
}

function selectSense(id) {
  state.spotSense = state.spotSense === id ? -1 : id;
  state.spotFamily = -1;
  const sense = ((state.groups && state.groups.senses) || []).find((s) => s.id === state.spotSense);
  showFamilyDetail(null, sense || null);
  if (state.spotSense >= 0) {
    setViewMode('spotlight');
    showTrace(sense ? sense.key : null);
  } else {
    clearTrace();
  }
  syncFamilyRows();
}

function syncFamilyRows() {
  for (const row of document.querySelectorAll('.fam-row[data-family]')) {
    row.setAttribute('aria-pressed', String(Number(row.dataset.family) === state.spotFamily));
  }
  applySpotlight();
}

/* The detail card answers "what is this, and how much of it is there" for whatever is selected —
 * the plain-English blurb, the exact cell count, and the current activity. */
function showFamilyDetail(familyId, sense) {
  const host = $('#fam-detail');
  if (!host) return;
  host.textContent = '';
  if (familyId == null && !sense) return;

  const card = el('div', 'fam-detail');
  if (sense) {
    card.appendChild(el('div', 'hd', sense.label));
    card.appendChild(el('div', 'bl', sense.blurb));
    // No spike figure here on purpose. The family activity payload is grouped by *family*, and a
    // sense is a different grouping — looking its number up there would always miss and quietly
    // print nothing, which is worse than not offering it. Activity for a sense is visible on the
    // cloud itself, which is the point of the spotlight.
    card.appendChild(el('div', '', `Neurons: ${fmtInt(sense.neurons)}`
      + (sense.trained ? ' · the pathway the light is read from' : '')
      + (sense.wired ? ' · wired to a live sensor' : ' · no sensor wired to it')));
  } else {
    const f = ((state.groups && state.groups.families) || []).find((x) => x.id === familyId);
    if (!f) return;
    card.appendChild(el('div', 'hd', f.label));
    card.appendChild(el('div', 'bl', f.blurb));
    card.appendChild(el('div', '', `Neurons: ${fmtInt(f.neurons)} of ${fmtInt(state.nNeurons)} `
      + `(${((f.neurons / Math.max(state.nNeurons, 1)) * 100).toFixed(1)}%)`));
    const act = (state.familyActivity || []).find((x) => x.id === familyId);
    if (act) card.appendChild(el('div', '', `Spiking this window: ${fmtCount(act.spikes)}`));
  }
  host.appendChild(card);
}

function setViewMode(mode) {
  if (!['activity', 'families', 'spotlight'].includes(mode)) return;
  state.viewMode = mode;
  for (const b of document.querySelectorAll('#view-modes button')) {
    b.setAttribute('aria-pressed', String(b.dataset.view === mode));
  }
  if (material) material.uniforms.uViewMode.value = viewModeCode();
  markViewDirty();
  const note = $('#view-note');
  if (note) {
    note.textContent = {
      activity: "Each dot is one neuron. Brightness is how hard it is firing right now. Drag to orbit · scroll to zoom · right-drag to pan.",
      families: 'Colour is the kind of cell, brightness is whether it is firing. Click a family in the legend to pick it out.',
      spotlight: 'Everything outside the selection fades back, so one pathway is readable against the whole brain.',
    }[mode];
  }
  setSummary('view', `${mode} · ${fpsEma ? fpsEma.toFixed(0) : '\u2014'} fps`);
  savePrefs();
}

/* ======================================================== orientation markers */

/* Labels placed from the annotation table, not guessed. The front is the retina-and-antennae
 * end (negative z), which three separate landmarks agree on — see the Python that computes
 * `orientation` in /api/groups. Nothing is claimed about up and down. */

function labelSprite(text, colour) {
  const canvas2 = document.createElement('canvas');
  const ctx = canvas2.getContext('2d');
  const font = '600 22px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.font = font;
  const w = Math.ceil(ctx.measureText(text).width) + 16;
  canvas2.width = w;
  canvas2.height = 34;
  const c = canvas2.getContext('2d');
  c.font = font;
  c.fillStyle = 'rgba(4, 6, 11, 0.62)';
  c.fillRect(0, 0, w, 34);
  c.fillStyle = colour;
  c.textBaseline = 'middle';
  c.fillText(text, 8, 18);

  const tex = new THREE.CanvasTexture(canvas2);
  tex.minFilter = THREE.LinearFilter;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: tex, transparent: true, opacity: 0.7,
    // Always readable rather than swallowed by the cloud: these are a frame of reference, and a
    // reference you cannot see is worse than none.
    depthTest: false, depthWrite: false,
  }));
  sprite.scale.set(w / 1100, 0.038, 1);
  return sprite;
}

function buildOrientation() {
  if (markerGroup) return;
  const o = state.groups && state.groups.orientation;
  if (!o || !o.front) return;
  markerGroup = new THREE.Group();
  markerGroup.visible = state.markers;

  const anchor = (k) => new THREE.Vector3(...(o[k] ? o[k].anchor : [0, 0, 0]));
  for (const [key, colour] of [['front', '#8fd0ff'], ['back', '#7f93a8'], ['left', '#5f7d99'], ['right', '#5f7d99']]) {
    if (!o[key]) continue;
    const at = anchor(key);
    const sprite = labelSprite(o[key].label, colour);
    sprite.position.copy(at);
    markerGroup.add(sprite);
  }
  if (o.front && o.back) {
    // One thin shaft through the long axis, so "front" and "back" are visibly opposite ends of
    // the same line rather than two floating words.
    const a = anchor('front'), b = anchor('back');
    const geom = new THREE.BufferGeometry().setFromPoints([a, b]);
    const line = new THREE.Line(geom, new THREE.LineBasicMaterial({
      color: 0x2f4a68, transparent: true, opacity: 0.55, depthWrite: false,
    }));
    markerGroup.add(line);
  }
  scene.add(markerGroup);
}

function setMarkers(on) {
  state.markers = !!on;
  if (markerGroup) markerGroup.visible = state.markers;
  markViewDirty();
  const btn = $('#btn-markers');
  if (btn) {
    btn.setAttribute('aria-pressed', String(state.markers));
    btn.textContent = state.markers ? 'On' : 'Off';
  }
  savePrefs();
}

/* ============================================================== connection trace */

/* A handful of group-to-group lines for the selected pathway. The counts are exact; the geometry
 * is a summary, and the panel says so. Everything here degrades to "no trace drawn". */

function clearTrace() {
  if (!traceGroup || !scene) return;
  scene.remove(traceGroup);
  traceGroup.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) o.material.dispose();
  });
  traceGroup = null;
}

async function showTrace(group) {
  clearTrace();
  if (!group || !scene) return;
  let payload;
  try {
    payload = await getJSON(`/api/trace?group=${encodeURIComponent(group)}`);
  } catch (err) {
    // An overlay must never break the page: the guide reads fine without it.
    console.warn('no connection trace', err);
    return;
  }
  const targets = Array.isArray(payload.targets) ? payload.targets : [];
  if (!targets.length) return;
  const max = Math.max(...targets.map((t) => t.synapses), 1);
  traceGroup = new THREE.Group();
  for (const t of targets) {
    const opacity = 0.35 + 0.6 * (t.synapses / max);
    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(...t.from), new THREE.Vector3(...t.to),
    ]);
    traceGroup.add(new THREE.Line(geom, new THREE.LineBasicMaterial({
      color: 0xffc46b, transparent: true, opacity, depthWrite: false,
    })));
    const label = labelSprite(`${t.name} · ${fmtCount(t.synapses)}`, '#f4c98a');
    label.position.set(...t.to);
    traceGroup.add(label);
  }
  scene.add(traceGroup);
  markViewDirty();
}

/* ============================================================ guided exploration */

/* ELI5 walkthroughs. Each one sets a documented view state and says what you are looking at.
 * The words restate the real role descriptions in mapping.py — no invented biology. */

const GUIDES = [
  {
    id: 'start',
    title: 'What am I even looking at?',
    body: 'Each dot is <b>one neuron</b>, and there are 138,639 of them — a whole fruit fly brain, '
      + 'taken from a real one and photographed slice by slice. The brain is <b>frozen</b>: it never '
      + 'learns. Only a small readout on top of it is trained, and that readout is what picks the '
      + 'colour of the light. Brightness means "this cell is firing right now".',
    setup: { view: 'activity' },
  },
  {
    id: 'orientation',
    title: 'Which end is the front?',
    body: 'The <b>front</b> is the eyes-and-antennae end, and it is marked. That is not a guess: the '
      + 'retina, the antennal sensors and the nerves coming up from the body all agree on which way '
      + 'round this brain is. Turn the orientation markers on if you have hidden them.',
    setup: { view: 'activity', markers: true, camera: 'front' },
  },
  {
    id: 'warmth',
    title: 'Follow the warmth',
    body: 'Your room temperature drives <b>29 neurons</b> — a tiny, specific pathway. Watch them: '
      + 'these are the only cells the light colour is actually read from. Everything else you can '
      + 'see is the rest of the brain doing whatever it does.',
    setup: { view: 'spotlight', sense: 'thermosensory', trace: true },
  },
  {
    id: 'light',
    title: 'Where does light come in?',
    body: 'The fly\u2019s <b>eyes</b> are the largest single group of cells here. Look for the big '
      + 'blue mass at the front-sides — that is the optic lobe, more than half the brain, and it does '
      + 'the first pass of seeing before anything is sent inwards.',
    setup: { view: 'families', family: 'optic', camera: 'front' },
  },
  {
    id: 'smell',
    title: 'What smells?',
    body: 'Flies live by smell. The <b>olfactory</b> cells sit in the antennae and the lobe right '
      + 'behind them, and they are wired straight into the learning centres — which is why a fly can '
      + 'learn an odour in one trial.',
    setup: { view: 'spotlight', sense: 'olfactory', trace: true },
  },
  {
    id: 'move',
    title: 'How does it decide to move?',
    body: '<b>Descending</b> neurons carry orders from the brain down to the body. If you only ever '
      + 'light up one pathway, make it this one: it is the fly\u2019s output, the equivalent of the '
      + 'colour this project sends to your lamp.',
    setup: { view: 'spotlight', family: 'descending', trace: true },
  },
  {
    id: 'why',
    title: 'What colour is it choosing, and why?',
    body: 'Open <b>Colour chosen</b>. Each dot is one real decision: room temperature across, the '
      + 'colour the brain picked up. The dashed line is what a perfect mapping would be. Click any '
      + 'dot and Jev will try to say why that decision came out the way it did.',
    setup: { view: 'activity', open: ['panel-history', 'panel-families'] },
  },
];

function buildGuides() {
  const host = $('#guide-list');
  if (!host) return;
  host.textContent = '';
  setSummary('guides', `${GUIDES.length} guides`);
  const tag = $('#guides-tag');
  if (tag) tag.textContent = `${GUIDES.length} guides`;

  GUIDES.forEach((g, i) => {
    const wrap = el('div', 'guide');
    wrap.dataset.guide = g.id;
    const btn = el('button', '');
    btn.type = 'button';
    btn.appendChild(el('span', 'num', String(i + 1)));
    btn.appendChild(el('span', '', g.title));
    btn.addEventListener('click', () => runGuide(g.id));
    wrap.appendChild(btn);
    host.appendChild(wrap);
  });
}

function runGuide(id) {
  const g = GUIDES.find((x) => x.id === id);
  if (!g) return;
  // Toggle: clicking the running guide again clears it, so a guide is never a mode you are stuck in.
  const active = state.activeGuide === id ? null : id;
  state.activeGuide = active;

  for (const wrap of document.querySelectorAll('.guide')) {
    wrap.dataset.active = String(wrap.dataset.guide === active);
  }
  const bodyHost = $('#guide-body');
  if (bodyHost) {
    bodyHost.textContent = '';
    if (active) {
      const card = el('div', 'fam-detail');
      card.appendChild(el('div', 'hd', g.title));
      const p = el('div', 'bl');
      p.innerHTML = g.body;
      card.appendChild(p);
      bodyHost.appendChild(card);
    }
  }
  setSummary('guides', active ? g.title : `${GUIDES.length} guides`);

  if (!active) {
    clearTrace();
    return;
  }
  const s = g.setup || {};
  if (s.open) for (const pid of s.open) {
    if (collapsedPanels.has(pid)) togglePanel(pid);
  }
  if (s.markers) setMarkers(true);
  // `state.groups` is null when the annotation table is missing, in which case a guide that names
  // a group has nothing to point at and says so rather than failing silently.
  const pick = (list, key) => (list || []).find((x) => x.key === key);
  if (s.family) {
    const f = pick(state.groups && state.groups.families, s.family);
    if (f) { state.spotFamily = -1; selectFamily(f.id); }
    else showNotice(`No annotated group called ${s.family} in this build.`, 9000);
  }
  if (s.sense) {
    const x = pick(state.groups && state.groups.senses, s.sense);
    if (x) { state.spotSense = -1; selectSense(x.id); }
    else showNotice(`No sensory pathway called ${s.sense} in this build.`, 9000);
  }
  if (s.view) setViewMode(s.view);
  if (s.camera === 'front') lookFromFront();
  if (s.trace && !s.sense) showTrace(s.family);
}

/* Point the camera at the eyes-and-antennae end. Uses the marker the server derived, so the
 * camera and the label cannot disagree about which end is the front. */
function lookFromFront() {
  if (!camera || !controls) return;
  const o = state.groups && state.groups.orientation;
  const front = o && o.front ? o.front.anchor : null;
  const target = front ? new THREE.Vector3(-front[0] * 0.1, -front[1] * 0.1, 0.05)
    : new THREE.Vector3(...DEFAULT_CAM.target);
  controls.target.copy(target);
  camera.position.set(target.x, 0.42, target.z - 1.28);
  controls.update();
}

/* ============================================================= Jev, and the inspector */

/* Two separate things share this section: the on/off switch, which is about *cost and privacy*,
 * and the decision inspector, which is the one placement built so far. Both go through the same
 * vocabulary the server uses, so the badge and the card can never tell different stories. */

const JEV_REASONS = {
  ok: 'answering',
  no_key: 'no key set',
  unauthorized: 'key rejected',
  unreachable: 'unreachable',
  disabled: 'off (jevless)',
  unprobed: 'not checked yet',
};

const ROUTE_WORDS = {
  act: 'verdict',
  confirm: 'possible — low confidence',
  needs_human: 'not enough to call it',
};

function paintJevSwitch(jev) {
  const btn = $('#btn-jev');
  if (!btn) return;
  const on = !!(jev && jev.config && jev.config.enabled);
  btn.setAttribute('aria-pressed', String(on));
  btn.textContent = on ? 'On' : 'Off';
  const sub = $('#jev-switch-sub');
  if (sub) {
    sub.textContent = on
      ? 'on — one decision at a time, ~$0.0002 each'
      : 'off — nothing is sent, nothing is spent';
  }
}

function buildJevToggle() {
  const btn = $('#btn-jev');
  if (!btn) return;
  btn.addEventListener('click', async () => {
    const want = btn.getAttribute('aria-pressed') !== 'true';
    btn.disabled = true;
    const was = btn.textContent;
    btn.textContent = '…';
    try {
      const res = await fetch('/api/jev/enabled', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ value: want }),
      });
      const payload = await res.json().catch(() => null);
      if (!res.ok) throw new Error((payload && payload.detail) || `HTTP ${res.status}`);
      paintJevSwitch(payload);
      if (state.lastTrust) paintTrust(state.lastTrust, payload);
    } catch (err) {
      btn.textContent = was;
      showNotice(`Could not change the Jev setting: ${err.message}`, 9000);
    } finally {
      btn.disabled = false;
    }
  });
}

function nearestChartPoint(clientX, clientY) {
  const map = state.chartMap;
  const rows = state.chartRows || [];
  if (!map || !chartCanvas || !rows.length) return null;
  const rect = chartCanvas.getBoundingClientRect();
  // getBoundingClientRect is in CSS pixels, and so are the chart's own coordinates, so no DPR
  // scaling belongs here even though the backing store is scaled.
  const mx = clientX - rect.left;
  const my = clientY - rect.top;
  let best = null;
  let bestD2 = 14 * 14;      // a generous grab radius: these dots are 4 px and hard to hit exactly
  for (const d of rows) {
    if (d.temperature_c == null || d.kelvin == null) continue;
    const dx = map.X(Number(d.temperature_c)) - mx;
    const dy = map.Y(Number(d.kelvin)) - my;
    const d2 = dx * dx + dy * dy;
    if (d2 < bestD2) { bestD2 = d2; best = d; }
  }
  return best;
}

function bindChartClicks() {
  if (!chartCanvas) return;
  chartCanvas.addEventListener('click', (ev) => {
    const point = nearestChartPoint(ev.clientX, ev.clientY);
    if (!point || point.seq == null) {
      showNotice('Click closer to a dot — each one is a single decision.', 5000);
      return;
    }
    inspectDecision(Number(point.seq));
  });
}

async function inspectDecision(seq) {
  state.selectedSeq = seq;
  const host = $('#verdict-host');
  if (!host) return;
  host.textContent = '';
  const card = el('div', 'verdict');
  card.dataset.tone = 'mute';
  card.appendChild(el('div', 'vk', 'Decision inspector'));
  card.appendChild(el('div', 'vl', 'asking Jev…'));
  host.appendChild(card);

  try {
    const res = await fetch('/api/jev', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ seq }),
    });
    const payload = await res.json().catch(() => null);
    if (!res.ok) {
      throw new Error((payload && payload.detail) || `HTTP ${res.status} ${res.statusText}`);
    }
    drawVerdict(host, payload, seq);
  } catch (err) {
    host.textContent = '';
    const bad = el('div', 'verdict');
    bad.dataset.tone = 'warn';
    bad.appendChild(el('div', 'vk', 'Decision inspector'));
    bad.appendChild(el('div', 'vl', 'could not be asked'));
    bad.appendChild(el('div', '', `That call did not come back: ${err.message}. Nothing was cached, so trying again is safe.`));
    host.appendChild(bad);
  }
}

function drawVerdict(host, payload, seq) {
  host.textContent = '';
  const card = el('div', 'verdict');
  const row = (state.chartRows || []).find((d) => Number(d.seq) === Number(seq)) || {};

  if (payload && payload.available === false) {
    // "Off" and "broken" must not look the same — the same distinction the badge makes.
    card.dataset.tone = 'mute';
    card.appendChild(el('div', 'vk', 'Decision inspector'));
    card.appendChild(el('div', 'vl', JEV_REASONS[payload.reason] || payload.reason || 'unavailable'));
    card.appendChild(el('div', '', payload.detail || ''));
    if (payload.reason === 'disabled') {
      const btn = el('button', 'enable', 'Turn Jev on');
      btn.type = 'button';
      btn.addEventListener('click', () => $('#btn-jev')?.click());
      card.appendChild(btn);
    }
    if (state.lastTrust) paintTrust(state.lastTrust, null);
    host.appendChild(card);
    return;
  }

  const label = payload.label || 'unknown';
  const tone = payload.action === 'act' ? 'good'
    : (payload.action === 'confirm' ? 'warn' : 'mute');
  card.dataset.tone = tone;
  card.appendChild(el('div', 'vk', `Decision #${seq} · ${ROUTE_WORDS[payload.action] || payload.action}`));
  card.appendChild(el('div', 'vl', String(label).replace(/_/g, ' ')));

  const conf = payload.confidence;
  const kv = (k, v) => {
    const line = el('div', 'vrow');
    line.appendChild(el('span', 'k', k));
    line.appendChild(el('span', 'v', v));
    card.appendChild(line);
  };
  if (conf != null) kv('confidence', Number(conf).toFixed(3));
  if (payload.floor_ms != null && payload.latency_ms != null) {
    kv('latency', `${Math.round(payload.latency_ms)} ms (network floor ${Math.round(payload.floor_ms)} ms)`);
  }
  if (payload.cost_usd != null) {
    kv('cost', `$${Number(payload.cost_usd).toFixed(6)}${payload.cost_reported ? '' : ' (computed)'}`);
  }
  if (payload.cached) kv('cached', 'yes — this click was free');

  const probs = payload.probabilities || {};
  const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  if (entries.length) {
    const wrap = el('div', 'prob');
    for (const [name, p] of entries) {
      const line = el('div', 'prob-row');
      const bar = el('div', 'bar');
      const fill = el('i');
      fill.style.width = `${Math.max(1, Math.min(100, p * 100)).toFixed(0)}%`;
      bar.appendChild(fill);
      line.appendChild(bar);
      line.appendChild(el('div', 'n', `${name.replace(/_/g, ' ')} ${(p * 100).toFixed(0)}%`));
      wrap.appendChild(line);
    }
    card.appendChild(wrap);
  }

  // The three layers, in the order docs/jev.md insists on: the house, then the brain, then what
  // *we* mapped it to. The ordering is what keeps it clear that the colour is ours.
  const layers = el('div', 'layerline');
  layers.innerHTML =
    `<div><b>The house said</b> ${row.temperature_c != null ? Number(row.temperature_c).toFixed(1) + ' °C' : '—'}</div>` +
    `<div><b>The brain did</b> ${row.active_neurons != null ? fmtCount(row.active_neurons) + ' neurons firing' : '—'}` +
    `${row.total_spikes != null ? ', ' + fmtCount(row.total_spikes) + ' spikes' : ''}</div>` +
    `<div><b>We mapped it to</b> ${row.kelvin != null ? Math.round(row.kelvin) + ' K' : '—'}` +
    `${row.ideal_kelvin != null ? ' (ideal ' + Math.round(row.ideal_kelvin) + ' K)' : ''}</div>`;
  card.appendChild(layers);

  if (payload.action === 'needs_human') {
    card.appendChild(el('div', '', 'That is a real answer, not a failure: the state did not '
      + 'determine a verdict. Our own measurements produce this when a case contains a '
      + 'contradiction.'));
  }
  host.appendChild(card);
  if (state.lastTrust) paintTrust(state.lastTrust, null);
}

/* ================================================================ regions */

let regionTimer = null;
let lastRegionOk = 0;

function startRegionPolling() {
  const tick = async () => {
    try {
      const data = await getJSON('/api/regions');
      renderRegions(
        Array.isArray(data.regions) ? data.regions : [],
        Array.isArray(data.families) ? data.families : [],
      );
      lastRegionOk = performance.now();
    } catch (err) {
      const p = $('#usage-poll');
      if (p) p.textContent = 'poll failed';
    }
  };
  tick();
  clearInterval(regionTimer);
  regionTimer = setInterval(tick, 2000);
}

function renderRegions(regions, families) {
  // The legend's live numbers ride on this poll so they can never be a different moment from the
  // top-12 list, and so one request feeds both panels.
  paintFamilyActivity(families);
  const host = $('#usage-rows');
  if (!host) return;
  state.regionCount = regions.length;

  $('#usage-poll').textContent = new Date().toLocaleTimeString('en-GB', { hour12: false });

  host.textContent = '';
  if (!regions.length) {
    host.appendChild(el('div', 'empty', 'no spiking regions in the last window'));
    return;
  }

  setSummary('usage', `${regions[0].name} busiest`);

  const head = el('div', 'row head');
  head.appendChild(el('span', 'name', 'cell class'));
  head.appendChild(el('span', 'n', 'neurons'));
  head.appendChild(el('span', 's', 'spikes'));
  host.appendChild(head);

  const maxSpikes = regions.reduce((m, r) => Math.max(m, Number(r.spikes) || 0), 0) || 1;
  paintFamilyActivity(dataFamilies);

  for (const r of regions) {
    const row = el('div', 'row');
    const fill = el('i', 'fill');
    fill.style.width = `${Math.max(1.5, ((Number(r.spikes) || 0) / maxSpikes) * 100).toFixed(1)}%`;
    row.appendChild(fill);
    row.appendChild(el('span', 'name', String(r.name ?? '—')));
    row.appendChild(el('span', 'n', fmtInt(Number(r.neurons) || 0)));
    row.appendChild(el('span', 's', fmtCount(Number(r.spikes) || 0)));
    const rate = Number(r.rate_hz);
    row.title = `${r.name} · ${fmtInt(Number(r.neurons) || 0)} neurons · ${fmtInt(Number(r.spikes) || 0)} spikes` +
      (Number.isFinite(rate) ? ` · ${rate.toFixed(2)} Hz/neuron` : '');
    host.appendChild(row);
  }
}

/* ================================================================= pet */
/* The state word, its contributors, the pacing dial, the three layers, the trust badges and
 * the memory trail all arrive from /api/status and /api/timeline. They are rendered from one
 * payload on purpose: five panels assembled from five responses would show five different
 * moments side by side and quietly disagree with each other. */

const PET_TONES = {
  resting: 'resting', curious: 'curious', startled: 'startled', settling: 'settling',
};

let statusTimer = null, timelineTimer = null, statusFails = 0;
// Interpolated countdown, so the dial moves between polls instead of ticking in 1 Hz steps.
let paceDeadline = null, paceHeartbeat = 0;

function fmtDuration(seconds) {
  const s = Math.max(0, Math.round(Number(seconds) || 0));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, '0')}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`;
}

function startStatusPolling() {
  const tick = async () => {
    let data;
    try {
      data = await getJSON('/api/status');
    } catch (err) {
      statusFails += 1;
      // One notice, not one per second: a backend that is restarting is not sixty problems.
      if (statusFails === 3) showNotice(`Status poll is failing (${err.message}).`, 12000);
      return;
    }
    statusFails = 0;
    // Kept in its own try on purpose. A rendering bug and a dead backend look identical from the
    // outside if they share a catch -- and the wrong one gets debugged.
    try {
      paintStatus(data);
    } catch (err) {
      console.error('paintStatus failed', err);
    }
  };
  tick();
  clearInterval(statusTimer);
  statusTimer = setInterval(tick, 1000);
}

function startTimelinePolling() {
  const tick = async () => {
    let data;
    try {
      data = await getJSON('/api/timeline');
    } catch (_) {
      return; // the trail is decoration; it must never produce a notice
    }
    try {
      paintTrail(data);
    } catch (err) {
      console.error('paintTrail failed', err);
    }
  };
  tick();
  clearInterval(timelineTimer);
  timelineTimer = setInterval(tick, 5000);
}

function paintStatus(s) {
  if (!s) return;
  state.status = s;
  paintPet(s.pet, s.trust);
  paintPacing(s.pacing, s.journal);
  paintLayers(s.layers);
  paintTrust(s.trust, s.jev);
  paintJournal(s.journal);
}

/* --- the pet ------------------------------------------------------------- */

function paintPet(pet, trust) {
  if (!pet) return;
  // A pause stops windows, and the pet is only fed by a completed window -- so its state is a
  // frozen snapshot, not a description of now. Showing "startled · running at full rate" next to
  // a Paused button is a claim about a moment that has passed, which is the one thing this panel
  // exists not to do. The observation is kept (it is data, and it is the evidence for the word),
  // and labelled with its age instead of being presented as current.
  const paused = !!(trust && trust.paused);
  const age = Number.isFinite(Number(pet.observed_age_s)) ? Number(pet.observed_age_s) : null;
  const seen = age == null ? 'no window observed yet' : `last observed ${fmtDuration(age)} ago`;
  const key = paused ? 'paused' : (PET_TONES[pet.state] || 'waking');
  setSummary('pet', paused
    ? `paused · ${pet.state ? `last seen ${pet.state} ` : ''}${age == null ? '' : `${fmtDuration(age)} ago`}`.trim()
    : (pet.state ? `${pet.state}${pet.since_s ? ` · ${fmtDuration(pet.since_s)}` : ''}` : 'starting'));

  const glyph = $('#pet-glyph');
  if (glyph) glyph.dataset.state = key;
  const chip = $('#pet-chip');
  if (chip) chip.dataset.state = key;

  const word = paused ? 'paused' : (pet.state || 'waking up');
  setText('#pet-state', word);
  setText('#pet-chip-label', word);

  const held = pet.since_s >= 1 ? `for ${fmtDuration(pet.since_s)}` : 'just changed';
  setText('#pet-since', paused ? seen : (pet.state ? held : 'no window yet'));

  // Attribute the sentence to the window it came from rather than to now. Only when there is a
  // state to attribute: with no completed window the default sentence already says so.
  setText('#pet-sentence', paused && pet.state
    ? `Before pausing — ${seen}: ${pet.sentence || '—'}`
    : (pet.sentence || '—'));
  setText('#pet-honesty', pet.honesty || '—');
  if ($('#pet-source')) {
    $('#pet-source').textContent = paused ? 'paused' : (pet.state ? 'measured' : 'no data');
  }

  // Every label shows its contributors. A chip is a number and where it came from; a label
  // without one would be a claim, which is exactly what this panel exists to avoid.
  const host = $('#pet-contributors');
  if (!host) return;
  host.textContent = '';
  const contributors = Array.isArray(pet.contributors) ? pet.contributors : [];
  if (!contributors.length) {
    host.appendChild(el('span', 'chip dim', 'no measurements yet'));
    return;
  }
  for (const c of contributors) {
    const chip = el('span', 'chip');
    chip.dataset.source = c.source || '';
    chip.appendChild(el('span', '', `${c.label} `));
    if (typeof c.value === 'boolean') {
      chip.appendChild(el('b', '', c.value ? 'yes' : 'no'));
    } else if (typeof c.value === 'number') {
      // A ratio gets two decimals: `fmtInt(0.99)` is "1", and "1" reads as *exactly* baseline
      // when the honest answer is 0.99. Counts stay rounded, because a spike count is an integer.
      const shown = c.unit === 'x' ? `${c.value.toFixed(2)}×` : `${fmtInt(c.value)}${c.unit || ''}`;
      chip.appendChild(el('b', '', shown));
    } else {
      chip.appendChild(el('b', '', String(c.value ?? '—')));
    }
    host.appendChild(chip);
  }
}

/* --- pacing -------------------------------------------------------------- */

function paintPacing(pacing, journal) {
  if (!pacing) return;
  const mode = pacing.mode || 'waiting';
  const dial = $('#pace-dial');
  if (dial) dial.dataset.mode = mode;

  const label = { flat_out: 'flat out', waiting: 'waiting', burst: 'bursting' }[mode] || mode;
  setText('#pace-mode', label);
  setSummary('pacing', `${label}${pacing.observed_duty != null ? ` · ${(pacing.observed_duty * 100).toFixed(0)}% duty` : ''}`);
  const tag = $('#pace-mode');
  if (tag) {
    tag.classList.toggle('warn', mode === 'burst' || mode === 'flat_out');
    tag.classList.toggle('on', mode === 'waiting');
  }

  paceHeartbeat = Number(pacing.heartbeat_s) || 0;
  // The dial counts down locally between polls; polling at 1 Hz and repainting the ring only
  // then would make a 60 s heartbeat look like a slideshow.
  paceDeadline = mode === 'waiting' && Number.isFinite(Number(pacing.next_in_s))
    ? performance.now() + Number(pacing.next_in_s) * 1000
    : null;
  paintPaceCountdown();

  const trigger = Number(pacing.trigger_delta);
  setText('#pace-heart', paceHeartbeat > 0 ? `every ${paceHeartbeat}s` : 'flat out (0)');
  setText('#pace-burst', paceHeartbeat > 0 && Number(pacing.burst_s) > 0
    ? `${pacing.burst_s}s on a change` : 'off');
  setText('#pace-trigger', trigger > 0
    ? `${trigger} °C · ${pacing.trigger}` : 'off — plain heartbeat');
  setText('#pace-poll', Number(pacing.poll_s) > 0 ? `every ${pacing.poll_s}s` : 'off');

  // Measured, not configured: with a trigger the cost depends on how interesting the house has
  // been, which no formula over the settings can predict.
  const duty = pacing.observed_duty;
  const watts = pacing.observed_watts;
  setText('#pace-duty', duty == null ? 'measuring…' : `${(duty * 100).toFixed(1)}%`);
  setText('#pace-watts', watts == null ? 'measuring…' : `~${Math.round(watts)} W`);
  setText('#pace-kwh', watts == null ? '—' : `~${(watts * 24 / 1000).toFixed(2)} kWh`);
  setText('#pace-steps', fmtInt(pacing.steps));
  const bar = $('#pace-bar i');
  if (bar) bar.style.width = `${Math.min(100, (duty || 0) * 100)}%`;

  if (journal && journal.recording === false && $('#pace-note2')) {
    // Nothing to add: the note is static prose. Kept as a hook rather than an unused branch.
  }
}

function paintPaceCountdown() {
  const dial = $('#pace-dial');
  const when = $('#pace-when');
  if (!dial || !when) return;
  const mode = dial.dataset.mode;
  if (mode === 'flat_out') {
    when.textContent = 'now';
    dial.style.setProperty('--pct', 100);
    return;
  }
  if (mode === 'burst') {
    when.textContent = 'now';
    dial.style.setProperty('--pct', 100);
    return;
  }
  if (paceDeadline == null || !paceHeartbeat) {
    when.textContent = '—';
    return;
  }
  const remaining = Math.max(0, (paceDeadline - performance.now()) / 1000);
  when.textContent = remaining >= 1 ? `${Math.round(remaining)}s` : 'now';
  const elapsed = paceHeartbeat - remaining;
  dial.style.setProperty('--pct', Math.max(0, Math.min(100, (elapsed / paceHeartbeat) * 100)));
}

/* --- the three layers ---------------------------------------------------- */

function paintLayers(layers) {
  if (!layers) return;
  const { house, brain, mapped } = layers;
  setSummary('layers', house && house.value != null ? String(house.value) : '—');

  if (house) {
    setText('#layer-house-v', house.value == null ? 'no reading'
      : `${Number(house.value).toFixed(1)} °C`);
    const bits = [house.entity];
    if (house.stale) bits.push('STALE — the loop is not acting on this');
    else if (house.age_s != null) bits.push(`${fmtDuration(house.age_s)} ago`);
    const moved = Object.entries(house.changed || {})
      .map(([k, v]) => `${k.split('.').pop()} ${v > 0 ? '+' : ''}${v}`).slice(0, 3);
    if (moved.length) bits.push(`moved: ${moved.join(', ')}`);
    if ((house.senses || []).length) {
      bits.push(`${(house.senses || []).length} extra sense(s) wired`);
    }
    setText('#layer-house-d', bits.filter(Boolean).join(' · '));
  }

  if (brain) {
    setText('#layer-brain-v', `${fmtInt(brain.active_neurons)} active · ${fmtInt(brain.spikes)} spikes`);
    const regions = (brain.regions || [])
      .map((r) => `${r.name} ${r.spikes}`).join(', ');
    setText('#layer-brain-d', [
      `${fmtInt(brain.driven_neurons)} driven → ${fmtInt(brain.readout_neurons)} read out`,
      `${Number(brain.sim_ms || 0).toFixed(0)} ms of brain time`,
      regions ? `busiest: ${regions}` : null,
    ].filter(Boolean).join(' · '));
  }

  if (mapped) {
    setText('#layer-mapped-v', mapped.kelvin == null ? 'not yet decided'
      : `${fmtInt(mapped.kelvin)} K · ${mapped.band || ''}`);
    const action = mapped.action;
    let what = `${mapped.entity}`;
    if (action) {
      if (action.reason === 'sensor_stale') what += ' · held: no reading';
      else if (action.sent) what += ' · sent';
      else if (action.suppressed) what += ' · suppressed by the deadband';
      else if (action.dry_run) what += ' · dry run, not sent';
      else what += ' · not sent';
    }
    setText('#layer-mapped-d', [
      mapped.ideal_kelvin == null ? null : `ideal ${fmtInt(mapped.ideal_kelvin)} K`,
      mapped.error_k == null ? null : `off by ${mapped.error_k > 0 ? '+' : ''}${mapped.error_k} K`,
      `${fmtInt(mapped.decisions)} decision(s)`,
      what,
    ].filter(Boolean).join(' · '));
  }
}

/* --- trust --------------------------------------------------------------- */

function paintTrust(trust, jev) {
  setSummary('trust', trust ? `${trust.dry_run ? 'dry run' : 'live'}${trust.paused ? ' · paused' : ''}` : '—');
  if (trust) state.lastTrust = trust;
  if (jev) {
    state.lastJev = jev;
    paintJevSwitch(jev);
  }
  if (!trust) return;
  const host = $('#trust-badges');
  const live = trust.mode && trust.mode !== 'mock';
  setText('#trust-mode', live ? 'real house' : 'simulated house');
  const tag = $('#trust-mode');
  if (tag) tag.classList.toggle('warn', live);

  if (host) {
    host.textContent = '';
    const badge = (text, tone, title) => {
      const b = el('span', 'badge', text);
      b.dataset.tone = tone || '';
      if (title) b.title = title;
      host.appendChild(b);
      return b;
    };
    badge(trust.mode === 'mock' ? 'simulated home' : 'real home', live ? 'warn' : '');
    // In mock mode `will_send` is true by design — the simulated light genuinely changes — so
    // "live, it can send" in red would be both alarming and beside the point: there is no real
    // device on the other end. The three cases are genuinely different and are labelled so.
    if (!live) {
      badge('applies to the mock only', 'good',
        'No real device is involved: HA_MODE=mock, so the action changes the simulated light.');
    } else if (trust.dry_run) {
      badge('dry run — nothing sent', 'good', 'Service calls are logged and never dispatched.');
    } else {
      badge('live — it can send', 'bad', 'Service calls reach the real device.');
    }
    if (trust.paused) badge('paused', 'mute');
    else badge('running', 'good');
    badge(trust.recording ? 'recording windows' : 'not recording', trust.recording ? 'good' : 'mute');
    if (trust.always_on) badge('always on', 'warn', 'Runs with no dashboard open.');
  }

  const outputs = $('#trust-outputs');
  if (outputs) {
    outputs.textContent = '';
    const rows = Array.isArray(trust.outputs) ? trust.outputs : [];
    if (!rows.length) {
      outputs.appendChild(el('div', 'empty', 'no outputs configured'));
    } else {
      for (const o of rows) {
        const row = el('div', 'kv');
        row.appendChild(el('span', 'k', o.entity_id));
        const v = el('span', o.enabled ? 'v accent' : 'v', o.enabled ? o.what : `${o.what} (off)`);
        v.title = o.reason || '';
        row.appendChild(v);
        outputs.appendChild(row);
      }
    }
  }

  const trained = trust.trained;
  if ($('#trust-trained')) {
    if (!trained) {
      $('#trust-trained').textContent = 'No trained readout found — the loop cannot decide.';
    } else {
      const when = trained.trained_at ? new Date(trained.trained_at).toLocaleDateString() : 'unknown date';
      const range = Array.isArray(trained.temp_range_c) ? trained.temp_range_c.join('–') + ' °C' : '—';
      $('#trust-trained').innerHTML =
        `Readout: <b>${trained.regime || 'unknown regime'}</b> · trained ${when} · ` +
        `${range} → ${Object.keys(trained.band_centres_k || {}).join('/')} · ` +
        `window ${trained.window_ms} ms. A readout is only valid under the regime it was fitted in.`;
    }
  }

  if ($('#trust-jev') && jev) {
    // One vocabulary, defined once, shared with the decision-inspector card. Two copies drifted
    // the moment `disabled` was added to the server's reason set.
    const reasons = JEV_REASONS;
    const good = !!jev.available;
    const floor = jev.floor_ms == null ? 'floor not measured'
      : `network floor ${Math.round(jev.floor_ms)} ms`;
    // The credit balance and the per-call cost, because a balance reaching zero is how a feature
    // stops working without anyone noticing, and the cost is what decides whether asking often is
    // affordable. Cost is labelled reported-or-computed rather than implied to be one or other.
    const last = jev.last_call || null;
    const bill = last
      ? ` · $${Number(last.cost_usd).toFixed(6)}${last.cost_reported ? '' : ' (computed)'}/call` +
        (last.credits_remaining_usd == null
          ? '' : ` · $${Number(last.credits_remaining_usd).toFixed(3)} credit left`)
      : '';
    const bits = [`model ${jev.config?.model || '—'}`, floor];
    if (jev.config?.key_hint) bits.push(`key ${jev.config.key_hint}`);
    if (jev.calls) bits.push(`${jev.calls} call(s)`);
    $('#trust-jev').innerHTML =
      `Jev judgment layer: <b>${reasons[jev.reason] || jev.reason}</b> · ${bits.join(' · ')}${bill}` +
      `. <span style="color:var(--dim)">${good ? '' : String(jev.detail || '').slice(0, 120)}</span>`;
    const badgeHost = $('#trust-badges');
    if (badgeHost && good) {
      const b = el('span', 'badge', `Jev ${jev.calls ? 'live' : 'ready'}`);
      b.dataset.tone = 'good';
      badgeHost.appendChild(b);
    }
  }
}

/* --- journal ------------------------------------------------------------ */

function paintJournal(journal) {
  if (!journal) return;
  setText('#journal-line', journal.line || '—');
  setText('#journal-session', journal.session || (journal.recording ? 'recording' : 'not recording'));
  setSummary('journal', journal.line || 'nothing yet');

  const host = $('#journal-states');
  if (!host) return;
  const seconds = journal.seconds_in_state || {};
  const total = Math.max(1, Number(journal.observed_s) || 0);
  host.textContent = '';
  for (const name of ['resting', 'curious', 'startled', 'settling']) {
    const value = Number(seconds[name]) || 0;
    const row = el('div', 'vocab-row');
    row.dataset.state = name;
    row.appendChild(el('span', 'name', name));
    const bar = el('span', 'bar');
    const fill = el('i');
    fill.style.width = `${Math.min(100, (value / total) * 100)}%`;
    bar.appendChild(fill);
    row.appendChild(bar);
    row.appendChild(el('span', 'val', value >= 1 ? fmtDuration(value) : '—'));
    host.appendChild(row);
  }
}

/* --- memory trail -------------------------------------------------------- */

function paintTrail(data) {
  const host = $('#trail');
  if (!host || !data) return;
  const entries = Array.isArray(data.entries) ? data.entries : [];
  const now = Number(data.now) || (Date.now() / 1000);

  host.textContent = '';
  host.appendChild(el('div', 'trail-axis'));
  if (!entries.length) {
    setText('#trail-span', 'nothing yet');
    const list = $('#trail-list');
    if (list) { list.textContent = ''; list.appendChild(el('div', 'empty', 'nothing has happened yet')); }
    return;
  }

  // The window is the span the entries actually cover, floored at five minutes so a quiet
  // fifteen seconds does not get stretched across the whole strip and look busy.
  const oldest = Math.min(...entries.map((e) => e.t));
  const span = Math.max(300, now - oldest);
  const spanText = `last ${fmtDuration(span)} · ${entries.length} mark(s)`;
  setText('#trail-span', spanText);
  setSummary('trail', `last ${fmtDuration(span)} · ${entries.length} mark(s)`);

  for (const entry of entries) {
    const mark = el('span', 'mark');
    mark.dataset.kind = entry.kind || 'decision';
    const pct = Math.max(0, Math.min(98, ((entry.t - (now - span)) / span) * 100));
    mark.style.left = `${pct}%`;
    mark.title = `${new Date(entry.t * 1000).toLocaleTimeString('en-GB', { hour12: false })} · ` +
      `${entry.text}${entry.detail ? ' — ' + entry.detail : ''}`;
    host.appendChild(mark);
  }

  const list = $('#trail-list');
  if (!list) return;
  state.trail = data;
  list.textContent = '';

  // Three newest by default. The strip above is the glance; this is the readable version, and
  // expanding it grows the panel in place rather than opening a second scroller inside the
  // already-scrolling column — which is what made the wheel do different things in different
  // parts of the same panel.
  const shown = state.trailExpanded ? entries : entries.slice(0, 3);
  for (const entry of shown) {
    // Deliberately not `.row`: that class is a three-column grid for the region panels, and two
    // children inside it land in the wrong tracks and overlap.
    const row = el('div', 'trail-row');
    row.appendChild(el('span', 'when', new Date(entry.t * 1000)
      .toLocaleTimeString('en-GB', { hour12: false })));
    const meta = el('span', 'what');
    meta.appendChild(el('b', '', entry.text));
    if (entry.detail) meta.appendChild(el('span', '', ` — ${entry.detail}`));
    row.appendChild(meta);
    row.title = entry.detail || entry.text;
    list.appendChild(row);
  }

  if (entries.length <= 3 && !state.trailExpanded) return;
  const more = el('button', 'trail-more',
    state.trailExpanded ? 'Show fewer' : `Show all ${entries.length}`);
  more.type = 'button';
  more.addEventListener('click', () => {
    state.trailExpanded = !state.trailExpanded;
    if (state.trail) paintTrail(state.trail);
  });
  list.appendChild(more);
}

/* ============================================================== senses */

/* The fly's own names for the pathways a sensor is wired to. Showing the biological word
   rather than the role key is the whole point of the panel: it answers "which part of the
   brain is this driving", which `visual` alone does not. */
const SENSE_WORDS = {
  temperature: 'thermosensory',
  thermosensory: 'thermosensory',
  humidity: 'hygrosensory',
  hygrosensory: 'hygrosensory',
  illuminance: 'sight',
  motion: 'sight',
  visual: 'sight',
  audio: 'hearing',
  contact: 'touch',
  mechanosensory: 'touch',
  olfactory: 'smell',
  power: '—',
};

function paintSenses(loop) {
  const host = $('#sense-rows');
  if (!host || !loop) return;

  const channels = Array.isArray(loop.channels) ? loop.channels : [];
  setSummary('senses', channels.length ? `${channels.length + 1} senses` : 'temperature only');
  const tag = $('#senses-tag');
  if (tag) tag.textContent = channels.length ? `${channels.length + 1} senses` : 'temperature only';

  // The trained temperature path is not in `channels` (it keeps its own encoding, with the
  // sensitivity stretch and smoothing), so it is listed explicitly - otherwise the panel
  // would hide the one sense that is always there.
  const rows = [
    {
      entity_id: loop.temperature_entity,
      kind: 'temperature',
      neurons: loop.driven_neurons,
      rate_hz: loop.sensor_rate_hz,
      primary: true,
    },
    ...channels,
  ];

  const maxRate = rows.reduce((m, c) => Math.max(m, Number(c.rate_hz) || 0), 0) || 1;

  host.textContent = '';
  const head = el('div', 'row head sense');
  head.appendChild(el('span', 'name', 'sensor'));
  head.appendChild(el('span', 'whence', 'fly sense'));
  head.appendChild(el('span', 'n', 'neurons'));
  head.appendChild(el('span', 's', 'rate'));
  host.appendChild(head);

  for (const c of rows) {
    const row = el('div', `row sense${c.primary ? ' primary' : ''}`);
    const rate = Number(c.rate_hz);
    const fill = el('i', 'fill');
    fill.style.width = `${Math.max(1.5, ((Number.isFinite(rate) ? rate : 0) / maxRate) * 100).toFixed(1)}%`;
    row.appendChild(fill);
    row.appendChild(el('span', 'name', String(c.entity_id || '—')
      .replace(/^(sensor|binary_sensor)\./, '')));
    row.appendChild(el('span', 'whence', SENSE_WORDS[c.kind] || c.kind || '—'));
    row.appendChild(el('span', 'n', fmtInt(Number(c.neurons) || 0)));
    row.appendChild(el('span', 's', Number.isFinite(rate) ? `${rate.toFixed(0)} Hz` : '—'));
    row.title = `${c.entity_id}\n${fmtInt(Number(c.neurons) || 0)} neurons` +
      (Number.isFinite(rate) ? ` at ${rate.toFixed(1)} Hz` : '') +
      (c.primary ? '\nthe trained channel' : '');
    host.appendChild(row);
  }
}

/* ==================================================================== WS */

let ws = null;
let wsState = 'connecting';
let reconnectAttempt = 0;
let reconnectTimer = null;
let pingTimer = null;
let everConnected = false;

function setConn(stateName, label) {
  wsState = stateName;
  const box = $('#conn');
  if (!box) return;
  box.dataset.state = stateName;
  $('#conn-label').textContent = label || stateName;
}

function wsURL() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${location.host}/ws/activity`;
}

function connectWS() {
  clearTimeout(reconnectTimer);
  stopPing();
  setConn(everConnected ? 'reconnecting' : 'connecting');

  let socket;
  try {
    socket = new WebSocket(wsURL());
  } catch (err) {
    scheduleReconnect();
    return;
  }
  socket.binaryType = 'arraybuffer';
  ws = socket;

  socket.onopen = () => {
    if (ws !== socket) return;
    everConnected = true;
    reconnectAttempt = 0;
    setConn('live');
    startPing();
    // Deliberately *not* re-sending this tab's pause state. Pause belongs to the server, which is
    // what actually holds the GPU: a tab left paused — or restored with the page — used to pause a
    // freshly started brain the moment it reconnected. The client adopts the server's state from
    // the `loop` payload instead, and only ever *changes* it from a real click or key press.
  };
  socket.onmessage = onSocketMessage;
  socket.onerror = () => { /* onclose follows */ };
  socket.onclose = () => {
    if (ws === socket) ws = null;
    stopPing();
    scheduleReconnect();
  };
}

function scheduleReconnect() {
  if (wsState !== 'reconnecting') setConn('reconnecting', 'reconnecting');
  reconnectAttempt += 1;
  const base = Math.min(15000, 450 * Math.pow(1.7, Math.min(reconnectAttempt - 1, 8)));
  const delay = Math.round(base + Math.random() * 250);
  $('#conn-label').textContent = `reconnecting ${reconnectAttempt}`;
  clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(connectWS, delay);
}

function sendWS(msg) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return false;
  try { ws.send(JSON.stringify(msg)); return true; } catch (_) { return false; }
}

function startPing() {
  stopPing();
  pingTimer = setInterval(() => sendWS({ type: 'ping' }), 15000);
}
function stopPing() {
  if (pingTimer) { clearInterval(pingTimer); pingTimer = null; }
}

function onSocketMessage(ev) {
  try {
    const data = ev.data;
    if (typeof data === 'string') { handleText(data); return; }
    if (data instanceof ArrayBuffer) { handleBinary(data); return; }
    if (data && typeof data.arrayBuffer === 'function') {
      data.arrayBuffer().then((buf) => {
        try { handleBinary(buf); } catch (_) { /* ignore malformed frame */ }
      }).catch(() => {});
      return;
    }
    if (data instanceof Blob) return; // unreachable with binaryType=arraybuffer
  } catch (err) {
    console.warn('[flybrain] dropped a malformed message', err);
  }
}

function handleText(raw) {
  let msg;
  try { msg = JSON.parse(raw); } catch (_) { return; }
  if (!msg || typeof msg !== 'object') return;

  switch (msg.type) {
    case 'hello':
      if (Number.isFinite(msg.n_neurons) && msg.n_neurons > 0) {
        state.nNeurons = state.nNeurons || msg.n_neurons;
      }
      if (Number.isFinite(msg.dt_ms)) state.dtMs = msg.dt_ms;
      if (msg.loop) paintLoop(msg.loop);
      break;
    case 'loop':
      paintLoop(msg.loop);
      break;
    case 'pause':
      setPaused(!!msg.value);
      break;
    case 'error':
      // The server refused something this page sent (a bad settings value, most likely). Say
      // so rather than leaving the controls showing a value the server never accepted.
      showNotice(`Rejected by the backend: ${msg.error || 'unknown error'}`, 8000);
      break;
    case 'pong':
    case 'metrics':
    case 'regions':
    case 'info':
      break;
    default:
      break; // unknown text frames are ignored on purpose
  }
}

function handleBinary(buf) {
  const hb = state.headerBytes;
  const byteLength = buf.byteLength;
  if (byteLength < hb) return;

  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== state.magic) return;   // magic guard
  const n = dv.getUint32(8, true);
  if (n > state.nNeurons || hb + n > byteLength) return;

  const seq = dv.getUint32(4, true);
  const simMs = dv.getFloat32(12, true);
  const totalSpikes = dv.getUint32(16, true);
  const activeNeurons = dv.getUint32(20, true);

  // In-place update: no reallocation, no geometry rebuild. The raw frame goes into `frameBase`
  // rather than straight into the attribute, because the attribute also carries the afterimage
  // and overwriting it would wipe the fade on every frame.
  const incoming = new Uint8Array(buf, hb, n);
  if (intensityArray && frameBase) {
    frameBase.set(incoming);
    if (n < state.nNeurons) frameBase.fill(0, n);
    registerGlow(incoming, n);
    intensityAttr.needsUpdate = true;
    markViewDirty();
  }

  const now = performance.now();
  // The server's frame rate is compute-bound and can sit well below the configured
  // fps, so measure the real inter-frame interval for the chart's time axis.
  if (state.lastFrameAt) {
    const dt = now - state.lastFrameAt;
    if (dt > 0 && dt < 10000) state.frameDtMs = state.frameDtMs ? state.frameDtMs * 0.9 + dt * 0.1 : dt;
  }
  state.lastFrameAt = now;
  state.lastSeq = seq;

  const loopWindow = state.loop && Number(state.loop.window_ms);
  const windowS = Math.max((loopWindow || 50) / 1000, 1e-6);
  state.spikeRate = totalSpikes / windowS;

  // The connectome is essentially silent with default settings (no spontaneous
  // activity beyond a handful of stray spikes): point the user at the drive
  // controls rather than leaving them staring at a dormant cloud. Wall-clock
  // based so it behaves at any delivery rate.
  if (activeNeurons < 25) {
    if (!state.silentSince) state.silentSince = now;
    else if (!state.silenceWarned && now - state.silentSince > 12000) {
      state.silenceWarned = true;
      showNotice(
        'The brain is nearly silent — this usually means the control loop has not ' +
        'applied a temperature yet, or the sensor reading is missing.',
        14000,
      );
    }
  } else {
    state.silentSince = 0;
    state.silenceWarned = false;
  }

  $('#st-seq').textContent = fmtInt(seq);
  $('#st-sim').innerHTML = `${fmtInt(simMs)}<small>ms</small>`;
  $('#st-spikes').textContent = fmtCount(totalSpikes);
  $('#st-active').textContent = fmtCount(activeNeurons);
  paintBrainActivity(activeNeurons, state.spikeRate);
}


/* ====================================================== light connection */

const connect = { pending: null, timer: null };

function kelvinGradient(el, kLo, kHi, invert) {
  if (!el) return;
  const a = invert ? kHi : kLo;
  const b = invert ? kLo : kHi;
  el.style.background = `linear-gradient(to right, ${kelvinToCss(a)}, ${kelvinToCss(b)})`;
}

function paintConnect(loop) {
  const s = loop && loop.settings;
  if (!s) return;
  const set = (id, v) => { const el = $(id); if (el && document.activeElement !== el) el.value = v; };
  set('#src-min', Number(s.source_min_c).toFixed(1));
  set('#src-max', Number(s.source_max_c).toFixed(1));
  set('#k-min', Math.round(s.kelvin_min));
  set('#k-max', Math.round(s.kelvin_max));
  set('#smooth', (Number(s.smooth_ms) / 1000).toFixed(1));
  set('#deadband', Math.round(s.deadband_k));
  set('#interval', Number(s.interval_s ?? 0).toFixed(0));
  set('#poll', Number(s.poll_s ?? 0).toFixed(0));
  set('#burst', Number(s.burst_s ?? 0).toFixed(0));
  set('#trigger', Number(s.trigger_delta ?? 0).toFixed(1));
  const pace = $('#pace-note');
  if (pace) {
    const iv = Number(s.interval_s ?? 0);
    const trig = Number(s.trigger_delta ?? 0);
    // Say what it actually costs, since that is the reason the control exists. With a trigger
    // the cost depends on how interesting the house has been, so prefer the *observed* duty
    // cycle the pacer reports over a formula that assumes a fixed interval.
    const observed = state.loop && state.loop.pacing && state.loop.pacing.observed_watts;
    if (iv <= 0) {
      pace.textContent = 'flat out · ~165 W';
    } else if (trig > 0) {
      const base = Math.round(19 + Math.min(1, 2.34 / iv) * 146);
      // "average" rather than "measured" on its own: the number is the mean over the whole run,
      // from a power model, so while the brain is paused it stays high and describes the past.
      pace.textContent = observed == null
        ? `every ${iv}s + bursts · ~${base} W at rest`
        : `every ${iv}s + bursts · ~${Math.round(observed)} W average`;
    } else {
      pace.textContent = `every ${iv}s · ~${Math.round(19 + Math.min(1, 2.34 / iv) * 146)} W`;
    }
  }

  const inv = !!s.invert;
  $('#btn-invert')?.classList.toggle('on', inv);
  $('#btn-dryrun')?.classList.toggle('on', !!s.dry_run);
  const invBtn = $('#btn-invert');
  if (invBtn) {
    invBtn.textContent = inv ? 'Direction reversed' : 'Reverse direction';
    invBtn.title = inv
      ? 'A warmer room makes the light cooler'
      : 'A warmer room makes the light warmer';
  }
  const dryBtn = $('#btn-dryrun');
  if (dryBtn) {
    dryBtn.classList.toggle('disabled', loop.mode === 'mock');
    dryBtn.title = loop.mode === 'mock'
      ? 'The simulated home is not a real device, so calls are always applied'
      : 'When on, the loop logs what it would send and never touches a real device';
  }

  const span = Math.max(Number(s.source_max_c) - Number(s.source_min_c), 0.01);
  const kLo = Number(s.kelvin_min), kHi = Number(s.kelvin_max);
  const perDeg = (kHi - kLo) / span;
  const sens = $('#sens-note');
  if (sens) sens.textContent = `${perDeg.toFixed(0)} K per \u00b0C`;
  const kn = $('#k-note');
  if (kn) kn.textContent = `${Math.round(kLo)}\u2013${Math.round(kHi)} K`;

  kelvinGradient($('#src-grad'), kLo, kHi, inv);
  kelvinGradient($('#k-grad'), kLo, kHi, false);
  const marker = $('#k-marker');
  if (marker && Number.isFinite(loop.kelvin)) {
    const frac = Math.max(0, Math.min(1, (loop.kelvin - kLo) / Math.max(kHi - kLo, 1)));
    marker.style.left = `${(frac * 100).toFixed(1)}%`;
    marker.hidden = false;
  } else if (marker) {
    marker.hidden = true;
  }

  const tag = $('#connect-state');
  if (tag) {
    tag.textContent = loop.mode === 'mock' ? 'mock home' : (s.dry_run ? 'dry run' : 'live');
    tag.classList.toggle('warn', loop.mode !== 'mock' && !!s.dry_run);
  }

  const note = $('#connect-note');
  if (note) {
    const obs = loop.observed_range_c;
    note.innerHTML =
      `Reading <code>${s.temperature_entity}</code> \u2192 <code>${s.light_entity}</code>. ` +
      `A 1 \u00b0C change in the room moves the light <b>${perDeg.toFixed(0)} K</b>.`
      + (obs ? ` This room has been ${obs[0]}\u2013${obs[1]} \u00b0C so far.` : '')
      // The simulated house only contains its own entities, so a real id here would read as
      // unavailable forever -- the loop makes no decisions and the colour chart stays empty.
      // `LoopConfig.from_env` resolves this at startup; saying it here is what makes the entity
      // ids above explicable rather than suspicious when someone opens a `.env`-configured mock.
      + (loop.mode === 'mock'
        ? ' <b>Simulated house</b> \u2014 it only contains <code>sensor.living_room_temperature</code>'
          + ' and <code>light.kitchen</code>, so those are what a real-house <code>.env</code>'
          + ' is mapped onto. Set <code>HA_MODE=rest</code> to use your own ids.'
        : '');
  }
}

function queueConnect(patch) {
  connect.pending = Object.assign(connect.pending || {}, patch);
  clearTimeout(connect.timer);
  connect.timer = setTimeout(flushConnect, 350);
}

async function flushConnect() {
  const patch = connect.pending;
  connect.pending = null;
  if (!patch) return;
  try {
    const res = await fetch('/api/loop/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status} ${(await res.text()).slice(0, 120)}`);
    paintLoop(await res.json());
  } catch (err) {
    showNotice(`Could not save connection settings: ${err.message}`, 8000);
    try { paintLoop(await getJSON('/api/loop')); } catch (_) { /* ignore */ }
  }
}

function buildConnectUI() {
  for (const [id, key, scale] of [
    ['#src-min', 'source_min_c', 1],
    ['#src-max', 'source_max_c', 1],
    ['#k-min', 'kelvin_min', 1],
    ['#k-max', 'kelvin_max', 1],
    ['#smooth', 'smooth_ms', 1000],
    ['#deadband', 'deadband_k', 1],
    ['#interval', 'interval_s', 1],
    ['#poll', 'poll_s', 1],
    ['#burst', 'burst_s', 1],
    ['#trigger', 'trigger_delta', 1],
  ]) {
    const el = $(id);
    if (!el) continue;
    const push = () => {
      const raw = Number(el.value);
      if (Number.isFinite(raw)) queueConnect({ [key]: raw * scale });
    };
    el.addEventListener('change', push);
    el.addEventListener('keydown', (e) => { if (e.key === 'Enter') el.blur(); });
  }

  $('#btn-invert')?.addEventListener('click', () => {
    const cur = !!(state.loop && state.loop.settings && state.loop.settings.invert);
    queueConnect({ invert: !cur });
  });
  $('#btn-dryrun')?.addEventListener('click', () => {
    const s = state.loop && state.loop.settings;
    if (!s || (state.loop && state.loop.mode === 'mock')) return;
    queueConnect({ dry_run: !s.dry_run });
  });
  $('#btn-fit')?.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/loop/fit-range', { method: 'POST' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      paintLoop(await res.json());
      showNotice('Sensitivity fitted to the readings seen so far.', 5000);
    } catch (err) {
      showNotice(`Could not fit the range: ${err.message}`, 8000);
    }
  });
}

/* ================================================================ controls */

function setPaused(value) {
  state.paused = !!value;
  const btn = $('#btn-pause');
  if (btn) {
    btn.textContent = state.paused ? 'Resume' : 'Pause';
    btn.classList.toggle('on', state.paused);
    btn.classList.toggle('warn', state.paused);
  }
  const tag = $('#view-tag');
  if (tag) tag.textContent = state.paused ? 'paused' : 'live';
  if (wsState === 'live' || wsState === 'paused') {
    setConn(state.paused ? 'paused' : 'live', state.paused ? 'paused' : 'live');
  }
  paintStaleness(performance.now());
}

/* Pause is the one control worth having: it is what stops the GPU. Kept here rather than in
   the click handler so the button, the space bar and the server's own state all go through
   the same path. */
function togglePause() {
  const next = !state.paused;
  setPaused(next);
  if (!sendWS({ type: 'pause', value: next })) {
    showNotice('Not connected to the activity stream \u2014 pause not applied.', 5000);
  }
}

function buildControls() {
  $('#btn-pause')?.addEventListener('click', togglePause);
  // Space is the shortcut, because pausing is the thing you reach for while watching the
  // brain and you should not have to find the button. Ignored while typing in a field.
  document.addEventListener('keydown', (e) => {
    if (e.code !== 'Space' && e.key !== ' ') return;
    const a = document.activeElement;
    if (a && (a.tagName === 'INPUT' || a.tagName === 'TEXTAREA' || a.isContentEditable)) return;
    e.preventDefault();
    togglePause();
  });
  $('#btn-reset')?.addEventListener('click', () => {
    if (!camera || !controls) return;
    camera.position.set(DEFAULT_CAM.pos[0], DEFAULT_CAM.pos[1], DEFAULT_CAM.pos[2]);
    controls.target.set(DEFAULT_CAM.target[0], DEFAULT_CAM.target[1], DEFAULT_CAM.target[2]);
    controls.update();
  });

  // Auto-orbit: a slow drift that makes the 3D shape readable without touching the mouse. Off
  // under a reduced-motion preference, and never fighting the user — OrbitControls stops it the
  // moment you drag, which is the behaviour people expect.
  const orbitBtn = $('#btn-orbit');
  const paintOrbit = () => {
    if (orbitBtn) {
      orbitBtn.setAttribute('aria-pressed', String(state.autoRotate));
      orbitBtn.classList.toggle('on', state.autoRotate);
    }
    if (controls) controls.autoRotate = state.autoRotate;
  };
  orbitBtn?.addEventListener('click', () => {
    state.autoRotate = !state.autoRotate;
    paintOrbit();
    savePrefs();
  });
  if (prefersReducedMotion()) state.autoRotate = false;
  paintOrbit();

  for (const b of document.querySelectorAll('#view-modes button')) {
    b.addEventListener('click', () => {
      setViewMode(b.dataset.view);
      applySpotlight();
    });
  }
  for (const b of document.querySelectorAll('#view-modes button')) {
    b.setAttribute('aria-pressed', String(b.dataset.view === state.viewMode));
  }

  const afterBtn = $('#btn-afterimage');
  const paintAfter = () => {
    if (!afterBtn) return;
    afterBtn.setAttribute('aria-pressed', String(state.afterimage));
    afterBtn.textContent = state.afterimage ? 'On' : 'Off';
  };
  afterBtn?.addEventListener('click', () => {
    state.afterimage = !state.afterimage;
    if (!state.afterimage) clearGlow();
    paintAfter();
    savePrefs();
  });
  paintAfter();

  const markerBtn = $('#btn-markers');
  markerBtn?.addEventListener('click', () => setMarkers(!state.markers));
  setMarkers(state.markers);

  bindChartClicks();
  setViewMode(state.viewMode);
}


/* ==================================================================== boot */

function paintFooterMeta() {
  const cfg = state.cfg;
  if (!cfg) return;
  $('#foot-meta').textContent =
    `${cfg.device} \u00b7 dt ${Number(cfg.dt_ms).toFixed(2)} ms \u00b7 ` +
    `${fmtInt(cfg.n_neurons)} neurons \u00b7 ${fmtInt(cfg.n_synapses)} synapses`;
}

function paintHeader(cfg) {
  const n = Number(cfg.n_neurons) || 0;
  const s = Number(cfg.n_synapses) || 0;
  $('#dataset-id').innerHTML =
    `FlyWire v783 \u00b7 <b>${fmtInt(n)}</b> neurons \u00b7 <b>${fmtInt(s)}</b> synapses`;
  paintFooterMeta();
  document.title = `FlyBrain \u00b7 ${fmtInt(n)} neurons \u2014 FlyWire v783`;
}

async function waitForConfig() {
  let networkFails = 0;
  let lastErr = null;
  for (let i = 0; i < 90; i++) {
    try {
      return await getJSON('/api/config');
    } catch (err) {
      lastErr = err;
      if (err && err.status === 503) {
        bootMsg('backend is loading the connectome\u2026');
      } else if (err && typeof err.status === 'number') {
        throw err;
      } else {
        networkFails += 1;
        bootMsg(`backend unreachable \u2014 retry ${networkFails}/12`);
        if (networkFails >= 12) throw err;
      }
      await sleep(700);
    }
  }
  throw lastErr || new Error('timed out waiting for /api/config');
}

async function loadPositions() {
  const res = await fetch('/api/positions', { cache: 'no-store' });
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
  const buf = await res.arrayBuffer();
  const expected = state.nNeurons * 3 * 4;
  if (buf.byteLength < expected) {
    showNotice(`Positions payload is ${fmtInt(buf.byteLength)} B, expected ${fmtInt(expected)} B \u2014 rendering anyway.`, 12000);
  }
  return new Float32Array(buf, 0, Math.min(buf.byteLength / 4, state.nNeurons * 3));
}

async function boot() {
  try {
    bootMsg('contacting backend\u2026');
    const cfg = await waitForConfig();
    state.cfg = cfg;
    state.nNeurons = Number(cfg.n_neurons) || 0;
    state.magic = Number(cfg.wire?.magic) || FALLBACK_MAGIC;
    state.headerBytes = Number(cfg.wire?.header_bytes) || FALLBACK_HEADER_BYTES;
    state.dtMs = Number(cfg.dt_ms) || 0.1;

    if (!state.nNeurons) throw new Error('/api/config reported n_neurons = 0');

    paintHeader(cfg);
    bootMsg(`allocating ${fmtInt(state.nNeurons)} neuron sprites\u2026`);

    // Panels first: the one-line summaries they expose are written by the painters further down,
    // and a panel built after the first poll would sit there with an empty heading until the next.
    buildPanels();
    buildGuides();
    buildJevToggle();

    initThree();
    buildControls();
    buildConnectUI();
    sizeChart();

    // The family and sense ids must be in hand before the cloud is built, because they become
    // static vertex attributes — attaching them later would mean rebuilding the geometry.
    bootMsg('fetching cell families\u2026');
    await loadGroups();

    bootMsg('fetching soma positions\u2026');
    let positions;
    try {
      positions = await loadPositions();
    } catch (err) {
      showNotice(`Could not load /api/positions (${err.message}). 3D view disabled.`, 20000);
      positions = null;
    }

    if (positions) {
      bootMsg('uploading point cloud\u2026');
      buildCloud(positions);
      buildFamiliesUI();
      buildOrientation();
      applySpotlight();
    }

    // The control loop's current state, so the panel is populated before the first
    // websocket update arrives.
    try { paintLoop(await getJSON('/api/loop')); } catch (_) { /* the WS will fill it in */ }

    requestAnimationFrame(animate);
    startRegionPolling();
    startStatusPolling();
    startTimelinePolling();
    // The dial's own clock. Painting it only on each poll would make a 60 s heartbeat move in
    // one-second jumps; this is what makes it read as a heartbeat rather than a progress bar.
    setInterval(paintPaceCountdown, 200);
    $('#pet-chip')?.addEventListener('click', () => {
      $('#panel-pet')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    connectWS();
    hideBoot();

    canvas.addEventListener('webglcontextlost', (e) => {
      e.preventDefault();
      showNotice('WebGL context lost \u2014 reload the page to restore the 3D view.', 30000);
    });
  } catch (err) {
    showFatal(
      'Cannot reach the flybrain backend',
      'The dashboard could not load /api/config from this origin.',
      String(err && err.message ? err.message : err),
    );
  }
}

$('#fatal-retry')?.addEventListener('click', () => location.reload());

boot();
