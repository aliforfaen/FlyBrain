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


/* ------------------------------------------------------------- app state */

const state = {
  cfg: null,
  nNeurons: 0,
  magic: FALLBACK_MAGIC,
  headerBytes: FALLBACK_HEADER_BYTES,
  dtMs: 0.1,
  paused: false,
  autoRotate: false,
  lastFrameAt: 0,
  spikeRate: 0,
  silentSince: 0,
  silenceWarned: false,
  frameDtMs: 0,
  loop: null,          // last /api/loop or websocket "loop" payload
};

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
let fpsEma = 0;
let lastTick = 0;
let lastFpsPaint = 0;
let lastChartPaint = 0;

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

  // Normalised connectome index — used by the "index order" colour mode.
  const idx = new Float32Array(n);
  const inv = n > 1 ? 1 / (n - 1) : 0;
  for (let i = 0; i < n; i++) idx[i] = i * inv;
  geom.setAttribute('aIndexT', new THREE.BufferAttribute(idx, 1).setUsage(THREE.StaticDrawUsage));

  geom.computeBoundingBox();

  material = new THREE.ShaderMaterial({
    uniforms: {
      uPointSize: { value: 1.9 },
      uPixelRatio: { value: renderer.getPixelRatio() },
      uExposure: { value: 1.35 },
      uFloor: { value: 0.05 },
      uColorMode: { value: 0 },
    },
    vertexShader: VERT,
    fragmentShader: FRAG,
    transparent: true,
    depthTest: true,
    depthWrite: true,
    blending: THREE.NormalBlending,
  });

  points = new THREE.Points(geom, material);
  points.frustumCulled = false;
  scene.add(points);

  if (material) material.uniforms.uColorMode.value = 0;

  // A very faint bounding box gives the cloud a sense of scale.
  const box = geom.boundingBox.clone().expandByScalar(0.015);
  const helper = new THREE.Box3Helper(box, new THREE.Color(0x14304a));
  helper.material.transparent = true;
  helper.material.opacity = 0.30;
  helper.material.depthWrite = false;
  scene.add(helper);
}

const VERT = `
  uniform float uPointSize;
  uniform float uPixelRatio;
  uniform int uColorMode;

  attribute float aIntensity;
  attribute float aIndexT;

  varying float vValue;
  varying float vIntensity;
  varying float vFog;

  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    float inten = clamp(aIntensity / 255.0, 0.0, 1.0);
    vIntensity = inten;
    vFog = clamp((-mv.z - 1.15) / 1.75, 0.0, 1.0);

    float v = inten;
    if (uColorMode == 1) {
      v = clamp(position.z * 0.5 + 0.5, 0.0, 1.0);          // anterior <-> posterior
    } else if (uColorMode == 2) {
      v = aIndexT;                                          // connectome index order
    } else if (uColorMode == 3) {
      v = clamp(length(position.xz) * 2.2, 0.0, 1.0);       // distance from centre
    }
    vValue = v;

    // Firing neurons swell slightly, which reads as emphasis at 1-3 px.
    float boost = (uColorMode == 0) ? (1.0 + 1.15 * inten) : 1.0;
    float size = uPointSize * uPixelRatio * boost * (2.4 / max(0.30, -mv.z));
    gl_PointSize = clamp(size, 1.0, ${MAX_POINT_CHUNK}.0);
    gl_Position = projectionMatrix * mv;
  }
`;

const FRAG = `
  uniform float uExposure;
  uniform float uFloor;
  uniform int uColorMode;

  varying float vValue;
  varying float vIntensity;
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

    float t = (uColorMode == 0) ? vIntensity : vValue;
    t = clamp(t * uExposure, 0.0, 1.0);

    vec3 col;
    if (uColorMode == 0) {
      // Resting connectome reads as dim steel blue; spiking neurons climb the ramp.
      vec3 rest = vec3(0.100, 0.175, 0.310);
      float k = smoothstep(uFloor, 0.70, t);
      col = mix(rest, ramp(0.30 + 0.70 * t), k);
    } else {
      col = ramp(uFloor + (1.0 - uFloor) * t);
    }
    // Depth cue: the near surface of the cloud stays bright, the far side sinks.
    col *= mix(1.10, 0.55, vFog);
    gl_FragColor = vec4(col, edge);
  }
`;

/* ------------------------------------------------------------- frame loop */

function animate(now) {
  requestAnimationFrame(animate);
  const dt = lastTick ? now - lastTick : 16.7;
  lastTick = now;
  if (dt > 0 && dt < 1000) fpsEma = fpsEma ? fpsEma * 0.92 + (1000 / dt) * 0.08 : 1000 / dt;

  if (!document.hidden) {
    if (controls) controls.update();
    if (renderer && scene && camera) renderer.render(scene, camera);

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
  if (state.paused) { tag.textContent = 'paused'; return; }
  if (!state.lastFrameAt) { tag.textContent = 'idle'; return; }
  const age = now - state.lastFrameAt;
  // Pacing means the brain deliberately steps in bursts and then waits. Frames *should* be
  // absent between decisions, so reporting "stalled" would cry wolf about the configured
  // behaviour. Only warn once the gap exceeds what the pacing actually asks for.
  const interval = Number(state.loop && state.loop.settings && state.loop.settings.interval_s) || 0;
  const budget = 3000 + interval * 1000 * 1.5;
  if (age <= budget) { tag.textContent = interval > 0 ? 'waiting' : 'live'; return; }
  tag.textContent = `stalled ${(age / 1000).toFixed(1)}s`;
}

/* ================================================================ regions */

let regionTimer = null;
let lastRegionOk = 0;

function startRegionPolling() {
  const tick = async () => {
    try {
      const data = await getJSON('/api/regions');
      renderRegions(Array.isArray(data.regions) ? data.regions : []);
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

function renderRegions(regions) {
  const host = $('#usage-rows');
  if (!host) return;
  state.regionCount = regions.length;

  $('#usage-poll').textContent = new Date().toLocaleTimeString('en-GB', { hour12: false });

  host.textContent = '';
  if (!regions.length) {
    host.appendChild(el('div', 'empty', 'no spiking regions in the last window'));
    return;
  }

  const head = el('div', 'row head');
  head.appendChild(el('span', 'name', 'cell class'));
  head.appendChild(el('span', 'n', 'neurons'));
  head.appendChild(el('span', 's', 'spikes'));
  host.appendChild(head);

  const maxSpikes = regions.reduce((m, r) => Math.max(m, Number(r.spikes) || 0), 0) || 1;

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
    if (state.paused) sendWS({ type: 'pause', value: true });
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

  // In-place update: no reallocation, no geometry rebuild.
  intensityArray.set(new Uint8Array(buf, hb, n));
  if (n < state.nNeurons) intensityArray.fill(0, n);
  intensityAttr.needsUpdate = true;

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
  const pace = $('#pace-note');
  if (pace) {
    const iv = Number(s.interval_s ?? 0);
    // Say what it actually costs, since that is the reason the control exists.
    pace.textContent = iv <= 0 ? 'flat out · ~165 W' : `every ${iv}s · ~${Math.round(19 + Math.min(1, 2.34 / iv) * 146)} W`;
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
      + (obs ? ` This room has been ${obs[0]}\u2013${obs[1]} \u00b0C so far.` : '');
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
  const rot = { on: false };
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
  // Keep the gentle drift, but never fight the user for control.
  if (controls) controls.autoRotate = false;
  return rot;
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

    initThree();
    buildControls();
    buildConnectUI();
    sizeChart();

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
    }

    // The control loop's current state, so the panel is populated before the first
    // websocket update arrives.
    try { paintLoop(await getJSON('/api/loop')); } catch (_) { /* the WS will fill it in */ }

    requestAnimationFrame(animate);
    startRegionPolling();
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
