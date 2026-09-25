# Documentation

The index for everything under `docs/`. For what the project *is*, start with the
[top-level README](../README.md); for orientation as a contributor or agent, read
[`AGENTS.md`](../AGENTS.md).

## The short version

A whole adult female *Drosophila* brain — the **FlyWire v783** connectome, **138,639 neurons**
and **15,091,983** synapses — runs as a frozen spiking neural network on a GPU. A small linear
readout on top of it is trained to turn a room temperature reading into a light colour, and Home
Assistant is on both ends.

The large data tables are **not committed**. Get them with:

```bash
.venv/bin/python tools/fetch_data.py     # downloads + byte-verifies, ~140 MB
```

See [`data.md`](data.md) for exactly what is fetched and from where.

## Guides

| Document | What is in it |
|---|---|
| [`architecture.md`](architecture.md) | How the pieces fit together, the LIF equations, and the integrator bug that caused a long "recurrent gain" wild goose chase |
| [`live-view.md`](live-view.md) | The dashboard, the control loop, the WebSocket wire protocol, and every configuration flag |
| [`data.md`](data.md) | How to fetch the data, schemas, provenance, parsing gotchas |
| [`engine.md`](engine.md) | Performance, power draw, and the plan for a real-time engine |
| [`wiring.md`](wiring.md) | Which Home Assistant entity drives which fly sensory pathway, and why the order of those rules matters |
| [`vision.md`](vision.md) | The camera → visual-column design. **Designed only, not built.** |
| [`roadmap.md`](roadmap.md) | What to build next, and an honest assessment of what a fly brain is good for |
| [`ha-inventory.md`](ha-inventory.md) | What a real house's Home Assistant actually exposes, and which plausible ideas that rules out |
| [`licensing.md`](licensing.md) | **Read before shipping anything.** Code is MIT; the connectome data is CC BY-NC 4.0 |

## Research

[`research/`](research/README.md) is the record of the work, including what went wrong:

| Document | What is in it |
|---|---|
| [`research/README.md`](research/README.md) | Index and the method behind the research |
| [`research/asset-inventory.md`](research/asset-inventory.md) | Every dataset, model and prior project found, with licences |
| [`research/simulation-backends.md`](research/simulation-backends.md) | The engine survey: Brian2, GeNN, NEST, NeuronGPU, flyvis and why PyTorch won |
| [`research/dead-ends.md`](research/dead-ends.md) | The approaches that looked promising and failed, and what they cost |

## Images

Screenshots in [`images/`](images/) are captured from the dashboard running against the built-in
**mock home**, never a real installation — so no real entity ids, addresses or house data appear
in them.
