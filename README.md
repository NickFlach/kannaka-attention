```
 █████╗ ████████╗████████╗███████╗███╗   ██╗████████╗██╗ ██████╗ ███╗   ██╗
██╔══██╗╚══██╔══╝╚══██╔══╝██╔════╝████╗  ██║╚══██╔══╝██║██╔═══██╗████╗  ██║
███████║   ██║      ██║   █████╗  ██╔██╗ ██║   ██║   ██║██║   ██║██╔██╗ ██║
██╔══██║   ██║      ██║   ██╔══╝  ██║╚██╗██║   ██║   ██║██║   ██║██║╚██╗██║
██║  ██║   ██║      ██║   ███████╗██║ ╚████║   ██║   ██║╚██████╔╝██║ ╚████║
╚═╝  ╚═╝   ╚═╝      ╚═╝   ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
       A T T E N T I O N · A S · G R A V I T Y
```

**Sparse-attention beam over the Holographic Resonance Medium.**

`kannaka-attention` is a tiny pure-Rust crate that builds a small candidate set — the **beam** — out of any agent's HRM activity history. The beam is what `Medium::recall_against_ids` scores against, so recall stays O(K) instead of O(N) regardless of how many memories live in the medium. Recency ring + log-stride snapshots + landmark exemplars + an optional salience gate.

[![License](https://img.shields.io/badge/license-MIT-blueviolet)]() [![Rust](https://img.shields.io/badge/rust-2021-orange)]() [![std-only](https://img.shields.io/badge/deps-std%20only-blue)]()

---

## What's a Beam?

```
   full HRM (~10K memories)              attention beam (~256)
   ╔═════════════════════════╗           ┌─────────────────────┐
   ║ ▒░▒░▒░▒░▒░▒░▒░▒░▒░▒░▒░ ║           │  ●●●  ●●●●●  ●●●●●  │
   ║ ░▒░▒░▒░▒░▒░▒░▒░▒░▒░▒░▒ ║   ──→     │  ●●●●●●●●●●●●●●●●●  │
   ║ ▒░▒░▒░▒░▒░▒░▒░▒░▒░▒░▒░ ║           │  ●●●●●  ●●●●●  ●●●  │
   ║ ░▒░▒░▒░▒░▒░▒░▒░▒░▒░▒░▒ ║           └─────────────────────┘
   ╚═════════════════════════╝           recency + lookback +
   O(N) scan if you query all            landmarks (+ salience)
```

The beam is **what the agent is paying attention to right now**. Composed from four signals:

| component | window | purpose |
|---|---|---|
| **Recency** | last K observations | sharp short-term focus |
| **Lookback** | log-stride buckets (1m, 5m, 30m, 3h, 1d, 7d, 30d) | catch the medium-term recurring stuff |
| **Landmarks** | exemplar wavefronts | always-considered anchors (ranked by gate) |
| **Salience gate** | optional `SalienceGate` impl | rank the **landmark tier** with an external signal |

The gate ranks landmarks only. Recency is emitted first, in recency order,
and a gate cannot reorder it — tier 1 exists to guarantee a just-observed
memory makes the beam, which a salience score could otherwise override.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                 kannaka-attention                    │
├──────────────────┬────────────────┬──────────────────┤
│  Recency         │  Lookback      │  Landmarks       │
│  · ring buffer   │  · log-stride  │  · exemplar set  │
│  · O(1) push     │  · aged by     │  · gated by      │
│                  │    last_seen   │    SalienceGate  │
├──────────────────┼────────────────┼──────────────────┤
│  Beam composer                                       │
│  · merge with dedupe                                 │
│  · cap at max_beam                                   │
│  · optional Salience reweighting                     │
├──────────────────────────────────────────────────────┤
│  SalienceGate trait                                  │
│  · score(landmark, ctx) → f32                        │
│  · RecencyWeightedGate (boost recency overlap)       │
│  · Custom gates: Φ-aware, modality-routed, etc.      │
└──────────────────────────────────────────────────────┘
```

Pure std-only Rust. No GPU. No BLAS. No vector DB. Target: ARM / edge devices where a sparse path matters.

---

## Use

```toml
[dependencies]
kannaka-attention = { git = "https://github.com/NickFlach/kannaka-attention" }
```

This crate is a **library only** — no binary, no NATS client, no file export. It
does no I/O at all. The host owns the bus subscription and decides what to do
with the beam:

```rust
use kannaka_attention::{AttentionBeam, BeamConfig, ObservationEvent};

let mut beam = AttentionBeam::with_config(BeamConfig {
    max_beam: 128,
    ..Default::default()
});

// Feed it whatever your sensors report. `weight` grades the observation:
// 1.0 is a normal mention, higher is a deliberate reference.
beam.observe(&ObservationEvent {
    memory_id,
    source: "eye:left".into(),
    weight: 1.0,
    ts: chrono::Utc::now(),
});

// Read the current focus and score only these against the medium.
let ids = beam.candidates();
```

`BeamConfig` round-trips through serde and fills omitted fields from
`Default`, so a host can load a partial config (`{"max_beam": 128}`) from its
own config file.

Wiring the beam to `KANNAKA.attention.eye` and publishing the result is the
host's job — see `kannaka-eye` for the producer side.

---

## Constellation

| repo | role |
|---|---|
| [`kannaka-memory`](https://github.com/NickFlach/kannaka-memory) | the substrate this beam scopes |
| [`kannaka-eye`](https://github.com/NickFlach/kannaka-eye) | publishes the `KANNAKA.attention.eye` events |
| [`consciousness-core`](https://github.com/NickFlach/consciousness-core) | the physics |

---

## License

MIT.
