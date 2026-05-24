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
| **Lookback** | log-stride snapshots (1m, 5m, 30m, 4h...) | catch the medium-term recurring stuff |
| **Landmarks** | exemplar wavefronts | always-considered anchors (chosen by gate) |
| **Salience gate** | optional `SalienceGate` impl | reshape weights with external signal |

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                 kannaka-attention                    │
├──────────────────┬────────────────┬──────────────────┤
│  Recency         │  Lookback      │  Landmarks       │
│  · ring buffer   │  · log-stride  │  · exemplar set  │
│  · O(1) push     │  · age buckets │  · gated by Φ    │
├──────────────────┼────────────────┼──────────────────┤
│  Beam composer                                       │
│  · merge with dedupe                                 │
│  · cap at top_k                                      │
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

Wired into `kannaka attention serve` — consumes `KANNAKA.attention.eye` events from the bus and exports the current beam to `/tmp/kannaka-attention-beam.json` so any consumer (TUI, observatory, recall path) can read which IDs are "in focus" right now.

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
