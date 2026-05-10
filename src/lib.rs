//! Attention-as-gravity over the Holographic Resonance Medium.
//!
//! This crate is the sparse-attention layer between the sensors (eye, ear) and
//! the medium. It produces an **attention beam** — a small set of memory IDs
//! that should be scored on the next recall — derived from three sources, the
//! same primitives RuVector's `ruvllm_sparse_attention` uses for LLM
//! attention sparsity, mapped onto a wave-interference store:
//!
//! 1. **Recency ring** (local window): a FIFO of the last N observed/recalled
//!    memory IDs. This is the "what's hot right now" channel.
//! 2. **Log-stride snapshots** (hierarchical lookback): cluster
//!    representatives at exponentially-spaced ages (1m, 5m, 30m, 3h, 1d, 7d,
//!    30d). Memory holds onto a sparse summary of its past.
//! 3. **Landmarks** (block summaries): cluster exemplars supplied by the host
//!    medium (kannaka-memory already publishes these on
//!    `KANNAKA.exemplar.<agent>.<cluster>`). These are the "everything-feeds-
//!    through-here" attractors.
//!
//! A future v2 adds a small **salience gate** (FastGRNN or single-layer MLP)
//! to pick which landmarks/snapshots get included this cycle, conditional on
//! the current query. v1 uses static union with light recency weighting.
//!
//! The output is a deduped beam of <= K memory IDs. The host calls
//! `Medium::recall_against_ids(&beam, query, ...)` which scores only that
//! subset — O(K) instead of O(N), independent of medium size.
//!
//! # Hardware envelope
//!
//! Pure Rust, no GPU / BLAS / FAISS / vector-DB dependencies. The whole beam
//! fits in a few KB at K=512. Designed to run on the same humble hardware
//! RuVector targets — Pi Zero 2W (Cortex-A53, 512 MB) is the spec, comfortably
//! within reach.
//!
//! # Threading model
//!
//! The beam is single-owner. The eye + ear publish observations into it from
//! one place (typically a NATS subscriber loop). Recall consumers borrow the
//! current beam snapshot immutably. If multi-producer is needed later, wrap
//! `AttentionBeam` in your own `Mutex`/`RwLock` — this crate does not assume
//! a concurrency primitive.

#![deny(missing_docs)]

use chrono::{DateTime, Utc};
use uuid::Uuid;

pub mod recency;
pub mod lookback;
pub mod landmarks;
pub mod beam;

pub use beam::{AttentionBeam, BeamConfig};
pub use recency::RecencyRing;
pub use lookback::LogStrideLookback;
pub use landmarks::LandmarkSet;

/// An attention event from a sensor (eye, ear, recall, dream, etc.). Drives
/// gravity in the beam — repeated events on the same memory pull harder.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ObservationEvent {
    /// The memory this event is about (must already exist in the medium).
    pub memory_id: Uuid,
    /// Where the event came from. Free-form so new senses can join without an
    /// enum bump; canonical values: "eye:left", "eye:right", "ear", "recall",
    /// "dream", "swarm".
    pub source: String,
    /// Strength multiplier. 1.0 = a normal observation. Higher = a deliberate
    /// signal (e.g. user explicitly references a memory). Lower than 1 = a
    /// background ambient mention. Clamped to [0.01, 10.0] in the beam.
    pub weight: f32,
    /// Wall-clock time of the event. The beam uses this for log-stride
    /// bucketing — passing `Utc::now()` is the common case.
    pub ts: DateTime<Utc>,
}

impl ObservationEvent {
    /// Build an event from the common case: a memory id + a source label.
    /// Uses `now()` for the timestamp and weight = 1.0.
    pub fn now(memory_id: Uuid, source: impl Into<String>) -> Self {
        Self { memory_id, source: source.into(), weight: 1.0, ts: Utc::now() }
    }
}
