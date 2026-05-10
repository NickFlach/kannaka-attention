//! The attention beam. Composes the three primitives into a single owner
//! that the host (kannaka-memory CLI, kannaka-eye daemon, etc.) feeds with
//! observations and reads candidate sets from.
//!
//! The beam is gravity. Memories that get observed pull their cluster in,
//! and their cluster pulls the beam. The beam emits a deduped, ranked set
//! of memory IDs that should be scored on the next recall.

use chrono::Utc;
use std::collections::HashSet;
use uuid::Uuid;

use crate::landmarks::{Landmark, LandmarkSet};
use crate::lookback::LogStrideLookback;
use crate::recency::RecencyRing;
use crate::ObservationEvent;

/// Configuration knobs for the beam. Defaults are tuned for an ARM A1 / Pi 5
/// class box with a medium of 10k–1M memories.
#[derive(Debug, Clone)]
pub struct BeamConfig {
    /// Recency ring capacity. 128 is enough to span a small conversation;
    /// raise to 512 for long-running narrative work.
    pub recency_capacity: usize,
    /// Max landmarks included per beam. Tight = stay-on-topic, wide = let
    /// cross-cluster ideas in. 16 is a reasonable default.
    pub max_landmarks: usize,
    /// Max lookback entries included per beam. Capped because the lookback
    /// can carry up to ~64 ids; we usually only need the top slice.
    pub max_lookback: usize,
    /// Total beam size cap (the K in O(K) recall). Hard ceiling on candidate
    /// set size — even if all three primitives are full, the beam is
    /// truncated to this. 256 is a sweet spot: large enough to capture
    /// associative recall, small enough that scoring is trivially fast.
    pub max_beam: usize,
}

impl Default for BeamConfig {
    fn default() -> Self {
        Self {
            recency_capacity: 128,
            max_landmarks: 16,
            max_lookback: 32,
            max_beam: 256,
        }
    }
}

/// The composed attention beam. One per process. Single-owner (wrap in your
/// own lock if you need multi-producer).
#[derive(Debug, Clone)]
pub struct AttentionBeam {
    config: BeamConfig,
    recency: RecencyRing,
    lookback: LogStrideLookback,
    landmarks: LandmarkSet,
    // Counter for periodic rebalance — every Nth observation triggers a
    // lookback rebalance so aging is reflected without doing the work on
    // every push.
    obs_count: u64,
}

impl AttentionBeam {
    /// Build with default config.
    pub fn new() -> Self { Self::with_config(BeamConfig::default()) }

    /// Build with explicit config.
    pub fn with_config(config: BeamConfig) -> Self {
        let recency = RecencyRing::new(config.recency_capacity);
        Self {
            recency,
            lookback: LogStrideLookback::new(),
            landmarks: LandmarkSet::new(),
            obs_count: 0,
            config,
        }
    }

    /// Observe a memory. Pushes into recency + lookback. The `weight` field
    /// of the event reserves room for a future salience gate but is not yet
    /// used in v1 (every observation lands with the same weight).
    pub fn observe(&mut self, ev: &ObservationEvent) {
        self.recency.push(ev.memory_id);
        self.lookback.observe(ev.memory_id, ev.ts);
        self.obs_count = self.obs_count.wrapping_add(1);
        // Every 32 observations, rebalance the lookback so age buckets
        // stay accurate. Cheap (O(total entries)) and bounded since the
        // lookback caps total entries at ~64.
        if self.obs_count % 32 == 0 {
            self.lookback.rebalance(Utc::now());
        }
    }

    /// Convenience: observe by id with `Utc::now()` and source label.
    pub fn observe_now(&mut self, id: Uuid, source: impl Into<String>) {
        let ev = ObservationEvent::now(id, source);
        self.observe(&ev);
    }

    /// Register/refresh a landmark. Typically fed from a NATS subscriber on
    /// `KANNAKA.exemplar.>` so the beam always knows the current exemplar
    /// per cluster.
    pub fn upsert_landmark(&mut self, l: Landmark) { self.landmarks.upsert(l); }

    /// Remove a landmark by cluster label.
    pub fn drop_landmark(&mut self, cluster_label: &str) {
        self.landmarks.remove(cluster_label);
    }

    /// Emit the current candidate set. Deduped, ordered by tier:
    /// 1. Recency (most-recent first) — these are always in.
    /// 2. Lookback (newer buckets first).
    /// 3. Landmarks (highest weight first).
    ///
    /// Truncated to `config.max_beam`. Empty if no observations have ever
    /// landed.
    pub fn candidates(&self) -> Vec<Uuid> {
        let mut out = Vec::with_capacity(self.config.max_beam);
        let mut seen = HashSet::with_capacity(self.config.max_beam);

        for id in self.recency.snapshot() {
            if seen.insert(id) {
                out.push(id);
                if out.len() >= self.config.max_beam { return out; }
            }
        }
        let mut taken = 0usize;
        for id in self.lookback.snapshot() {
            if taken >= self.config.max_lookback { break; }
            if seen.insert(id) {
                out.push(id);
                taken += 1;
                if out.len() >= self.config.max_beam { return out; }
            }
        }
        let mut taken = 0usize;
        for id in self.landmarks.snapshot() {
            if taken >= self.config.max_landmarks { break; }
            if seen.insert(id) {
                out.push(id);
                taken += 1;
                if out.len() >= self.config.max_beam { return out; }
            }
        }
        out
    }

    /// Stats for instrumentation / observatory display.
    pub fn stats(&self) -> BeamStats {
        BeamStats {
            recency_len: self.recency.len(),
            lookback_len: self.lookback.len(),
            landmarks_len: self.landmarks.len(),
            beam_size: self.candidates().len(),
            observations: self.obs_count,
        }
    }
}

impl Default for AttentionBeam {
    fn default() -> Self { Self::new() }
}

/// Snapshot of beam internals for telemetry. Cheap to compute.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct BeamStats {
    /// Current recency ring population.
    pub recency_len: usize,
    /// Current lookback population (sum across age buckets).
    pub lookback_len: usize,
    /// Current landmark count.
    pub landmarks_len: usize,
    /// Size of the candidate set this cycle.
    pub beam_size: usize,
    /// Total observations ever fed to the beam (wraps).
    pub observations: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_beam_returns_no_candidates() {
        let b = AttentionBeam::new();
        assert!(b.candidates().is_empty());
    }

    #[test]
    fn observation_lands_in_recency_first_tier() {
        let mut b = AttentionBeam::new();
        let id = Uuid::new_v4();
        b.observe_now(id, "eye:left");
        let cands = b.candidates();
        assert!(!cands.is_empty());
        assert_eq!(cands[0], id);
    }

    #[test]
    fn landmark_only_no_observations_still_emits() {
        let mut b = AttentionBeam::new();
        let id = Uuid::new_v4();
        b.upsert_landmark(Landmark { id, cluster_label: "philosophy".into(), weight: 1.0 });
        let cands = b.candidates();
        assert_eq!(cands, vec![id]);
    }

    #[test]
    fn dedup_across_tiers() {
        let mut b = AttentionBeam::new();
        let id = Uuid::new_v4();
        b.observe_now(id, "eye:left");
        b.upsert_landmark(Landmark { id, cluster_label: "philosophy".into(), weight: 1.0 });
        let cands = b.candidates();
        assert_eq!(cands.iter().filter(|c| **c == id).count(), 1);
    }

    #[test]
    fn beam_truncated_at_max_beam() {
        let mut b = AttentionBeam::with_config(BeamConfig {
            recency_capacity: 1000,
            max_landmarks: 0,
            max_lookback: 0,
            max_beam: 5,
        });
        for _ in 0..50 {
            b.observe_now(Uuid::new_v4(), "test");
        }
        assert_eq!(b.candidates().len(), 5);
    }
}
