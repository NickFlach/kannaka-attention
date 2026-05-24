//! Salience gate — RuVector's FastGRNN analog.
//!
//! RuVector's gated path runs a tiny RNN over the keys to pick K dynamic
//! long-range tokens per query. Our equivalent operates over the landmark
//! set: given the current beam state (recency + lookback) and a candidate
//! landmark, the gate emits a float salience that re-orders / re-weights
//! the landmark before it enters the candidate set.
//!
//! v1 ships two implementations:
//!
//! - [`StaticGate`] — uses only `Landmark.weight`. No-op gate; current
//!   behavior. The fallback when no gate is registered on the beam.
//! - [`RecencyWeightedGate`] — boosts landmarks whose `id` appears in the
//!   recency ring. The intuition: a cluster whose exemplar was *just*
//!   observed should outrank a cluster whose exemplar hasn't moved in
//!   weeks. Tunable: `recency_boost` (default 2.0×), `observation_decay`
//!   (default 0.95 per non-observation tick).
//!
//! v2 will add a `LearnedGate` that loads small weights from disk
//! (FastGRNN or 1-layer GRU) — trained from dream-report deltas. The
//! trait is the architectural commitment that v2 can slot in without
//! touching the beam's plumbing.

use std::collections::HashMap;
use uuid::Uuid;

use crate::landmarks::Landmark;

/// Context the beam hands a gate when scoring a landmark.
#[derive(Debug, Clone)]
pub struct GateContext<'a> {
    /// Memory ids currently in the recency ring (most-recent first).
    pub recency: &'a [Uuid],
    /// Memory ids currently in the log-stride lookback (any age).
    pub lookback: &'a [Uuid],
    /// Total observations the beam has processed (monotonic).
    pub observations: u64,
}

/// A scoring function over landmarks. Implementations must be pure with
/// respect to the gate's own state — the beam may call `score` many times
/// per snapshot and expects deterministic results within one snapshot
/// cycle.
pub trait SalienceGate: std::fmt::Debug + Send + Sync {
    /// Return the salience score for `landmark` given the current beam
    /// context. Higher = include first / weight more heavily.
    fn score(&self, landmark: &Landmark, ctx: &GateContext<'_>) -> f32;

    /// Optional name for telemetry / observatory display.
    fn name(&self) -> &'static str {
        "unnamed"
    }
}

/// The no-op gate. Returns `landmark.weight` unchanged. This is the
/// fallback when the beam has no gate registered.
#[derive(Debug, Clone, Default)]
pub struct StaticGate;

impl SalienceGate for StaticGate {
    fn score(&self, landmark: &Landmark, _ctx: &GateContext<'_>) -> f32 {
        landmark.weight
    }
    fn name(&self) -> &'static str {
        "static"
    }
}

/// Recency-weighted gate. A landmark whose exemplar id is currently in
/// the recency ring gets a `recency_boost` multiplier; a landmark whose
/// id is in the lookback (but not recency) gets a smaller boost; cold
/// landmarks get `landmark.weight` as-is.
///
/// This is the "warm landmarks rise" rule, and it's what makes the gate
/// produce visibly different behavior from `StaticGate` — without
/// needing any learned weights.
#[derive(Debug, Clone)]
pub struct RecencyWeightedGate {
    /// Multiplier applied when the landmark's id is in the recency ring.
    /// Default 2.0×.
    pub recency_boost: f32,
    /// Multiplier applied when the landmark's id is in the lookback but
    /// NOT in the recency ring. Default 1.3×.
    pub lookback_boost: f32,
}

impl Default for RecencyWeightedGate {
    fn default() -> Self {
        Self {
            recency_boost: 2.0,
            lookback_boost: 1.3,
        }
    }
}

impl SalienceGate for RecencyWeightedGate {
    fn score(&self, landmark: &Landmark, ctx: &GateContext<'_>) -> f32 {
        // Recency check — small N (typically 128), linear scan is fine.
        if ctx.recency.iter().any(|x| *x == landmark.id) {
            return landmark.weight * self.recency_boost;
        }
        if ctx.lookback.iter().any(|x| *x == landmark.id) {
            return landmark.weight * self.lookback_boost;
        }
        landmark.weight
    }
    fn name(&self) -> &'static str {
        "recency-weighted"
    }
}

/// Score every landmark in `set` using `gate` against `ctx`, returning
/// (id, score) sorted descending by score. Helper for callers that want
/// to rank a snapshot once.
pub fn rank_landmarks(
    set: &[(Uuid, &Landmark)],
    gate: &dyn SalienceGate,
    ctx: &GateContext<'_>,
) -> Vec<(Uuid, f32)> {
    let mut scored: Vec<(Uuid, f32)> = set
        .iter()
        .map(|(id, l)| (*id, gate.score(l, ctx)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
}

/// Snapshot helper that wraps a `LandmarkSet`-like map + an optional
/// gate. Internal use; the beam composes this through `AttentionBeam`.
pub(crate) fn snapshot_with_gate(
    by_cluster: &HashMap<String, Landmark>,
    gate: Option<&dyn SalienceGate>,
    ctx: &GateContext<'_>,
) -> Vec<Uuid> {
    if by_cluster.is_empty() {
        return Vec::new();
    }
    let mut entries: Vec<(Uuid, f32, &Landmark)> = by_cluster
        .values()
        .map(|l| {
            let s = if let Some(g) = gate {
                g.score(l, ctx)
            } else {
                l.weight
            };
            (l.id, s, l)
        })
        .collect();
    entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    entries.into_iter().map(|(id, _, _)| id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn landmark(id: Uuid, w: f32) -> Landmark {
        Landmark {
            id,
            cluster_label: format!("c-{}", &id.to_string()[..4]),
            weight: w,
        }
    }

    #[test]
    fn static_gate_returns_weight_unchanged() {
        let id = Uuid::new_v4();
        let l = landmark(id, 0.7);
        let ctx = GateContext {
            recency: &[],
            lookback: &[],
            observations: 0,
        };
        assert_eq!(StaticGate.score(&l, &ctx), 0.7);
    }

    #[test]
    fn recency_gate_boosts_when_id_in_recency_ring() {
        let id = Uuid::new_v4();
        let l = landmark(id, 1.0);
        let ctx = GateContext {
            recency: &[id],
            lookback: &[],
            observations: 1,
        };
        let g = RecencyWeightedGate::default();
        let s = g.score(&l, &ctx);
        assert!((s - 2.0).abs() < 1e-6, "expected ~2.0 boost, got {}", s);
    }

    #[test]
    fn recency_gate_lookback_only_lighter_boost() {
        let id = Uuid::new_v4();
        let l = landmark(id, 1.0);
        let ctx = GateContext {
            recency: &[],
            lookback: &[id],
            observations: 5,
        };
        let g = RecencyWeightedGate::default();
        let s = g.score(&l, &ctx);
        assert!((s - 1.3).abs() < 1e-6, "expected ~1.3 boost, got {}", s);
    }

    #[test]
    fn recency_gate_cold_returns_weight() {
        let id = Uuid::new_v4();
        let other = Uuid::new_v4();
        let l = landmark(id, 0.5);
        let ctx = GateContext {
            recency: &[other],
            lookback: &[],
            observations: 0,
        };
        let g = RecencyWeightedGate::default();
        assert_eq!(g.score(&l, &ctx), 0.5);
    }
}
