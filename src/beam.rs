//! The attention beam. Composes the three primitives into a single owner
//! that the host (kannaka-memory CLI, kannaka-eye daemon, etc.) feeds with
//! observations and reads candidate sets from.
//!
//! The beam is gravity. Memories that get observed pull their cluster in,
//! and their cluster pulls the beam. The beam emits a deduped, ranked set
//! of memory IDs that should be scored on the next recall.

use std::collections::HashSet;
use uuid::Uuid;

use crate::landmarks::{Landmark, LandmarkSet};
use crate::lookback::LogStrideLookback;
use crate::recency::RecencyRing;
use crate::salience::{snapshot_with_gate, GateContext, SalienceGate};
use crate::ObservationEvent;

/// Configuration knobs for the beam. Defaults are tuned for an ARM A1 / Pi 5
/// class box with a medium of 10k–1M memories.
///
/// `#[serde(default)]` is what makes a partial config usable: a payload that
/// sets only `max_beam` fills the rest from [`BeamConfig::default`] instead
/// of failing on the first absent field. Without it, every consumer wanting
/// to override one knob had to restate all four out of band. (#16)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
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
    /// Restrict the landmark tier to exemplars currently present in recency
    /// or lookback. Default `false`.
    ///
    /// Off (the default), landmarks are always-considered anchors: every
    /// registered exemplar is eligible whether or not it has ever been
    /// observed, which is what lets a freshly started process emit a useful
    /// beam before any observation has landed.
    ///
    /// On, the beam reflects *attended* state rather than catalog state —
    /// nothing appears that was not actually seen. Consumers that treat the
    /// beam as "what is the agent looking at right now" want this; be aware
    /// it makes the beam empty until observations accumulate. (#5)
    pub observed_only: bool,
    /// Key observation identity on `(source, memory_id)` rather than
    /// `memory_id` alone. Default `false`.
    ///
    /// Off (the default), two sensors observing the same memory collapse
    /// onto one entry in recency and lookback.
    ///
    /// On, they hold independent slots, so a chiral mirror pair
    /// (`eye:left` / `eye:right`) each pull on the beam in their own right.
    /// `kannaka-eye` publishes `source` and `hemisphere` precisely so a
    /// subscriber can choose — its own docs note that subscribers "can fold
    /// both into the same beam, or treat them as separate attention
    /// sources".
    ///
    /// Costs population: the lookback's ~64-entry ceiling is per key, so
    /// enabling this multiplies worst-case retained entries by the number
    /// of active sources. That ceiling underwrites the crate's edge-device
    /// envelope, which is why this is opt-in. (#3)
    pub per_source: bool,
}

impl Default for BeamConfig {
    fn default() -> Self {
        Self {
            recency_capacity: 128,
            max_landmarks: 16,
            max_lookback: 32,
            max_beam: 256,
            observed_only: false,
            per_source: false,
        }
    }
}

/// The composed attention beam. One per process. Single-owner (wrap in your
/// own lock if you need multi-producer).
#[derive(Debug)]
pub struct AttentionBeam {
    config: BeamConfig,
    recency: RecencyRing,
    lookback: LogStrideLookback,
    landmarks: LandmarkSet,
    /// Optional salience gate. When set, ranks landmarks during
    /// `candidates()`. When None, landmarks fall back to weight-sorted.
    gate: Option<Box<dyn SalienceGate>>,
    // Counter for periodic rebalance — every Nth observation triggers a
    // lookback rebalance so aging is reflected without doing the work on
    // every push.
    obs_count: u64,
}

impl AttentionBeam {
    /// Build with default config.
    pub fn new() -> Self {
        Self::with_config(BeamConfig::default())
    }

    /// Build with explicit config.
    pub fn with_config(config: BeamConfig) -> Self {
        let recency = RecencyRing::new(config.recency_capacity);
        Self {
            recency,
            lookback: LogStrideLookback::new(),
            landmarks: LandmarkSet::new(),
            gate: None,
            obs_count: 0,
            config,
        }
    }

    /// Install a salience gate. Replaces any previously-installed gate.
    /// Call with `RecencyWeightedGate::default()` for the default
    /// learnable-shape-but-static-weights implementation, or roll your
    /// own by implementing `SalienceGate`.
    pub fn set_gate(&mut self, gate: Box<dyn SalienceGate>) {
        self.gate = Some(gate);
    }

    /// Name of the active gate, or "none". Surfaced in stats.
    pub fn gate_name(&self) -> &'static str {
        self.gate.as_ref().map(|g| g.name()).unwrap_or("none")
    }

    /// Observe a memory. Pushes into recency + lookback.
    ///
    /// `ev.weight` is honored: it is clamped to the documented
    /// `[0.01, 10.0]` range and accumulated as the memory's standing in the
    /// lookback, so a deliberate reference outranks an ambient mention
    /// rather than counting the same. It used to be dropped on the floor,
    /// which made the documented clamp contract false and flattened
    /// graded attention to binary observed/not-observed. (#13)
    ///
    /// Bucket placement uses `ev.ts`, the event's own timestamp, not wall
    /// clock — so replayed and backdated events land where they belong and
    /// callers can write exact-time fixtures. (#25)
    /// Recency ordering also uses `ev.ts`, so a replayed batch produces the
    /// same beam as a live one. Arrival-order insertion meant a backdated
    /// event took the front of the ring regardless of when it happened,
    /// which left tier 1 disagreeing with the (already event-timed)
    /// lookback under replay. (#4)
    ///
    /// When `config.per_source` is set, both structures key on
    /// `(ev.source, ev.memory_id)`, so a mirror pair pulls independently
    /// rather than collapsing onto one entry. (#3)
    pub fn observe(&mut self, ev: &ObservationEvent) {
        if self.config.per_source {
            self.recency.push_keyed(ev.memory_id, &ev.source, ev.ts);
            self.lookback
                .observe_keyed(ev.memory_id, &ev.source, ev.ts, ev.weight);
        } else {
            self.recency.push_at(ev.memory_id, ev.ts);
            self.lookback
                .observe_weighted(ev.memory_id, ev.ts, ev.weight);
        }
        self.obs_count = self.obs_count.wrapping_add(1);
    }

    /// Convenience: observe by id with `Utc::now()` and source label.
    pub fn observe_now(&mut self, id: Uuid, source: impl Into<String>) {
        let ev = ObservationEvent::now(id, source);
        self.observe(&ev);
    }

    /// Register/refresh a landmark. Typically fed from a NATS subscriber on
    /// `KANNAKA.exemplar.>` so the beam always knows the current exemplar
    /// per cluster.
    pub fn upsert_landmark(&mut self, l: Landmark) {
        self.landmarks.upsert(l);
    }

    /// Remove a landmark by cluster label.
    pub fn drop_landmark(&mut self, cluster_label: &str) {
        self.landmarks.remove(cluster_label);
    }

    /// Emit the current candidate set. Deduped, ordered by tier:
    /// 1. Recency (most-recent first) — always in.
    /// 2. Lookback (newer buckets first).
    /// 3. Landmarks (gate-ranked if a gate is installed, weight-sorted
    ///    otherwise).
    ///
    /// Truncated to `config.max_beam`. Empty if no observations have ever
    /// landed.
    pub fn candidates(&self) -> Vec<Uuid> {
        let mut out = Vec::with_capacity(self.config.max_beam);
        let mut seen = HashSet::with_capacity(self.config.max_beam);

        // Check the ceiling BEFORE pushing, not after. The post-push check
        // let `max_beam = 0` still emit one id — a hard budget that can be
        // overrun by one is not a budget, and downstream scheduling and
        // backpressure assume it holds exactly. (#1)
        if self.config.max_beam == 0 {
            return out;
        }

        let recency_snap = self.recency.snapshot();
        let lookback_snap = self.lookback.snapshot();

        for id in &recency_snap {
            if out.len() >= self.config.max_beam {
                return out;
            }
            if seen.insert(*id) {
                out.push(*id);
            }
        }
        // `take` bounds the lookback POSITIONS INSPECTED, not the unique
        // ids admitted. The old loop only counted successful inserts, so an
        // entry that overlapped recency cost nothing and the walk continued
        // deeper into history until it had admitted `max_lookback` *new*
        // ids — turning a locality knob into "reach as far back as
        // needed". (#21)
        for id in lookback_snap.iter().take(self.config.max_lookback) {
            if out.len() >= self.config.max_beam {
                return out;
            }
            if seen.insert(*id) {
                out.push(*id);
            }
        }
        // Landmark ordering goes through the gate when one is installed.
        // Without a gate, fall back to weight-sorted (LandmarkSet::snapshot).
        let landmark_order: Vec<Uuid> = if let Some(ref g) = self.gate {
            let ctx = GateContext {
                recency: &recency_snap,
                lookback: &lookback_snap,
                observations: self.obs_count,
            };
            snapshot_with_gate(&self.landmarks.by_cluster, Some(g.as_ref()), &ctx)
        } else {
            self.landmarks.snapshot()
        };
        // `observed_only` narrows the landmark tier to exemplars actually
        // present in an active window, so the beam reports attended state
        // rather than catalog state. Off by default — see BeamConfig. (#5)
        let landmark_order: Vec<Uuid> = if self.config.observed_only {
            landmark_order
                .into_iter()
                .filter(|id| self.recency.contains(id) || self.lookback.contains(id))
                .collect()
        } else {
            landmark_order
        };
        // Same positions-inspected accounting as the lookback tier (#21):
        // a landmark that already arrived via recency still spends its
        // slot, so `max_landmarks` bounds how far down the ranked landmark
        // list the beam reaches.
        for id in landmark_order.into_iter().take(self.config.max_landmarks) {
            if out.len() >= self.config.max_beam {
                return out;
            }
            if seen.insert(id) {
                out.push(id);
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
    fn default() -> Self {
        Self::new()
    }
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
        b.upsert_landmark(Landmark {
            id,
            cluster_label: "philosophy".into(),
            weight: 1.0,
        });
        let cands = b.candidates();
        assert_eq!(cands, vec![id]);
    }

    #[test]
    fn dedup_across_tiers() {
        let mut b = AttentionBeam::new();
        let id = Uuid::new_v4();
        b.observe_now(id, "eye:left");
        b.upsert_landmark(Landmark {
            id,
            cluster_label: "philosophy".into(),
            weight: 1.0,
        });
        let cands = b.candidates();
        assert_eq!(cands.iter().filter(|c| **c == id).count(), 1);
    }

    #[test]
    fn recency_weighted_gate_reorders_landmarks() {
        let mut b = AttentionBeam::with_config(BeamConfig {
            recency_capacity: 4,
            max_landmarks: 8,
            max_lookback: 0,
            max_beam: 16,
            ..Default::default()
        });
        let cold = Uuid::new_v4();
        let warm = Uuid::new_v4();
        b.upsert_landmark(Landmark {
            id: cold,
            cluster_label: "cold".into(),
            weight: 2.0,
        });
        b.upsert_landmark(Landmark {
            id: warm,
            cluster_label: "warm".into(),
            weight: 1.0,
        });
        b.observe_now(warm, "test");
        // Without a gate: cold (weight 2.0) ranks above warm (weight 1.0).
        // The recency ring also contains `warm` so when we read candidates
        // it appears in the recency tier first. To isolate the gate effect
        // on the landmark tier, install the gate and check the landmark
        // ordering after recency dedup.
        b.set_gate(Box::new(crate::salience::RecencyWeightedGate::default()));
        let cands = b.candidates();
        // warm is in recency (first tier), then landmarks tier should put
        // it back too — but dedup removes it. Cold should still appear
        // somewhere; the test just ensures gate doesn't crash and
        // produces a deduped beam.
        assert!(cands.contains(&warm));
        assert!(cands.contains(&cold));
    }

    #[test]
    fn max_beam_zero_emits_nothing() {
        // Regression for #1 — the cap was checked after pushing, so a
        // budget of 0 still produced one candidate.
        let mut b = AttentionBeam::with_config(BeamConfig {
            recency_capacity: 8,
            max_landmarks: 8,
            max_lookback: 8,
            max_beam: 0,
            ..Default::default()
        });
        let id = Uuid::new_v4();
        b.observe_now(id, "test");
        b.upsert_landmark(Landmark {
            id: Uuid::new_v4(),
            cluster_label: "c".into(),
            weight: 5.0,
        });
        assert!(b.candidates().is_empty());
    }

    #[test]
    fn beam_never_exceeds_max_beam_for_any_config() {
        // Property: whatever the tier caps say, the total budget holds.
        for max_beam in 0..12usize {
            let mut b = AttentionBeam::with_config(BeamConfig {
                recency_capacity: 32,
                max_landmarks: 32,
                max_lookback: 32,
                max_beam,
                ..Default::default()
            });
            for i in 0..20 {
                b.observe_now(Uuid::new_v4(), "test");
                b.upsert_landmark(Landmark {
                    id: Uuid::new_v4(),
                    cluster_label: format!("cluster-{i}"),
                    weight: i as f32,
                });
            }
            let c = b.candidates();
            assert!(
                c.len() <= max_beam,
                "max_beam={max_beam} produced {} candidates",
                c.len()
            );
        }
    }

    #[test]
    fn max_lookback_bounds_reach_even_when_recency_overlaps() {
        // Regression for #21 — `taken` only counted successful inserts, so
        // lookback entries already covered by recency were free and the
        // loop reached deeper into history than the budget allowed.
        let mut b = AttentionBeam::with_config(BeamConfig {
            recency_capacity: 2,
            max_landmarks: 0,
            max_lookback: 1,
            max_beam: 16,
            ..Default::default()
        });
        let a = Uuid::new_v4();
        let mid = Uuid::new_v4();
        let c = Uuid::new_v4();
        b.observe_now(a, "test");
        b.observe_now(mid, "test");
        b.observe_now(c, "test");

        let cands = b.candidates();
        // recency holds [c, mid]; the single lookback slot inspected is
        // already covered by recency, so `a` must NOT be resurrected.
        assert!(
            !cands.contains(&a),
            "max_lookback=1 must not reach past the first lookback slot: {cands:?}"
        );
    }

    #[test]
    fn candidate_order_is_deterministic_across_runs() {
        // Regression for #7 — equal-weight landmarks kept HashMap
        // iteration order, which Rust randomises per process.
        let build = || {
            let mut b = AttentionBeam::with_config(BeamConfig {
                recency_capacity: 0,
                max_landmarks: 8,
                max_lookback: 0,
                max_beam: 16,
                ..Default::default()
            });
            // Same weight for all — nothing but the tiebreak to separate them.
            for (n, label) in ["delta", "alpha", "charlie", "bravo"].iter().enumerate() {
                b.upsert_landmark(Landmark {
                    id: Uuid::from_u128(n as u128 + 1),
                    cluster_label: (*label).into(),
                    weight: 1.0,
                });
            }
            b.candidates()
        };
        let first = build();
        assert_eq!(first.len(), 4);
        for _ in 0..16 {
            assert_eq!(first, build(), "equal-weight ordering must be stable");
        }
    }

    #[test]
    fn observation_weight_reaches_the_lookback() {
        // #13 — a deliberate observation should outrank ambient noise.
        // recency is disabled so the ordering under test is the lookback's.
        let mut b = AttentionBeam::with_config(BeamConfig {
            recency_capacity: 0,
            max_landmarks: 0,
            max_lookback: 8,
            max_beam: 16,
            ..Default::default()
        });
        let ambient = Uuid::new_v4();
        let deliberate = Uuid::new_v4();
        let t = chrono::Utc::now();
        b.observe(&ObservationEvent {
            memory_id: ambient,
            source: "ambient".into(),
            weight: 0.05,
            ts: t,
        });
        b.observe(&ObservationEvent {
            memory_id: deliberate,
            source: "user".into(),
            weight: 10.0,
            ts: t,
        });
        let cands = b.candidates();
        assert_eq!(cands.len(), 2);
        // Same bucket, same last_seen — weight is the only separator.
        assert_eq!(
            cands[0], deliberate,
            "the deliberate reference should lead: {cands:?}"
        );
    }

    #[test]
    fn observe_uses_event_timestamp_not_wall_clock() {
        // #25 — bucketing off Utc::now() made replayed/backdated events
        // land in the wrong bucket and made tests unwriteable.
        let mut b = AttentionBeam::with_config(BeamConfig {
            recency_capacity: 0,
            max_landmarks: 0,
            max_lookback: 8,
            max_beam: 16,
            ..Default::default()
        });
        let now = chrono::Utc::now();
        let ancient = Uuid::new_v4();
        let fresh = Uuid::new_v4();
        b.observe(&ObservationEvent {
            memory_id: ancient,
            source: "replay".into(),
            weight: 1.0,
            ts: now - chrono::Duration::days(20),
        });
        b.observe(&ObservationEvent {
            memory_id: fresh,
            source: "live".into(),
            weight: 1.0,
            ts: now,
        });
        // The 20-day-old event must sort into a far older bucket, so the
        // fresh id leads even though it was observed second.
        assert_eq!(b.candidates(), vec![fresh, ancient]);
    }

    #[test]
    fn observed_only_filters_unobserved_landmarks() {
        // #5 — opt-in attended-only beam.
        let cfg = |observed_only| BeamConfig {
            recency_capacity: 8,
            max_landmarks: 8,
            max_lookback: 8,
            max_beam: 16,
            observed_only,
            per_source: false,
        };
        let seen = Uuid::from_u128(1);
        let never = Uuid::from_u128(2);

        let build = |observed_only| {
            let mut b = AttentionBeam::with_config(cfg(observed_only));
            b.upsert_landmark(Landmark {
                id: seen,
                cluster_label: "seen".into(),
                weight: 1.0,
            });
            b.upsert_landmark(Landmark {
                id: never,
                cluster_label: "never".into(),
                weight: 9.0,
            });
            b.observe_now(seen, "eye:left");
            b.candidates()
        };

        // Default: the never-observed high-weight landmark is still an anchor.
        let permissive = build(false);
        assert!(permissive.contains(&never));
        assert!(permissive.contains(&seen));

        // observed_only: catalog state is excluded, attended state remains.
        let strict = build(true);
        assert!(
            !strict.contains(&never),
            "unobserved landmark must be filtered: {strict:?}"
        );
        assert!(
            strict.contains(&seen),
            "observed landmark must survive: {strict:?}"
        );
    }

    #[test]
    fn observed_only_beam_is_empty_before_any_observation() {
        // The documented cost of the flag, pinned so it can't surprise
        // anyone who enables it. (#5)
        let mut b = AttentionBeam::with_config(BeamConfig {
            observed_only: true,
            ..Default::default()
        });
        b.upsert_landmark(Landmark {
            id: Uuid::from_u128(7),
            cluster_label: "cold".into(),
            weight: 5.0,
        });
        assert!(b.candidates().is_empty());
    }

    #[test]
    fn per_source_lets_a_mirror_pair_pull_independently() {
        // #3 — left/right eye observing the same memory.
        let cfg = |per_source| BeamConfig {
            recency_capacity: 8,
            max_landmarks: 0,
            max_lookback: 8,
            max_beam: 16,
            observed_only: false,
            per_source,
        };
        let id = Uuid::from_u128(42);
        let t = chrono::Utc::now();
        let build = |per_source| {
            let mut b = AttentionBeam::with_config(cfg(per_source));
            for (n, source) in ["eye:left", "eye:right"].iter().enumerate() {
                b.observe(&ObservationEvent {
                    memory_id: id,
                    source: (*source).into(),
                    weight: 1.0,
                    ts: t + chrono::Duration::seconds(n as i64),
                });
            }
            b.stats()
        };

        // Collapsed (default): one entry in each structure.
        let collapsed = build(false);
        assert_eq!(collapsed.recency_len, 1);
        assert_eq!(collapsed.lookback_len, 1);

        // Per-source: each hemisphere holds its own.
        let split = build(true);
        assert_eq!(split.recency_len, 2, "left and right are distinct entries");
        assert_eq!(split.lookback_len, 2);

        // Either way the emitted beam is deduped by memory id.
        let mut b = AttentionBeam::with_config(cfg(true));
        for source in ["eye:left", "eye:right"] {
            b.observe(&ObservationEvent {
                memory_id: id,
                source: source.into(),
                weight: 1.0,
                ts: t,
            });
        }
        assert_eq!(b.candidates(), vec![id], "candidates are still deduped");
    }

    #[test]
    fn recency_tier_honors_event_time() {
        // #4 — a replayed backdated event must not take the front of the
        // beam just because it arrived last.
        let mut b = AttentionBeam::with_config(BeamConfig {
            recency_capacity: 8,
            max_landmarks: 0,
            max_lookback: 0,
            max_beam: 16,
            ..Default::default()
        });
        let now = chrono::Utc::now();
        let fresh = Uuid::from_u128(1);
        let ancient = Uuid::from_u128(2);
        b.observe(&ObservationEvent {
            memory_id: fresh,
            source: "live".into(),
            weight: 1.0,
            ts: now,
        });
        b.observe(&ObservationEvent {
            memory_id: ancient,
            source: "replay".into(),
            weight: 1.0,
            ts: now - chrono::Duration::days(30),
        });
        assert_eq!(
            b.candidates(),
            vec![fresh, ancient],
            "recency tier must be ordered by event time"
        );
    }

    #[test]
    fn beam_config_deserializes_partially() {
        // Regression for #16 — a partial config had to restate every field.
        let cfg: BeamConfig = serde_json::from_str(r#"{"max_beam": 8}"#).expect("partial config");
        assert_eq!(cfg.max_beam, 8);
        let d = BeamConfig::default();
        assert_eq!(cfg.recency_capacity, d.recency_capacity);
        assert_eq!(cfg.max_landmarks, d.max_landmarks);
        assert_eq!(cfg.max_lookback, d.max_lookback);

        // And an empty payload is the full default.
        let empty: BeamConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(empty.max_beam, d.max_beam);
    }

    #[test]
    fn beam_truncated_at_max_beam() {
        let mut b = AttentionBeam::with_config(BeamConfig {
            recency_capacity: 1000,
            max_landmarks: 0,
            max_lookback: 0,
            max_beam: 5,
            ..Default::default()
        });
        for _ in 0..50 {
            b.observe_now(Uuid::new_v4(), "test");
        }
        assert_eq!(b.candidates().len(), 5);
    }
}
