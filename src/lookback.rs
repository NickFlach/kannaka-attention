//! Log-stride lookback — the hierarchical-reach primitive.
//!
//! RuVector's sparse attention reaches into the past at exponentially-spaced
//! strides (i-1, i-2, i-4, i-8, ...). The equivalent for a memory medium that
//! lives in clock-time rather than token-time is **exponentially-aged
//! buckets**: keep representatives of recent (~1m), middle-aged (~hour, ~day),
//! and old (~week, ~month) memories so the beam reaches across the lifespan
//! without scanning every entry.
//!
//! The buckets do NOT store every memory at that age — they cap each bucket
//! at a small N (default 8). When a bucket is over cap, the lowest-weight
//! entries are dropped, so the survivors are the ones that mattered.
//!
//! # Aging is by `last_seen`, not `first_seen`
//!
//! A memory's bucket is decided by how long it has been since it was last
//! *observed*, not since it was first created. Aging by `first_seen` (the
//! previous behaviour) meant a memory observed a hundred times in the last
//! minute still sat in the 30-day bucket because it happened to be old —
//! which defeats the whole point of a reach structure that the beam reads
//! newest-bucket-first. Re-observing a memory makes it fresh. (#23)
//!
//! # Placement is exact at every observation
//!
//! `observe` re-buckets the whole structure against the timestamp of the
//! event being recorded. The structure is capped at
//! `bucket_cap × (boundaries + 1)` entries — 64 by default — so a full
//! re-bucket is a few dozen comparisons, cheap enough that there is no
//! reason to defer it. The previous "rebalance every 32nd observation
//! against `Utc::now()`" scheme was the direct cause of two defects: buckets
//! were stale for up to 31 observations (#22), and aging was driven by wall
//! clock rather than the event timeline, so replayed or backdated events
//! bucketed wrongly and tests could not be deterministic (#25).

use chrono::{DateTime, Duration, Utc};
use uuid::Uuid;

/// The exponentially-spaced age boundaries used by [`LogStrideLookback`].
/// An entry whose age is `< boundary[k]` belongs in bucket `k`. The implicit
/// final bucket holds everything older than the last boundary.
pub const DEFAULT_BOUNDARIES_SECONDS: &[i64] = &[
    60,             // < 1 min
    5 * 60,         // < 5 min
    30 * 60,        // < 30 min
    3 * 3600,       // < 3 h
    24 * 3600,      // < 1 d
    7 * 24 * 3600,  // < 7 d
    30 * 24 * 3600, // < 30 d
];

/// The documented clamp range for an observation's strength multiplier.
/// Mirrors the contract on `ObservationEvent.weight`.
pub const MIN_OBSERVATION_WEIGHT: f32 = 0.01;
/// Upper end of the observation strength clamp.
pub const MAX_OBSERVATION_WEIGHT: f32 = 10.0;

/// Clamp an observation weight into the documented `[0.01, 10.0]` range,
/// mapping non-finite input to the neutral 1.0. (#13)
pub(crate) fn sanitize_weight(weight: f32) -> f32 {
    if weight.is_finite() {
        weight.clamp(MIN_OBSERVATION_WEIGHT, MAX_OBSERVATION_WEIGHT)
    } else {
        1.0
    }
}

/// Per-memory state inside the lookback.
#[derive(Debug, Clone)]
struct Entry {
    id: Uuid,
    /// Present only when the lookback is keying per-source. `None` means
    /// all sources collapse onto the memory id, which is the default. (#3)
    source: Option<String>,
    first_seen: DateTime<Utc>,
    last_seen: DateTime<Utc>,
    /// Accumulated observation strength. A deliberate observation at
    /// weight 10.0 pulls as hard as ten ambient ones — this is what makes
    /// `ObservationEvent.weight` mean something. (#13)
    weight: f32,
}

impl Entry {
    fn matches(&self, id: Uuid, source: Option<&str>) -> bool {
        self.id == id && self.source.as_deref() == source
    }
}

/// Log-stride lookback. Owns a per-bucket cap and the bucket boundaries.
#[derive(Debug, Clone)]
pub struct LogStrideLookback {
    boundaries: Vec<Duration>,
    bucket_cap: usize,
    /// Flat entry list. Bucketing is derived from `last_seen` on demand
    /// rather than stored, so there are no `(bucket, position)` indices to
    /// go stale behind an eviction.
    entries: Vec<Entry>,
    /// The newest timestamp observed so far — the reference "now" used to
    /// bucket entries in `snapshot`.
    clock: Option<DateTime<Utc>>,
}

impl LogStrideLookback {
    /// Build with the default boundaries (1m, 5m, 30m, 3h, 1d, 7d, 30d) and
    /// a per-bucket cap of 8. That gives at most 64 ids across all buckets —
    /// negligible memory, but spans the whole observable lifespan.
    pub fn new() -> Self {
        Self::with_config(DEFAULT_BOUNDARIES_SECONDS, 8)
    }

    /// Custom boundaries + per-bucket cap. The boundaries must be ascending;
    /// duplicates are tolerated but waste a bucket.
    pub fn with_config(boundaries_seconds: &[i64], bucket_cap: usize) -> Self {
        let boundaries: Vec<Duration> = boundaries_seconds
            .iter()
            .copied()
            .map(Duration::seconds)
            .collect();
        Self {
            boundaries,
            bucket_cap: bucket_cap.max(1),
            entries: Vec::new(),
            clock: None,
        }
    }

    /// Number of buckets, including the implicit overflow bucket.
    fn n_buckets(&self) -> usize {
        self.boundaries.len() + 1
    }

    /// Record an observation at unit weight.
    ///
    /// `now` is passed in (rather than read from the clock inside) so
    /// callers can do exact-time test fixtures, and so replayed events
    /// bucket against the event timeline rather than wall clock. (#25)
    pub fn observe(&mut self, id: Uuid, now: DateTime<Utc>) {
        self.observe_weighted(id, now, 1.0);
    }

    /// Record an observation carrying an explicit strength multiplier.
    ///
    /// `weight` is clamped to the documented `[0.01, 10.0]` range. Repeated
    /// observations accumulate, so a memory's standing in its bucket
    /// reflects both how often and how strongly it was attended. (#13)
    pub fn observe_weighted(&mut self, id: Uuid, now: DateTime<Utc>, weight: f32) {
        self.record(id, None, now, weight);
    }

    /// Record an observation keyed on `(source, id)` rather than `id` alone.
    ///
    /// Used when the beam is configured `per_source`, so a left/right
    /// mirror pair accumulate their own weight and bucket placement
    /// independently instead of collapsing onto one entry. (#3)
    pub fn observe_keyed(&mut self, id: Uuid, source: &str, now: DateTime<Utc>, weight: f32) {
        self.record(id, Some(source), now, weight);
    }

    fn record(&mut self, id: Uuid, source: Option<&str>, now: DateTime<Utc>, weight: f32) {
        let weight = sanitize_weight(weight);

        // The reference clock only moves forward. An out-of-order or
        // backdated event still updates its own entry, but must not drag
        // the whole structure back in time and un-age everything else.
        self.clock = Some(match self.clock {
            Some(prev) if prev > now => prev,
            _ => now,
        });

        if let Some(entry) = self.entries.iter_mut().find(|e| e.matches(id, source)) {
            entry.last_seen = entry.last_seen.max(now);
            entry.weight += weight;
        } else {
            self.entries.push(Entry {
                id,
                source: source.map(str::to_owned),
                first_seen: now,
                last_seen: now,
                weight,
            });
        }

        self.enforce_caps();
    }

    /// Re-bucket every entry against `now` and re-apply the per-bucket caps.
    ///
    /// `observe` already does this on every call, so this is only needed
    /// when time has passed without any new observation and the caller
    /// wants aging reflected before reading [`snapshot`](Self::snapshot).
    /// It is idempotent.
    pub fn rebalance(&mut self, now: DateTime<Utc>) {
        self.clock = Some(match self.clock {
            Some(prev) if prev > now => prev,
            _ => now,
        });
        self.enforce_caps();
    }

    /// Drop the weakest entries from any over-cap bucket.
    ///
    /// Ranking within a bucket is `(weight desc, last_seen desc, id asc)`.
    /// The `last_seen` tiebreak is what unblocks a full bucket 0: a
    /// newcomer arrives at weight 1.0 and used to *tie* every unit-weight
    /// resident, and a tie was resolved in the resident's favour — so once
    /// bucket 0 held `cap` entries, no new memory was ever admitted again
    /// until something aged out. Freshest wins the tie now. (#14)
    fn enforce_caps(&mut self) {
        let now = match self.clock {
            Some(c) => c,
            None => return,
        };
        let cap = self.bucket_cap;
        let boundaries = &self.boundaries;
        let n_buckets = boundaries.len() + 1;

        let mut kept: Vec<Entry> = Vec::with_capacity(self.entries.len().min(cap * n_buckets));
        for bucket in 0..n_buckets {
            let mut in_bucket: Vec<&Entry> = self
                .entries
                .iter()
                .filter(|e| Self::bucket_for_age(boundaries, now - e.last_seen) == bucket)
                .collect();
            in_bucket.sort_by(|a, b| {
                b.weight
                    .total_cmp(&a.weight)
                    .then(b.last_seen.cmp(&a.last_seen))
                    .then(a.id.cmp(&b.id))
                    .then_with(|| a.source.cmp(&b.source))
            });
            in_bucket.truncate(cap);
            kept.extend(in_bucket.into_iter().cloned());
        }
        self.entries = kept;
    }

    fn bucket_for_age(boundaries: &[Duration], age: Duration) -> usize {
        for (i, b) in boundaries.iter().enumerate() {
            if age < *b {
                return i;
            }
        }
        boundaries.len() // overflow bucket
    }

    /// Snapshot the lookback as a flat vec of memory IDs, newest-bucket
    /// first and **newest-first within each bucket**.
    ///
    /// Within-bucket order used to be raw insertion order, i.e. oldest
    /// first, which contradicted this type's own documentation and meant a
    /// beam with a tight `max_lookback` spent its budget on the stalest
    /// entries of the freshest bucket. (#15)
    pub fn snapshot(&self) -> Vec<Uuid> {
        let now = match self.clock {
            Some(c) => c,
            None => return Vec::new(),
        };
        let mut out = Vec::with_capacity(self.entries.len());
        for bucket in 0..self.n_buckets() {
            let mut in_bucket: Vec<&Entry> = self
                .entries
                .iter()
                .filter(|e| Self::bucket_for_age(&self.boundaries, now - e.last_seen) == bucket)
                .collect();
            in_bucket.sort_by(|a, b| {
                b.last_seen
                    .cmp(&a.last_seen)
                    .then(b.weight.total_cmp(&a.weight))
                    .then(a.id.cmp(&b.id))
                    .then_with(|| a.source.cmp(&b.source))
            });
            out.extend(in_bucket.into_iter().map(|e| e.id));
        }
        out
    }

    /// Total entries across all buckets.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// True if `id` is currently retained by the lookback.
    pub fn contains(&self, id: &Uuid) -> bool {
        self.entries.iter().any(|e| e.id == *id)
    }

    /// When `id` was first observed, if it is still retained.
    pub fn first_seen(&self, id: &Uuid) -> Option<DateTime<Utc>> {
        self.entries
            .iter()
            .find(|e| e.id == *id)
            .map(|e| e.first_seen)
    }
}

impl Default for LogStrideLookback {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_observation_lands_in_first_bucket() {
        let mut l = LogStrideLookback::new();
        let a = Uuid::new_v4();
        l.observe(a, Utc::now());
        assert_eq!(l.snapshot(), vec![a]);
    }

    #[test]
    fn aging_rebalance_moves_entry_to_older_bucket() {
        let mut l = LogStrideLookback::new();
        let a = Uuid::new_v4();
        let t0 = Utc::now() - Duration::hours(2);
        l.observe(a, t0);
        l.rebalance(t0 + Duration::hours(2));
        assert_eq!(l.snapshot(), vec![a]);
        assert_eq!(l.len(), 1);
    }

    #[test]
    fn cap_evicts_lowest_weight() {
        let mut l = LogStrideLookback::with_config(&[60], 2);
        let high = Uuid::new_v4();
        let mid = Uuid::new_v4();
        let low = Uuid::new_v4();
        let t = Utc::now();
        l.observe(high, t);
        l.observe(high, t);
        l.observe(high, t);
        l.observe(mid, t);
        l.observe(mid, t);
        l.observe(low, t);
        let snap = l.snapshot();
        assert!(snap.contains(&high));
        assert!(snap.contains(&mid));
        assert!(!snap.contains(&low), "weakest entry is the one dropped");
    }

    #[test]
    fn fresh_memories_still_admitted_after_bucket_zero_fills() {
        // Regression for #14. Every newcomer arrives at weight 1.0 and used
        // to tie the unit-weight residents; the tie went to the resident,
        // so once bucket 0 held `cap` entries the lookback stopped
        // admitting new memories entirely.
        let mut l = LogStrideLookback::with_config(&[60], 4);
        let t = Utc::now();
        let filler: Vec<Uuid> = (0..4).map(|_| Uuid::new_v4()).collect();
        for (i, id) in filler.iter().enumerate() {
            l.observe(*id, t + Duration::seconds(i as i64));
        }
        assert_eq!(l.len(), 4, "bucket 0 is full");

        let newcomer = Uuid::new_v4();
        l.observe(newcomer, t + Duration::seconds(10));
        assert!(
            l.snapshot().contains(&newcomer),
            "a fresh memory must still get in once the bucket is full"
        );
        // The staleset unit-weight resident is the one that made way.
        assert!(!l.snapshot().contains(&filler[0]));
    }

    #[test]
    fn reobserving_refreshes_bucket_placement() {
        // Regression for #23 — aging by first_seen pinned a heavily
        // re-observed memory in the 30d bucket forever.
        let mut l = LogStrideLookback::new();
        let old = Uuid::new_v4();
        let t0 = Utc::now() - Duration::days(20);
        l.observe(old, t0);

        let fresh = Uuid::new_v4();
        let now = Utc::now();
        l.observe(fresh, now);
        // Re-observe the ancient memory right now.
        l.observe(old, now);

        let snap = l.snapshot();
        assert_eq!(snap.len(), 2);
        // Both are now in bucket 0; the re-observed one is not stranded at
        // the far end of the snapshot behind six intervening buckets.
        assert!(snap.contains(&old) && snap.contains(&fresh));
        // first_seen is still preserved for callers that want provenance.
        assert_eq!(l.first_seen(&old), Some(t0));
    }

    #[test]
    fn snapshot_is_newest_first_within_a_bucket() {
        // Regression for #15 — insertion order meant oldest-first inside a
        // bucket, so a tight max_lookback spent its budget on the stalest
        // entries of the freshest bucket.
        let mut l = LogStrideLookback::new();
        let t = Utc::now();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();
        l.observe(a, t);
        l.observe(b, t + Duration::seconds(1));
        l.observe(c, t + Duration::seconds(2));
        assert_eq!(l.snapshot(), vec![c, b, a]);
    }

    #[test]
    fn placement_is_exact_without_waiting_for_a_rebalance_tick() {
        // Regression for #22 — buckets were only re-aged every 32nd
        // observation, so a snapshot taken in between reported stale
        // placement.
        let mut l = LogStrideLookback::new();
        let t0 = Utc::now();
        let old = Uuid::new_v4();
        l.observe(old, t0);

        // One observation later, four hours on. `old` must already read as
        // aged even though no rebalance tick has been reached.
        let fresh = Uuid::new_v4();
        l.observe(fresh, t0 + Duration::hours(4));

        let snap = l.snapshot();
        assert_eq!(
            snap,
            vec![fresh, old],
            "the freshly observed id sorts into an earlier bucket immediately"
        );
    }

    #[test]
    fn weight_outranks_bare_repetition() {
        // #13 — a single deliberate observation should outweigh a couple of
        // ambient mentions.
        let mut l = LogStrideLookback::with_config(&[60], 1);
        let t = Utc::now();
        let ambient = Uuid::new_v4();
        let deliberate = Uuid::new_v4();
        l.observe_weighted(ambient, t, 0.05);
        l.observe_weighted(ambient, t, 0.05);
        l.observe_weighted(deliberate, t, 10.0);
        assert_eq!(l.snapshot(), vec![deliberate]);
    }

    #[test]
    fn observation_weight_is_clamped_to_documented_range() {
        assert_eq!(sanitize_weight(-5.0), MIN_OBSERVATION_WEIGHT);
        assert_eq!(sanitize_weight(1e9), MAX_OBSERVATION_WEIGHT);
        assert_eq!(sanitize_weight(f32::NAN), 1.0);
        assert_eq!(sanitize_weight(f32::INFINITY), 1.0);
        assert_eq!(sanitize_weight(2.5), 2.5);
    }

    #[test]
    fn total_population_never_exceeds_cap_times_buckets() {
        // Property: the structure's whole selling point is a hard memory
        // ceiling. 8 buckets x cap 3 = 24, no matter how much we feed it.
        let mut l = LogStrideLookback::with_config(DEFAULT_BOUNDARIES_SECONDS, 3);
        let t = Utc::now();
        for i in 0..500 {
            l.observe(Uuid::new_v4(), t + Duration::seconds(i));
        }
        assert!(l.len() <= 3 * 8, "population {} exceeded the cap", l.len());
    }

    #[test]
    fn backdated_event_does_not_unage_the_structure() {
        // The reference clock only moves forward, so a late-arriving old
        // event cannot drag everything else back into bucket 0.
        let mut l = LogStrideLookback::new();
        let t = Utc::now();
        let recent = Uuid::new_v4();
        l.observe(recent, t);

        let backdated = Uuid::new_v4();
        l.observe(backdated, t - Duration::days(10));

        let snap = l.snapshot();
        assert_eq!(snap.len(), 2);
        assert_eq!(snap[0], recent, "the genuinely recent id still sorts first");
    }

    #[test]
    fn rebalance_is_idempotent() {
        let mut l = LogStrideLookback::new();
        let t = Utc::now();
        for i in 0..20 {
            l.observe(Uuid::new_v4(), t + Duration::seconds(i));
        }
        let once = {
            l.rebalance(t + Duration::hours(1));
            l.snapshot()
        };
        l.rebalance(t + Duration::hours(1));
        assert_eq!(once, l.snapshot());
    }
}
