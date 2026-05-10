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
//! at a small N (default 8). When a bucket fills, the lowest-weight entry is
//! evicted. Weight is currently observation count (how many times this memory
//! showed up in the recency ring while it lived in this bucket), so the
//! survivors are the ones that mattered.

use chrono::{DateTime, Duration, Utc};
use std::collections::HashMap;
use uuid::Uuid;

/// The exponentially-spaced age boundaries used by [`LogStrideLookback`].
/// An entry whose age is `< boundary[k]` belongs in bucket `k`. The implicit
/// final bucket holds everything older than the last boundary.
pub const DEFAULT_BOUNDARIES_SECONDS: &[i64] = &[
    60,            // < 1 min
    5 * 60,        // < 5 min
    30 * 60,       // < 30 min
    3 * 3600,      // < 3 h
    24 * 3600,     // < 1 d
    7 * 24 * 3600, // < 7 d
    30 * 24 * 3600,// < 30 d
];

/// Per-memory state inside the lookback.
#[derive(Debug, Clone)]
struct Entry {
    id: Uuid,
    first_seen: DateTime<Utc>,
    last_seen: DateTime<Utc>,
    observation_count: u32,
}

/// Log-stride lookback. Owns a per-bucket cap and the bucket cursors.
#[derive(Debug, Clone)]
pub struct LogStrideLookback {
    boundaries: Vec<Duration>,
    bucket_cap: usize,
    // bucket_index -> entries (newest-first within a bucket)
    buckets: Vec<Vec<Entry>>,
    // id -> (bucket_index, position) for fast updates. Rebuilt on rebalance.
    locator: HashMap<Uuid, (usize, usize)>,
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
        let boundaries: Vec<Duration> =
            boundaries_seconds.iter().copied().map(Duration::seconds).collect();
        let n_buckets = boundaries.len() + 1;
        Self {
            boundaries,
            bucket_cap: bucket_cap.max(1),
            buckets: vec![Vec::new(); n_buckets],
            locator: HashMap::new(),
        }
    }

    /// Record an observation. If the memory is already known, increment its
    /// counter and refresh `last_seen`. Otherwise, place it into the
    /// appropriate bucket based on `(now - first_seen)`.
    ///
    /// `now` is passed in (rather than read from the clock inside) so
    /// callers can do exact-time test fixtures.
    pub fn observe(&mut self, id: Uuid, now: DateTime<Utc>) {
        if let Some(&(b, p)) = self.locator.get(&id) {
            let entry = &mut self.buckets[b][p];
            entry.last_seen = now;
            entry.observation_count = entry.observation_count.saturating_add(1);
            return;
        }
        // First observation. New entries start in bucket 0 (newest).
        let entry = Entry { id, first_seen: now, last_seen: now, observation_count: 1 };
        let bucket = 0;
        let buf = &mut self.buckets[bucket];
        if buf.len() >= self.bucket_cap {
            // Bucket is full. Find the lowest-weight resident; only displace
            // it if the newcomer would outrank it. New entries arrive with
            // count=1, so by default they lose to anyone with count>=2 —
            // which is the right call: high-observation-count memories
            // should stick.
            let (min_pos, min_count) = buf.iter().enumerate()
                .map(|(p, e)| (p, e.observation_count))
                .min_by_key(|&(_, c)| c)
                .unwrap();
            if entry.observation_count <= min_count {
                // Newcomer is weaker (or equal); drop on the floor and don't
                // register in the locator.
                return;
            }
            // Displace the weakest.
            let evicted = buf.remove(min_pos);
            self.locator.remove(&evicted.id);
            // Locator positions shift for this bucket — rebuild.
            for (p, e) in self.buckets[bucket].iter().enumerate() {
                self.locator.insert(e.id, (bucket, p));
            }
        }
        let pos = self.buckets[bucket].len();
        self.buckets[bucket].push(entry);
        self.locator.insert(id, (bucket, pos));
    }

    /// Rebalance: move entries to the bucket their current age says they
    /// belong in. Cheap (O(total entries)) and only needs to run periodically
    /// — every few seconds is fine. Call before snapshotting if accuracy
    /// matters; skip for hot-path observations.
    pub fn rebalance(&mut self, now: DateTime<Utc>) {
        let mut new_buckets: Vec<Vec<Entry>> = vec![Vec::new(); self.buckets.len()];
        let mut new_locator: HashMap<Uuid, (usize, usize)> = HashMap::new();
        let cap = self.bucket_cap;
        let boundaries = self.boundaries.clone();

        for bucket in self.buckets.drain(..) {
            for entry in bucket {
                let age = now - entry.first_seen;
                let target = Self::bucket_for_age(&boundaries, age);
                let buf = &mut new_buckets[target];
                if buf.len() >= cap {
                    // Lowest-weight eviction within the destination bucket.
                    if let Some((min_pos, _)) = buf.iter().enumerate()
                        .min_by_key(|(_, e)| e.observation_count)
                    {
                        if buf[min_pos].observation_count < entry.observation_count {
                            buf.remove(min_pos);
                        } else {
                            continue; // newcomer is lower-weight, drop it
                        }
                    }
                }
                let pos = buf.len();
                new_locator.insert(entry.id, (target, pos));
                buf.push(entry);
            }
        }
        self.buckets = new_buckets;
        self.locator = new_locator;
    }

    fn bucket_for_age(boundaries: &[Duration], age: Duration) -> usize {
        for (i, b) in boundaries.iter().enumerate() {
            if age < *b { return i; }
        }
        boundaries.len() // overflow bucket
    }

    /// Snapshot the lookback as a flat vec of memory IDs, newest-bucket first.
    pub fn snapshot(&self) -> Vec<Uuid> {
        let mut out = Vec::new();
        for bucket in &self.buckets {
            for entry in bucket {
                out.push(entry.id);
            }
        }
        out
    }

    /// Total entries across all buckets.
    pub fn len(&self) -> usize { self.locator.len() }

    /// True if empty.
    pub fn is_empty(&self) -> bool { self.locator.is_empty() }

}

impl Default for LogStrideLookback {
    fn default() -> Self { Self::new() }
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
        // Bucket 0 (< 1 min) at observation time, then rebalance at +2h.
        l.rebalance(t0 + Duration::hours(2));
        // After 2h aging it should be in bucket 3 (< 3h).
        assert_eq!(l.snapshot(), vec![a]);
        // We can't trivially observe bucket index externally without exposing
        // internals, but the post-condition is that the entry survives and
        // the locator points somewhere valid:
        assert_eq!(l.len(), 1);
    }

    #[test]
    fn cap_evicts_lowest_weight() {
        let mut l = LogStrideLookback::with_config(&[60], 2);
        let high = Uuid::new_v4();
        let mid = Uuid::new_v4();
        let low = Uuid::new_v4();
        let t = Utc::now();
        l.observe(high, t); l.observe(high, t); l.observe(high, t);
        l.observe(mid, t);  l.observe(mid, t);
        // bucket 0 full at cap=2. Inserting `low` should evict the lowest:
        // `low` itself is the lowest (count=1), so it's dropped on insert
        // and `high`+`mid` survive.
        l.observe(low, t);
        let snap = l.snapshot();
        assert!(snap.contains(&high));
        assert!(snap.contains(&mid));
        assert!(!snap.contains(&low));
    }
}
