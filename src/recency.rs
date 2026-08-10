//! Recency ring — the "local window" primitive.
//!
//! A ring of the most-recently-observed memory IDs. Capacity bounded; the
//! oldest entries fall out as new ones come in. This is the cheapest and
//! most reliable signal of "what should be in the next recall's candidate
//! set" — if a memory was just observed by a sensor, it's almost certainly
//! relevant to the next query.
//!
//! Equivalent to RuVector's local-window primitive: a fixed N=W around the
//! cursor that's always scored without any selection logic.
//!
//! # Ordering is by event time, not arrival
//!
//! Entries are ordered by the timestamp on the observation, most-recent
//! first, so a replayed or out-of-order batch produces the same ring as a
//! live one. Arrival-order insertion meant a backdated event took the front
//! of the ring regardless of when it actually happened. (#4)
//!
//! Note this needs no reference clock, unlike [`crate::lookback`]: the ring
//! only *orders*, it does not *age*. A backdated event sorts into its
//! correct position by comparison alone and falls off the back if it lands
//! past capacity, so there is no shared "now" for it to drag backwards.

use chrono::{DateTime, Utc};
use std::collections::VecDeque;
use uuid::Uuid;

/// One entry in the ring.
#[derive(Debug, Clone)]
struct RecencyEntry {
    id: Uuid,
    /// Present only when the ring is keying per-source. `None` means all
    /// sources collapse onto the memory id, which is the default. (#3)
    source: Option<String>,
    ts: DateTime<Utc>,
}

impl RecencyEntry {
    fn matches(&self, id: Uuid, source: Option<&str>) -> bool {
        self.id == id && self.source.as_deref() == source
    }
}

/// Ring of recently observed memory IDs, ordered by event time.
#[derive(Debug, Clone)]
pub struct RecencyRing {
    capacity: usize,
    /// Most-recent-first by `ts`.
    buf: VecDeque<RecencyEntry>,
}

impl RecencyRing {
    /// Create a ring with the given capacity. A reasonable default is 128 —
    /// large enough to span a small conversation, small enough to be cheap
    /// even on a Pi Zero.
    ///
    /// A capacity of `0` disables the recency tier entirely. It used to be
    /// silently promoted to 1, so a config that asked for no recency still
    /// got one id in every beam. (#17)
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buf: VecDeque::with_capacity(capacity),
        }
    }

    /// Push an observation stamped with the current wall clock.
    ///
    /// Convenience for callers with no event timestamp to hand; prefer
    /// [`Self::push_at`] when you have one.
    pub fn push(&mut self, id: Uuid) {
        self.push_at(id, Utc::now());
    }

    /// Push an observation that happened at `ts`.
    ///
    /// If `id` is already in the ring its timestamp advances to the later
    /// of the two and it repositions — observing the same memory twice in
    /// quick succession keeps the ring tight rather than flushing it.
    pub fn push_at(&mut self, id: Uuid, ts: DateTime<Utc>) {
        self.insert(id, None, ts);
    }

    /// Push an observation keyed on `(source, id)` rather than `id` alone.
    ///
    /// Used when the beam is configured `per_source`, so a left/right
    /// mirror pair each hold their own slot and pull on the beam
    /// independently instead of collapsing onto one entry. (#3)
    pub fn push_keyed(&mut self, id: Uuid, source: &str, ts: DateTime<Utc>) {
        self.insert(id, Some(source), ts);
    }

    fn insert(&mut self, id: Uuid, source: Option<&str>, ts: DateTime<Utc>) {
        if self.capacity == 0 {
            return;
        }

        // Dedup on the configured key. Linear scan; at capacity 128 this is
        // never the bottleneck.
        if let Some(pos) = self.buf.iter().position(|e| e.matches(id, source)) {
            let mut existing = self.buf.remove(pos).expect("position just found");
            // Never move an entry backwards in time on a late-arriving
            // duplicate.
            existing.ts = existing.ts.max(ts);
            self.insert_sorted(existing);
            return;
        }

        self.insert_sorted(RecencyEntry {
            id,
            source: source.map(str::to_owned),
            ts,
        });

        // Trim from the back — the oldest by event time.
        while self.buf.len() > self.capacity {
            self.buf.pop_back();
        }
    }

    /// Insert most-recent-first. Ties place the newcomer ahead, preserving
    /// the "same instant, latest push wins" feel of the old arrival-ordered
    /// ring.
    fn insert_sorted(&mut self, entry: RecencyEntry) {
        let pos = self
            .buf
            .iter()
            .position(|e| e.ts <= entry.ts)
            .unwrap_or(self.buf.len());
        self.buf.insert(pos, entry);
    }

    /// Snapshot the ring, most-recent first.
    ///
    /// When keying per-source the same memory id can appear more than once
    /// — that is the point, it pulls harder — and the beam dedups ids as it
    /// composes tiers.
    pub fn snapshot(&self) -> Vec<Uuid> {
        self.buf.iter().map(|e| e.id).collect()
    }

    /// Current population.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// True if the ring is empty.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Configured capacity (max population).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// True if `id` is present under any source key.
    pub fn contains(&self, id: &Uuid) -> bool {
        self.buf.iter().any(|e| e.id == *id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn push_and_snapshot_are_most_recent_first() {
        let mut r = RecencyRing::new(4);
        let t = Utc::now();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();
        r.push_at(a, t);
        r.push_at(b, t + Duration::seconds(1));
        r.push_at(c, t + Duration::seconds(2));
        assert_eq!(r.snapshot(), vec![c, b, a]);
    }

    #[test]
    fn dedup_moves_to_front_without_growing() {
        let mut r = RecencyRing::new(3);
        let t = Utc::now();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        r.push_at(a, t);
        r.push_at(b, t + Duration::seconds(1));
        r.push_at(a, t + Duration::seconds(2));
        assert_eq!(r.len(), 2);
        assert_eq!(r.snapshot(), vec![a, b]);
    }

    #[test]
    fn zero_capacity_disables_the_tier() {
        // Regression for #17 — capacity 0 was promoted to 1, so a config
        // asking for no recency tier still emitted one id per beam.
        let mut r = RecencyRing::new(0);
        r.push(Uuid::new_v4());
        r.push(Uuid::new_v4());
        assert_eq!(r.len(), 0);
        assert!(r.snapshot().is_empty());
        assert_eq!(r.capacity(), 0);
    }

    #[test]
    fn capacity_evicts_oldest() {
        let mut r = RecencyRing::new(2);
        let t = Utc::now();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();
        r.push_at(a, t);
        r.push_at(b, t + Duration::seconds(1));
        r.push_at(c, t + Duration::seconds(2));
        assert_eq!(r.snapshot(), vec![c, b]);
    }

    #[test]
    fn ordering_follows_event_time_not_arrival() {
        // Regression for #4 — a backdated event used to take the front of
        // the ring purely because it arrived last.
        let mut r = RecencyRing::new(8);
        let now = Utc::now();
        let fresh = Uuid::new_v4();
        let ancient = Uuid::new_v4();
        r.push_at(fresh, now);
        r.push_at(ancient, now - Duration::days(30));
        assert_eq!(
            r.snapshot(),
            vec![fresh, ancient],
            "the genuinely recent id must stay in front"
        );
    }

    #[test]
    fn replay_order_does_not_change_the_ring() {
        // Property: feeding the same events in any arrival order yields the
        // same ring, because ordering is by event time.
        let t = Utc::now();
        let ids: Vec<Uuid> = (0..5).map(|_| Uuid::new_v4()).collect();
        let stamped: Vec<(Uuid, DateTime<Utc>)> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, t + Duration::seconds(i as i64)))
            .collect();

        let mut forward = RecencyRing::new(8);
        for (id, ts) in &stamped {
            forward.push_at(*id, *ts);
        }
        let mut reversed = RecencyRing::new(8);
        for (id, ts) in stamped.iter().rev() {
            reversed.push_at(*id, *ts);
        }
        assert_eq!(forward.snapshot(), reversed.snapshot());
    }

    #[test]
    fn a_backdated_event_past_capacity_is_dropped() {
        // The ring is full of fresh entries; a very old event should not
        // displace any of them.
        let mut r = RecencyRing::new(2);
        let t = Utc::now();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        r.push_at(a, t);
        r.push_at(b, t + Duration::seconds(1));
        let ancient = Uuid::new_v4();
        r.push_at(ancient, t - Duration::days(1));
        assert_eq!(r.snapshot(), vec![b, a]);
        assert!(!r.contains(&ancient));
    }

    #[test]
    fn duplicate_never_moves_an_entry_backwards_in_time() {
        let mut r = RecencyRing::new(4);
        let t = Utc::now();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        r.push_at(a, t + Duration::seconds(10));
        r.push_at(b, t + Duration::seconds(5));
        // A late-arriving OLD duplicate of `a` must not demote it.
        r.push_at(a, t);
        assert_eq!(r.snapshot(), vec![a, b]);
    }

    #[test]
    fn per_source_keys_hold_independent_slots() {
        // #3 — a left/right mirror pair observing the same memory each get
        // their own slot instead of collapsing onto one.
        let mut r = RecencyRing::new(8);
        let t = Utc::now();
        let id = Uuid::new_v4();
        r.push_keyed(id, "eye:left", t);
        r.push_keyed(id, "eye:right", t + Duration::seconds(1));
        assert_eq!(r.len(), 2, "distinct sources are distinct entries");

        // Same source re-observing still dedups.
        r.push_keyed(id, "eye:left", t + Duration::seconds(2));
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn keyed_and_unkeyed_entries_do_not_collide() {
        let mut r = RecencyRing::new(8);
        let t = Utc::now();
        let id = Uuid::new_v4();
        r.push_at(id, t);
        r.push_keyed(id, "eye:left", t + Duration::seconds(1));
        assert_eq!(r.len(), 2);
        assert_eq!(r.snapshot(), vec![id, id]);
    }
}
