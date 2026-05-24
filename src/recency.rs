//! Recency ring — the "local window" primitive.
//!
//! A FIFO of the most-recently-observed memory IDs. Capacity bounded; oldest
//! entries fall out of the ring as new ones come in. This is the cheapest and
//! most reliable signal of "what should be in the next recall's candidate
//! set" — if a memory was just observed by a sensor, it's almost certainly
//! relevant to the next query.
//!
//! Equivalent to RuVector's local-window primitive: a fixed N=W around the
//! cursor that's always scored without any selection logic.

use std::collections::VecDeque;
use uuid::Uuid;

/// FIFO ring of recently observed memory IDs.
#[derive(Debug, Clone)]
pub struct RecencyRing {
    capacity: usize,
    buf: VecDeque<Uuid>,
}

impl RecencyRing {
    /// Create a ring with the given capacity. A reasonable default is 128 —
    /// large enough to span a small conversation, small enough to be cheap
    /// even on a Pi Zero.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            buf: VecDeque::with_capacity(capacity),
        }
    }

    /// Push an observation. If `id` is already in the ring it's moved to the
    /// front (most-recent), not duplicated — observing the same memory twice
    /// in quick succession should keep the ring tight, not flush it.
    pub fn push(&mut self, id: Uuid) {
        // Cheap-but-correct dedup: scan + remove. For capacity 128 this is
        // never the bottleneck.
        if let Some(pos) = self.buf.iter().position(|x| *x == id) {
            self.buf.remove(pos);
        }
        if self.buf.len() == self.capacity {
            self.buf.pop_back();
        }
        self.buf.push_front(id);
    }

    /// Snapshot the ring, most-recent first.
    pub fn snapshot(&self) -> Vec<Uuid> {
        self.buf.iter().copied().collect()
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_snapshot_are_most_recent_first() {
        let mut r = RecencyRing::new(4);
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();
        r.push(a);
        r.push(b);
        r.push(c);
        assert_eq!(r.snapshot(), vec![c, b, a]);
    }

    #[test]
    fn dedup_moves_to_front_without_growing() {
        let mut r = RecencyRing::new(3);
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        r.push(a);
        r.push(b);
        r.push(a);
        assert_eq!(r.len(), 2);
        assert_eq!(r.snapshot(), vec![a, b]);
    }

    #[test]
    fn capacity_evicts_oldest() {
        let mut r = RecencyRing::new(2);
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();
        r.push(a);
        r.push(b);
        r.push(c);
        assert_eq!(r.snapshot(), vec![c, b]);
    }
}
