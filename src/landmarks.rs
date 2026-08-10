//! Landmark set — the block-summary primitive.
//!
//! RuVector's third primitive is "landmark block summaries" — running
//! mean-key/mean-value over fixed-size blocks of the key sequence, so the
//! beam can reach the entire context through O(N/block_size) anchors. The
//! HRM equivalent already exists in kannaka-memory: **cluster exemplars**,
//! which are published on the `KANNAKA.exemplar.<agent>.<cluster>` NATS
//! subject and represent the canonical wavefront for each phase-coherent
//! cluster.
//!
//! This module is a thin owner for those exemplar IDs. The host wires them
//! in from outside (typically: subscribe to the NATS exemplar subjects,
//! upsert here). It is intentionally NOT a redo of the clustering — the
//! medium owns that.

use std::collections::HashMap;
use uuid::Uuid;

/// One landmark — a cluster exemplar.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Landmark {
    /// The memory ID of the exemplar.
    pub id: Uuid,
    /// Stable label for the cluster this exemplar represents (free-form;
    /// kannaka-memory tends to use semantic labels like "audio" / "philosophy").
    pub cluster_label: String,
    /// Importance hint provided by the medium. Higher = include before
    /// lower-importance landmarks when the beam is tight. Default to 1.0.
    pub weight: f32,
}

/// Owns the current set of landmarks. Inexpensive to mutate from a NATS
/// subscriber loop — uses a HashMap keyed by cluster label so a re-emit on
/// the same cluster replaces rather than duplicates.
#[derive(Debug, Default, Clone)]
pub struct LandmarkSet {
    /// cluster_label -> Landmark
    pub(crate) by_cluster: HashMap<String, Landmark>,
}

impl LandmarkSet {
    /// Empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Upsert a landmark. If the cluster_label is already known, replace it
    /// — the medium's exemplar can drift over time as the cluster evolves.
    ///
    /// If the same exemplar `id` is already registered under a *different*
    /// cluster label, that stale alias is dropped. Clusters get relabelled
    /// as the medium re-clusters, and keying only by label left the old
    /// entry behind: the same exemplar then occupied two landmark slots,
    /// double-counting one cluster's pull and crowding a real cluster out
    /// of a tight beam. One exemplar id, one landmark. (#24)
    pub fn upsert(&mut self, mut l: Landmark) {
        // Sanitize untrusted weights (NATS exemplars can carry NaN/Inf).
        // A non-finite weight breaks strict-weak-ordering in the sort and
        // scrambles landmark order. Default such weights to 1.0. (#12)
        if !l.weight.is_finite() {
            l.weight = 1.0;
        }
        let id = l.id;
        let label = l.cluster_label.clone();
        self.by_cluster
            .retain(|existing_label, existing| existing.id != id || *existing_label == label);
        self.by_cluster.insert(label, l);
    }

    /// Drop a cluster entirely (e.g. on prune).
    pub fn remove(&mut self, cluster_label: &str) {
        self.by_cluster.remove(cluster_label);
    }

    /// Snapshot — landmark IDs ordered by weight descending, ties broken by
    /// cluster label. Caller decides how many to actually use.
    pub fn snapshot(&self) -> Vec<Uuid> {
        let mut entries: Vec<&Landmark> = self.by_cluster.values().collect();
        // total_cmp gives a total order over all f32 (incl. any non-finite
        // that slipped past sanitization), keeping the sort well-formed. (#12)
        //
        // The cluster_label tiebreak is what makes the result reproducible.
        // `sort_by` is stable, so equal weights preserved the order they
        // came out of the HashMap in — which Rust deliberately randomises
        // per process. Two runs over identical state could emit different
        // beams, and with a tight `max_landmarks` that means genuinely
        // different candidate sets. (#7)
        entries.sort_by(|a, b| {
            b.weight
                .total_cmp(&a.weight)
                .then_with(|| a.cluster_label.cmp(&b.cluster_label))
        });
        entries.into_iter().map(|l| l.id).collect()
    }

    /// Number of clusters tracked.
    pub fn len(&self) -> usize {
        self.by_cluster.len()
    }

    /// True if no landmarks have been registered yet.
    pub fn is_empty(&self) -> bool {
        self.by_cluster.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upsert_replaces_by_cluster_label() {
        let mut s = LandmarkSet::new();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        s.upsert(Landmark {
            id: a,
            cluster_label: "philosophy".into(),
            weight: 1.0,
        });
        s.upsert(Landmark {
            id: b,
            cluster_label: "philosophy".into(),
            weight: 1.0,
        });
        assert_eq!(s.len(), 1);
        assert_eq!(s.snapshot(), vec![b]);
    }

    #[test]
    fn reemitting_an_exemplar_under_a_new_label_drops_the_stale_alias() {
        // Regression for #24 — keying only by cluster_label left the old
        // entry behind when the medium relabelled a cluster, so one
        // exemplar occupied two landmark slots and double-counted its pull.
        let mut s = LandmarkSet::new();
        let exemplar = Uuid::new_v4();
        s.upsert(Landmark {
            id: exemplar,
            cluster_label: "audio".into(),
            weight: 1.0,
        });
        s.upsert(Landmark {
            id: exemplar,
            cluster_label: "music".into(),
            weight: 1.0,
        });
        assert_eq!(s.len(), 1, "one exemplar id, one landmark");
        assert_eq!(s.snapshot(), vec![exemplar]);

        // A genuinely different exemplar under the old label is unaffected.
        let other = Uuid::new_v4();
        s.upsert(Landmark {
            id: other,
            cluster_label: "audio".into(),
            weight: 1.0,
        });
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn equal_weight_snapshot_is_deterministic() {
        // Regression for #7 — equal weights fell back to HashMap order.
        let ids: Vec<Uuid> = (0..6).map(|_| Uuid::new_v4()).collect();
        let build = || {
            let mut s = LandmarkSet::new();
            for (i, id) in ids.iter().enumerate() {
                s.upsert(Landmark {
                    id: *id,
                    cluster_label: format!("cluster-{i}"),
                    weight: 1.0,
                });
            }
            s.snapshot()
        };
        let first = build();
        for _ in 0..16 {
            assert_eq!(first, build());
        }
        // And it is the cluster_label order, not an arbitrary one.
        assert_eq!(first, ids);
    }

    #[test]
    fn snapshot_orders_by_weight_descending() {
        let mut s = LandmarkSet::new();
        let high = Uuid::new_v4();
        let low = Uuid::new_v4();
        s.upsert(Landmark {
            id: low,
            cluster_label: "a".into(),
            weight: 0.3,
        });
        s.upsert(Landmark {
            id: high,
            cluster_label: "b".into(),
            weight: 0.9,
        });
        assert_eq!(s.snapshot(), vec![high, low]);
    }
}
