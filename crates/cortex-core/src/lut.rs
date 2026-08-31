//! TF-LUT — the monotonic term-frequency reverse index.
//!
//! The reference `TokenLUT` stores `encoding_id → hash_position (+ TF)`;
//! materialization disambiguates collision groups by term frequency. Here:
//!
//! - `register` interns an encoding ID in the append-only LUT;
//! - `observe` accumulates per-ID term frequency from ingested inputs
//!   (monotonic, pre-gate);
//! - `materialize` resolves an HLLSet through the LUT and ranks the
//!   candidates by TF (ties broken by byte order).

use std::collections::HashMap;

use hllset_attn::{KStorage, TokenLutStorage};
use hllset_core::HLLSet;

/// Monotonic TF reverse index over encoding IDs.
#[derive(Clone, Debug, Default)]
pub struct TfLut {
    storage: TokenLutStorage,
    tf: HashMap<Vec<u8>, u64>,
}

impl TfLut {
    /// Create an empty TF-LUT.
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern an encoding ID (append-only, idempotent).
    pub fn register(&mut self, id: Vec<u8>) {
        self.storage.insert(id.clone());
        self.tf.entry(id).or_insert(0);
    }

    /// Accumulate term frequency for a batch of ingested IDs.
    pub fn observe(&mut self, ids: &[Vec<u8>]) {
        for id in ids {
            self.register(id.clone());
            *self.tf.get_mut(id).expect("registered above") += 1;
        }
    }

    /// Term frequency of an encoding ID (0 if never observed).
    pub fn tf(&self, id: &[u8]) -> u64 {
        self.tf.get(id).copied().unwrap_or(0)
    }

    /// Whether the ID is registered in the forward map (exact LUT gate).
    pub fn known(&self, id: &[u8]) -> bool {
        self.tf.contains_key(id)
    }

    /// Materialize an HLLSet through the LUT, TF-ranked (ties: byte order).
    pub fn materialize(&self, hllset: &HLLSet) -> Vec<Vec<u8>> {
        let mut candidates = self.storage.candidates(hllset);
        candidates.sort_by(|a, b| {
            self.tf(b)
                .cmp(&self.tf(a))
                .then_with(|| a.cmp(b))
        });
        candidates
    }

    /// Coverage gauge over `hllset` (1.0 iff every bit resolves).
    pub fn confidence(&self, hllset: &HLLSet) -> f64 {
        self.storage.confidence(hllset)
    }

    /// Number of registered encoding IDs.
    pub fn len(&self) -> usize {
        self.tf.len()
    }

    /// Whether the LUT is empty.
    pub fn is_empty(&self) -> bool {
        self.tf.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tf_ranking_orders_materialization() {
        let mut lut = TfLut::new();
        lut.observe(&[b"a".to_vec(), b"a".to_vec(), b"a".to_vec(), b"b".to_vec()]);
        let doc = HLLSet::from_tokens(&[b"a".as_slice(), b"b".as_slice()]);
        let restored = lut.materialize(&doc);
        assert_eq!(restored, vec![b"a".to_vec(), b"b".to_vec()]);
        assert_eq!(lut.tf(b"a"), 3);
        assert_eq!(lut.tf(b"b"), 1);
    }

    #[test]
    fn register_is_idempotent() {
        let mut lut = TfLut::new();
        lut.register(b"x".to_vec());
        lut.register(b"x".to_vec());
        assert_eq!(lut.len(), 1);
        assert!(lut.known(b"x"));
    }
}
