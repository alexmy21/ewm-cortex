//! `gate_TF` — the decoder-vocabulary sketch gate.
//!
//! Built once from the decoder's valid encoding-ID vocabulary and applied
//! at the lattice level by intersection: `doc ∩ gate_TF`. The sketch gate is
//! a probabilistic filter; the authoritative membership check is the exact
//! LUT forward map (`Gate::exact_known`), per the reference design.

use hllset_core::HLLSet;

/// The gate: a content-addressed HLLSet over the valid vocabulary plus the
/// exact membership list.
#[derive(Clone, Debug)]
pub struct Gate {
    /// `HLLSet::from_tokens(valid ids)` — the sketch gate.
    pub hllset: HLLSet,
    /// Sorted valid encoding IDs (exact membership, `0 leak / 0 FN`).
    vocab: Vec<Vec<u8>>,
}

impl Gate {
    /// Build the gate from the decoder vocabulary.
    pub fn from_vocab<I, B>(vocab: I) -> Self
    where
        I: IntoIterator<Item = B>,
        B: AsRef<[u8]>,
    {
        let mut vocab: Vec<Vec<u8>> = vocab.into_iter().map(|v| v.as_ref().to_vec()).collect();
        vocab.sort();
        vocab.dedup();
        let hllset = HLLSet::from_tokens(vocab.iter());
        Self { hllset, vocab }
    }

    /// Exact membership: is this encoding ID in the decoder vocabulary?
    pub fn exact_known(&self, id: &[u8]) -> bool {
        self.vocab.binary_search_by(|v| v.as_slice().cmp(id)).is_ok()
    }

    /// Apply the sketch gate: `doc ∩ gate_TF`.
    pub fn apply(&self, doc: &HLLSet) -> HLLSet {
        doc.intersection(&self.hllset)
    }

    /// Number of valid encoding IDs.
    pub fn len(&self) -> usize {
        self.vocab.len()
    }

    /// Whether the gate vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.vocab.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_filters_unknown_bits() {
        let gate = Gate::from_vocab(["tid0", "tid1"]);
        let doc = HLLSet::from_tokens(["tid0", "tid999"]);
        let gated = gate.apply(&doc);
        // Every surviving bit must belong to a valid id's positions.
        assert!(gated.popcount() <= doc.popcount());
        assert!(gate.exact_known(b"tid0"));
        assert!(!gate.exact_known(b"tid999"));
    }

    #[test]
    fn gate_is_deterministic() {
        let g1 = Gate::from_vocab(["a", "b", "c"]);
        let g2 = Gate::from_vocab(["c", "b", "a"]);
        assert_eq!(g1.hllset.popcount(), g2.hllset.popcount());
        assert_eq!(g1.len(), 3);
    }
}
