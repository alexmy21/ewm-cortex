//! `gate_TF` — the output TokenGate (decoder-vocabulary limit).
//!
//! Built once from the decoder's valid encoding-ID vocabulary. The gate acts
//! **only on the output**: materialized IDs are filtered to the vocabulary,
//! and out-of-vocab IDs are reported. The authoritative membership check is
//! the exact vocabulary list (`Gate::exact_known`) — 0 leak / 0 FN.
//!
//! A sketch HLLSet (`HLLSet::from_tokens(vocab)`) is kept as an optional
//! bit-level pre-filter for future extensions, but it is NOT part of the
//! main path — the LUT and TF-LUT are never gated.

use hllset_core::HLLSet;

/// The gate: a content-addressed HLLSet over the valid vocabulary plus the
/// exact membership list.
#[derive(Clone, Debug)]
pub struct Gate {
    /// `HLLSet::from_tokens(valid ids)` — optional sketch pre-filter.
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

    /// The output TokenGate: keep only the materialized IDs that are in the
    /// decoder vocabulary.
    ///
    /// This is the only place the gate acts in the main pipeline — the LUT
    /// and TF-LUT are never gated, only the restored output is.
    pub fn filter_tokens(&self, ids: &[Vec<u8>]) -> Vec<Vec<u8>> {
        ids.iter()
            .filter(|id| self.exact_known(id))
            .cloned()
            .collect()
    }

    /// The out-of-vocab IDs among the materialized tokens (reported, never
    /// silently dropped).
    pub fn out_of_vocab(&self, ids: &[Vec<u8>]) -> Vec<Vec<u8>> {
        ids.iter()
            .filter(|id| !self.exact_known(id))
            .cloned()
            .collect()
    }

    /// The sketch gate `doc ∩ gate_TF` — an optional bit-level pre-filter
    /// for future extensions; NOT part of the main (output-gated) path.
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
