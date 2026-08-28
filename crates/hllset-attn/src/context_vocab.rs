//! Incremental context vocabulary maintenance.
//!
//! The context sub-lattice evolves through commits `C(t) = (S(t), C(t-1),
//! D, R, N)` with `D = C(t-1) \ S(t)` (departed), `R = C(t-1) ∩ S(t)`
//! (retained), `N = S(t) \ C(t-1)` (new). Only the new bits carry unknown
//! vocabulary, so instead of re-materializing the whole context union,
//! maintain the token vocabulary incrementally:
//!
//! ```text
//! V(t) = ( V(t-1) \ M(D(t)) )  ∪  M(N(t))
//! ```
//!
//! where `M = KStorage::candidates`. Per-step work is proportional to the
//! churn `|N ∪ D|`, not to `|C(t)|` — the natural bound on context
//! vocabulary.
//!
//! # Two gates over one LUT
//!
//! `V(t)` is a [`TokenMask`]: a gate of **content addresses** (murmur3
//! seed-0 hashes), not token bytes. The bytes live only in the LUT; the
//! mask references them by address, so the vocabulary view can never drift
//! out of sync with the K-storage. The context HLLSet is the twin **bit
//! gate** (compact, lossy); the mask is exact token identity.
//!
//! # Exactness
//!
//! For tier 1 (single-seed `TokenLutStorage`) removals are exact at bit
//! granularity and additions are exact up to collision groups (the
//! structural 2-/3-gram layers resolve the groups). For tier 2
//! (`CatalogLutStorage`, 3 seeds, 2-of-3 quorum) both directions are exact:
//! a value is added iff `|pos(v) ∩ N| ≥ quorum` and removed iff
//! `|pos(v) ∩ D| ≥ quorum`, which coincides with presence in `C(t)`.
//!
//! # Append-only LUT
//!
//! `M(D(t))` is only computable if the LUT still remembers departed tokens.
//! The LUT is therefore **append-only**: departure is a lattice operation
//! (bits leave the context sub-lattice), never a LUT deletion. `V(t)` is a
//! derived view — the only place removal actually happens.

use hllset_core::core::hashing::murmur3_hash;
use hllset_core::HLLSet;

use crate::kstorage::KStorage;
use crate::token_mask::TokenMask;

/// The vocabulary change produced by one commit.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VocabDelta {
    /// Tokens added by `M(N(t))` (sorted, deduplicated).
    pub added: Vec<Vec<u8>>,
    /// Tokens removed by `M(D(t))` (sorted, deduplicated).
    pub removed: Vec<Vec<u8>>,
}

impl VocabDelta {
    /// Whether the commit changed nothing.
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty()
    }
}

/// Incremental token-vocabulary maintainer over a context sub-lattice.
///
/// Holds the current context `C(t)` as an [`HLLSet`] (the bit gate) and the
/// derived vocabulary as a [`TokenMask`] (the hash gate); the K-storage LUT
/// is the single canonical store of token bytes.
#[derive(Clone, Debug)]
pub struct ContextVocabulary<S: KStorage> {
    storage: S,
    /// Current context sub-lattice C(t) — the bit gate.
    context: HLLSet,
    /// Derived token vocabulary V(t) — the hash gate.
    mask: TokenMask,
}

impl<S: KStorage> ContextVocabulary<S> {
    /// Create an empty vocabulary over the given K-storage.
    pub fn new(storage: S) -> Self {
        Self {
            storage,
            context: HLLSet::new(),
            mask: TokenMask::new(),
        }
    }

    /// Rebuild from an existing context sub-lattice: `V = M(context)`.
    pub fn from_context(storage: S, context: HLLSet) -> Self {
        let mut mask = TokenMask::new();
        for token in storage.candidates(&context) {
            mask.insert(murmur3_hash(&token));
        }
        Self {
            storage,
            context,
            mask,
        }
    }

    /// Register a token in the append-only LUT and, if its key is already
    /// present in the context, gate it into V immediately.
    ///
    /// This keeps the invariant `V == M(C(t))` for tier 1: `accumulate`
    /// materializes only the new bits, so a token that collides with an
    /// already-present bit would otherwise be missed. Registering before
    /// accumulating handles both the new-bit case (via `M(N)`) and the
    /// existing-bit case (via this method).
    pub fn register(&mut self, token: Vec<u8>) {
        let hash = murmur3_hash(&token);
        let key = self.storage.key_of(&token);
        self.storage.register(token);
        let present = match &key {
            crate::kstorage::KeyRef::Cell(bit) => self.context.bitmap().contains(*bit),
            crate::kstorage::KeyRef::Cells(cells, quorum) => cells
                .iter()
                .filter(|c| self.context.bitmap().contains(**c))
                .count()
                >= *quorum,
        };
        if present {
            self.mask.insert(hash);
        }
    }

    /// The K-storage (LUT).
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// The current context sub-lattice C(t) — the bit gate.
    pub fn context(&self) -> &HLLSet {
        &self.context
    }

    /// The hash gate V(t): content addresses of the current vocabulary.
    pub fn mask(&self) -> &TokenMask {
        &self.mask
    }

    /// The current token vocabulary V(t), materialized from the LUT in
    /// deterministic (hash-sorted) order. Bytes are resolved on demand;
    /// they are never stored in the mask.
    pub fn tokens(&self) -> Vec<Vec<u8>> {
        self.mask
            .sorted_hashes()
            .into_iter()
            .filter_map(|hash| self.storage.resolve(hash))
            .collect()
    }

    /// Whether `token` is in V(t) (by content address).
    pub fn contains(&self, token: &[u8]) -> bool {
        self.mask.contains_token(token)
    }

    /// Number of tokens in V(t).
    pub fn len(&self) -> usize {
        self.mask.len()
    }

    /// Whether the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.mask.is_empty()
    }

    /// Coverage gauge over C(t): `1.0` iff every context bit is resolvable
    /// through the LUT.
    pub fn coverage(&self) -> f64 {
        self.storage.confidence(&self.context)
    }

    /// Panic if the coverage invariant is violated.
    pub fn assert_coverage(&self) {
        let c = self.coverage();
        assert_eq!(c, 1.0, "coverage invariant violated: confidence={c}");
    }

    /// Advance with **commit semantics**: `C(t) = S(t)`.
    ///
    /// `N = S \ C(t-1)`, `D = C(t-1) \ S`. The vocabulary update is
    /// `V(t) = (V(t-1) \ M(D)) ∪ M(N)`. Returns the applied delta.
    pub fn advance(&mut self, source: &HLLSet) -> VocabDelta {
        let n = source.difference(&self.context);
        let d = self.context.difference(source);
        self.context = source.clone();
        self.apply_delta(&n, &d)
    }

    /// Accumulate with **union semantics**: `C(t) = C(t-1) ∪ S(t)`
    /// (the conversation-context case). `D = ∅`; `N = S \ C(t-1)`.
    pub fn accumulate(&mut self, source: &HLLSet) -> VocabDelta {
        let n = source.difference(&self.context);
        let d = HLLSet::new();
        self.context.merge(source);
        self.apply_delta(&n, &d)
    }

    /// Apply a pre-computed D/R/N delta: `V = (V \ M(d)) ∪ M(n)`.
    pub fn apply_delta(&mut self, n: &HLLSet, d: &HLLSet) -> VocabDelta {
        let mut added = self.storage.candidates(n);
        added.sort();
        added.dedup();

        let mut removed = self.storage.candidates(d);
        removed.sort();
        removed.dedup();

        let mut delta = VocabDelta::default();
        for token in added {
            if self.mask.insert(murmur3_hash(&token)) {
                delta.added.push(token);
            }
        }
        for token in removed {
            if self.mask.remove(murmur3_hash(&token)) {
                delta.removed.push(token);
            }
        }
        delta
    }
}

impl<S: KStorage + Default> Default for ContextVocabulary<S> {
    fn default() -> Self {
        Self::new(S::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kstorage::{CatalogLutStorage, TokenLutStorage};
    use hllset_core::core::hashing::murmur3_hash_seeded;
    use std::collections::HashSet;

    #[test]
    fn tier1_advance_replaces_vocabulary() {
        let storage = TokenLutStorage::from_tokens(&["alpha", "beta", "gamma"]);
        let mut vocab = ContextVocabulary::from_context(
            storage,
            HLLSet::from_tokens(&["alpha", "beta"]),
        );
        assert_eq!(vocab.len(), 2);
        assert!(vocab.contains(b"alpha"));
        assert!(vocab.contains(b"beta"));
        assert_eq!(vocab.coverage(), 1.0);

        let delta = vocab.advance(&HLLSet::from_tokens(&["beta", "gamma"]));
        assert_eq!(delta.added, vec![b"gamma".to_vec()]);
        assert_eq!(delta.removed, vec![b"alpha".to_vec()]);
        assert_eq!(vocab.len(), 2);
        assert!(vocab.contains(b"beta"));
        assert!(vocab.contains(b"gamma"));
        assert!(!vocab.contains(b"alpha"));
    }

    #[test]
    fn tier1_accumulate_only_adds() {
        let storage = TokenLutStorage::from_tokens(&["alpha", "beta", "gamma"]);
        let mut vocab =
            ContextVocabulary::from_context(storage, HLLSet::from_tokens(&["alpha"]));

        let delta = vocab.accumulate(&HLLSet::from_tokens(&["alpha", "beta"]));
        assert!(delta.removed.is_empty());
        assert_eq!(delta.added, vec![b"beta".to_vec()]);
        assert!(vocab.contains(b"alpha"));
        assert!(vocab.contains(b"beta"));
    }

    #[test]
    fn register_keeps_collision_partner_in_vocabulary() {
        use hllset_core::core::hashing::token_to_position;
        use hllset_core::BITS_PER_REG;

        // Find two distinct tokens that hash to the same bit.
        let mut seen: std::collections::HashMap<u32, Vec<u8>> = std::collections::HashMap::new();
        let pair = (0..5000).find_map(|i| {
            let token = format!("w{i}").into_bytes();
            let (reg, zeros) = token_to_position(&token);
            let bit = reg * BITS_PER_REG + zeros;
            if let Some(prev) = seen.get(&bit) {
                Some((prev.clone(), token))
            } else {
                seen.insert(bit, token);
                None
            }
        });
        let Some((a, b)) = pair else {
            return; // astronomically unlikely with 5000 candidates
        };
        assert_eq!(token_to_position(&a), token_to_position(&b));

        let mut vocab = ContextVocabulary::new(TokenLutStorage::new());

        // First token: its bit is new, so accumulate adds it via M(N).
        vocab.register(a.clone());
        vocab.accumulate(&HLLSet::from_tokens(&[a.as_slice()]));
        assert!(vocab.contains(&a));

        // Second token: same bit, already present in the context. Without
        // the register-time fix, accumulate would miss it (N is empty).
        vocab.register(b.clone());
        vocab.accumulate(&HLLSet::from_tokens(&[b.as_slice()]));
        assert!(vocab.contains(&b));
    }

    #[test]
    fn tier1_advance_matches_full_materialization() {
        let storage = TokenLutStorage::from_tokens(&["a", "b", "c", "d"]);
        let mut vocab = ContextVocabulary::from_context(
            storage,
            HLLSet::from_tokens(&["a", "b", "c"]),
        );
        vocab.advance(&HLLSet::from_tokens(&["c", "d"]));
        let full: HashSet<Vec<u8>> = vocab
            .storage()
            .candidates(&HLLSet::from_tokens(&["c", "d"]))
            .into_iter()
            .collect();
        let maintained: HashSet<Vec<u8>> = vocab.tokens().into_iter().collect();
        assert_eq!(maintained, full);
    }

    #[test]
    fn mask_holds_hashes_not_bytes() {
        let storage = TokenLutStorage::from_tokens(&["alpha", "beta"]);
        let vocab = ContextVocabulary::from_context(storage, HLLSet::from_tokens(&["alpha"]));
        assert!(vocab.mask().contains_token(b"alpha"));
        assert!(!vocab.mask().contains_token(b"beta"));
        // bytes are resolved through the LUT, not stored in the mask
        assert_eq!(vocab.tokens(), vec![b"alpha".to_vec()]);
    }

    fn catalog_hllset(values: &[&str]) -> HLLSet {
        let mut h = HLLSet::new();
        for seed in [0u64, 1, 2] {
            for v in values {
                let hash = murmur3_hash_seeded(v.as_bytes(), seed);
                h.add_hash(hash);
            }
        }
        h
    }

    #[test]
    fn tier2_advance_is_exact() {
        let values = ["alice", "bob", "carol", "dave"];
        let storage = CatalogLutStorage::from_values(&values);

        let mut vocab = ContextVocabulary::from_context(
            storage,
            catalog_hllset(&["alice", "bob", "carol"]),
        );
        assert_eq!(vocab.len(), 3);

        // carol departs, dave arrives.
        let delta = vocab.advance(&catalog_hllset(&["alice", "bob", "dave"]));
        assert_eq!(delta.added, vec![b"dave".to_vec()]);
        assert_eq!(delta.removed, vec![b"carol".to_vec()]);
        assert_eq!(vocab.len(), 3);
        assert!(vocab.contains(b"dave"));
        assert!(!vocab.contains(b"carol"));
    }

    #[test]
    fn tier2_accumulate_is_exact() {
        let values = ["x", "y", "z"];
        let storage = CatalogLutStorage::from_values(&values);
        let mut vocab =
            ContextVocabulary::from_context(storage, catalog_hllset(&["x"]));
        let delta = vocab.accumulate(&catalog_hllset(&["x", "y"]));
        assert_eq!(delta.added, vec![b"y".to_vec()]);
        assert!(delta.removed.is_empty());
        assert!(vocab.contains(b"x"));
        assert!(vocab.contains(b"y"));
        assert!(!vocab.contains(b"z"));
    }
}
