//! Phase 1 — K-storage attach.
//!
//! Attaches the HLLSet K-storage (`TokenLutStorage`) to the reproduction
//! corpus vocabulary and verifies the two Phase 1 exit criteria:
//!
//! 1. **Coverage invariant** — every corpus bit resolves through the LUT
//!    (`confidence == 1.0`).
//! 2. **Collision statistics** — the observed pairwise collision rate
//!    matches the theory (`p = 1/3072`, hence `C(n,2)/3072` expected pairs).
//!
//! The corpus vocabulary is tiny (23 characters), so a synthetic 5000-token
//! vocabulary is also measured to exercise the theory at scale.

use std::collections::HashMap;

use hllset_attn::{KStorage, TokenLutStorage};
use hllset_core::HLLSet;

use crate::data::CharDataset;

/// Pairwise collision probability of the HLLSet partition (theory, §4 of
/// `_DOCS/dev/HLLSET_K_SPACE_MATH.md`).
pub const EXPECTED_PAIRWISE_PROB: f64 = 1.0 / 3072.0;

/// Collision statistics of a token vocabulary under the HLLSet partition.
#[derive(Clone, Debug)]
pub struct CollisionStats {
    /// Number of distinct tokens measured.
    pub vocab_size: usize,
    /// Number of occupied bit cells.
    pub occupied_cells: usize,
    /// Largest collision group (tokens sharing one bit).
    pub max_group_size: usize,
    /// Observed number of colliding token pairs `Σ C(group_size, 2)`.
    pub observed_pairs: u64,
    /// Expected number of colliding pairs `C(n,2) / 3072`.
    pub expected_pairs: f64,
    /// Observed pairwise collision probability `2·pairs / (n(n-1))`.
    pub observed_prob: f64,
}

/// Result of the Phase 1 attach over a character dataset.
#[derive(Clone, Debug)]
pub struct Phase1Report {
    /// Corpus vocabulary size (distinct characters).
    pub vocab_size: usize,
    /// Coverage gauge over the full corpus sub-lattice (must be 1.0).
    pub coverage: f64,
    /// Bits set by the full corpus vocabulary.
    pub corpus_bits: u64,
    /// Collision statistics of the corpus vocabulary.
    pub corpus: CollisionStats,
    /// Collision statistics of a synthetic 5000-token vocabulary.
    pub synthetic: CollisionStats,
    /// Whether the synthetic measurement matches the theory within tolerance.
    pub synthetic_matches_theory: bool,
}

/// Measure collision statistics for `tokens` against a tier-1 storage.
pub fn collision_stats(storage: &TokenLutStorage, tokens: &[Vec<u8>]) -> CollisionStats {
    let mut groups: HashMap<u32, usize> = HashMap::new();
    for token in tokens {
        let bit = storage.key_of(token).cells()[0];
        *groups.entry(bit).or_insert(0) += 1;
    }
    let n = tokens.len();
    let observed_pairs: u64 = groups
        .values()
        .map(|&g| (g * (g - 1) / 2) as u64)
        .sum();
    let expected_pairs = (n * (n - 1)) as f64 / 2.0 * EXPECTED_PAIRWISE_PROB;
    CollisionStats {
        vocab_size: n,
        occupied_cells: groups.len(),
        max_group_size: groups.values().copied().max().unwrap_or(0),
        observed_pairs,
        expected_pairs,
        observed_prob: if n > 1 {
            2.0 * observed_pairs as f64 / (n * (n - 1)) as f64
        } else {
            0.0
        },
    }
}

/// Run the Phase 1 attach: build the LUT over the corpus vocabulary,
/// instrument every token, and verify coverage + collision statistics.
pub fn attach(dataset: &CharDataset) -> Phase1Report {
    // The corpus vocabulary: one token per distinct character.
    let tokens: Vec<Vec<u8>> = dataset
        .chars()
        .iter()
        .map(|c| c.to_string().into_bytes())
        .collect();

    let storage = TokenLutStorage::from_tokens(tokens.iter());
    let corpus_hllset = HLLSet::from_tokens(tokens.iter());

    let corpus = collision_stats(&storage, &tokens);

    // Synthetic vocabulary for a statistically meaningful collision check.
    let synthetic_tokens: Vec<Vec<u8>> =
        (0..5000).map(|i| format!("w{i}").into_bytes()).collect();
    let synthetic_storage = TokenLutStorage::from_tokens(synthetic_tokens.iter());
    let synthetic = collision_stats(&synthetic_storage, &synthetic_tokens);
    let synthetic_matches_theory = (synthetic.observed_pairs as f64 - synthetic.expected_pairs).abs()
        <= synthetic.expected_pairs * 0.15
        || (synthetic.expected_pairs < 1.0 && synthetic.observed_pairs <= 1);

    Phase1Report {
        vocab_size: dataset.vocab_size(),
        coverage: storage.confidence(&corpus_hllset),
        corpus_bits: corpus_hllset.popcount(),
        corpus,
        synthetic,
        synthetic_matches_theory,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_is_full_on_corpus_vocabulary() {
        let ds = CharDataset::from_text("abcabc");
        let report = attach(&ds);
        assert_eq!(report.coverage, 1.0);
        assert_eq!(report.vocab_size, 3);
        assert!(report.corpus_bits >= 1);
    }

    #[test]
    fn synthetic_collisions_match_theory() {
        let ds = CharDataset::from_text("hello world");
        let report = attach(&ds);
        assert!(
            report.synthetic_matches_theory,
            "observed={} expected={}",
            report.synthetic.observed_pairs,
            report.synthetic.expected_pairs
        );
        // Sanity: with 5000 tokens the expected pair count is substantial.
        assert!(report.synthetic.expected_pairs > 3000.0);
    }

    #[test]
    fn tiny_vocabulary_has_no_collisions() {
        let storage = TokenLutStorage::from_tokens(&["a", "b", "c"]);
        let tokens: Vec<Vec<u8>> = ["a", "b", "c"].iter().map(|s| s.as_bytes().to_vec()).collect();
        let stats = collision_stats(&storage, &tokens);
        assert_eq!(stats.occupied_cells, 3);
        assert_eq!(stats.observed_pairs, 0);
    }
}
