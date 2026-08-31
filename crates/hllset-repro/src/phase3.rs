//! Phase 3 — set-key / hybrid attention via the MoE-ETT resolution.
//!
//! Implements the formal definitions of `HLLSET_K_SPACE_MATH.md` §8 over a
//! conversation corpus:
//!
//! 1. observations = sentence HLLSets (original, unordered by time);
//! 2. MoE = convolution after the SHA1 IICA shuffle;
//! 3. ETT / EL by pure BSS coverage of the recent observation;
//! 4. resolution A: `F(t) = EL ∪ { E_k : ρ_k ≥ τ_min }`;
//! 5. resolution B: materialize `F(t)` and TF-rank the tokens by their
//!    frequency in the source sentences of the included experts.

use std::collections::HashMap;

use hllset_attn::{KStorage, MoE, TokenLutStorage};
use hllset_core::HLLSet;

/// Result of the Phase 3 pipeline.
#[derive(Clone, Debug)]
pub struct Phase3Report {
    pub observations: usize,
    pub experts: usize,
    pub leader_cid: String,
    pub leader_relevance: f64,
    pub resolved_bits: u64,
    pub resolved_tokens: usize,
    /// TF-ranked tokens from the resolved sub-lattice (resolution B).
    pub top_tokens: Vec<(String, usize)>,
    /// LUT coverage over `F(t)` (invariant: 1.0).
    pub coverage: f64,
}

/// Split a text into sentence token lists (whitespace tokenization).
fn sentence_tokens(text: &str) -> Vec<Vec<Vec<u8>>> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split_whitespace()
                .map(|w| w.as_bytes().to_vec())
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Run the MoE-ETT pipeline over the corpus sentences.
pub fn run_phase3(
    corpus: &str,
    width: usize,
    stride: usize,
    tau_min: f64,
    recent_window: usize,
) -> Phase3Report {
    let sentences = sentence_tokens(corpus);
    assert!(sentences.len() >= 3, "need at least 3 sentences");

    // Original observations.
    let observations: Vec<HLLSet> = sentences
        .iter()
        .map(|tokens| HLLSet::from_tokens(tokens.iter()))
        .collect();

    // Recent observation window (the only temporal step).
    let recent_start = sentences.len().saturating_sub(recent_window);
    let mut recent = HLLSet::new();
    for obs in &observations[recent_start..] {
        recent.merge(obs);
    }

    // MoE + ETT + EL + resolution A.
    let moe = MoE::from_observations(&observations, width, stride, None);
    let ett = moe.think_tank(&recent);
    let leader = &ett[0];
    let resolved = moe.resolve(&recent, tau_min);

    // Resolution B: TF-rank the materialized tokens by their frequency in
    // the source sentences of the included experts (leader + ρ ≥ τ_min).
    let mut included_indices: Vec<usize> = Vec::new();
    for ranked in &ett {
        if included_indices.is_empty() || ranked.relevance >= tau_min {
            included_indices.extend(ranked.expert.frame.iter().copied());
        } else {
            break;
        }
    }
    let mut tf: HashMap<Vec<u8>, usize> = HashMap::new();
    for idx in &included_indices {
        for token in &sentences[*idx] {
            *tf.entry(token.clone()).or_insert(0) += 1;
        }
    }

    // Materialize F(t) through the LUT.
    let all_tokens: Vec<&Vec<u8>> = sentences.iter().flatten().collect();
    let storage = TokenLutStorage::from_tokens(all_tokens.iter().copied());
    let candidates = storage.candidates(&resolved);
    let coverage = storage.confidence(&resolved);

    let mut ranked: Vec<(String, usize)> = candidates
        .iter()
        .map(|t| (String::from_utf8_lossy(t).to_string(), tf.get(t).copied().unwrap_or(0)))
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    ranked.truncate(10);

    Phase3Report {
        observations: observations.len(),
        experts: moe.experts.len(),
        leader_cid: leader.expert.cid.clone(),
        leader_relevance: leader.relevance,
        resolved_bits: resolved.popcount(),
        resolved_tokens: candidates.len(),
        top_tokens: ranked,
        coverage,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CORPUS: &str = "\
the cat sat on the mat
the dog ran in the sun
the cat purred on the mat
the dog chased the cat
";

    #[test]
    fn pipeline_is_deterministic_and_covered() {
        let a = run_phase3(CORPUS, 2, 1, 0.5, 2);
        let b = run_phase3(CORPUS, 2, 1, 0.5, 2);
        assert_eq!(a.leader_cid, b.leader_cid, "IICA pipeline must be deterministic");
        assert_eq!(a.resolved_bits, b.resolved_bits);
        assert_eq!(a.coverage, 1.0, "coverage invariant over F(t)");
        assert!(a.resolved_tokens > 0);
    }

    #[test]
    fn leader_explains_recent_observation() {
        let report = run_phase3(CORPUS, 2, 1, 0.5, 1);
        assert!(report.leader_relevance > 0.0);
        assert!(!report.top_tokens.is_empty());
    }
}
