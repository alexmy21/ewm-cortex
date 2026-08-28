//! Phase 2 — address-key attention (Mode A).
//!
//! Trains two transformers on the same corpus and identical training
//! schedule:
//!
//! - **baseline** — Phase 0 learned K (`k = x W_K` per head);
//! - **address** — Phase 2 Mode A (`k_t = E_bit[bit(t)] + K_pos[t]`), where
//!   `bit(t)` comes from the HLLSet K-storage (`TokenLutStorage::key_of`).
//!
//! Exit criterion: the address model reaches within 5% of the baseline's
//! final loss on the same corpus.

use hllset_attn::{KStorage, KeyRef, TokenLutStorage};
use hllset_core::HLLSet;

use crate::autograd::{Adam, XorShift};
use crate::data::CharDataset;
use crate::model::{Config, Transformer};

/// Phase 2 acceptance threshold: relative gap allowed vs the baseline.
pub const MAX_GAP: f32 = 0.05;

/// The HLLSet bit cell of every token id in the dataset vocabulary.
pub fn bits_for_dataset(dataset: &CharDataset) -> Vec<u32> {
    let storage = TokenLutStorage::from_tokens(
        dataset.chars().iter().map(|c| c.to_string().into_bytes()),
    );
    dataset
        .chars()
        .iter()
        .map(|c| match storage.key_of(c.to_string().as_bytes()) {
            KeyRef::Cell(bit) => bit,
            other => panic!("expected tier-1 cell, got {other:?}"),
        })
        .collect()
}

/// Result of the Phase 2 comparison.
#[derive(Clone, Debug)]
pub struct Phase2Report {
    pub steps: usize,
    pub baseline_loss: f32,
    pub address_loss: f32,
    /// `(address - baseline) / baseline`.
    pub gap_ratio: f32,
    /// Whether the gap is within [`MAX_GAP`].
    pub passes: bool,
}

/// Train `model` for `steps` steps on a deterministic window schedule and
/// return the average loss over the last 50 steps.
pub fn train_to_loss(
    model: &Transformer,
    data: &[usize],
    block: usize,
    steps: usize,
    lr: f32,
    rng_seed: u32,
) -> f32 {
    let mut opt = Adam::new(lr);
    let mut rng = XorShift::new(rng_seed);
    let max_start = data.len().saturating_sub(block + 1).max(1);
    let mut last = Vec::new();
    for step in 0..steps {
        let start = rng.index(max_start);
        let inputs = data[start..start + block].to_vec();
        let targets = data[start + 1..start + 1 + block].to_vec();
        let loss = model.train_step(&mut opt, &inputs, &targets);
        if step + 1 > steps.saturating_sub(50) {
            last.push(loss);
        }
    }
    last.iter().sum::<f32>() / last.len().max(1) as f32
}

/// Run the Phase 2 comparison on a dataset.
pub fn run_phase2(dataset: &CharDataset, steps: usize, lr: f32) -> Phase2Report {
    let cfg = Config::small(dataset.vocab_size(), 32);
    let data = dataset.data();
    let block = cfg.block;

    // Phase 1 sanity: the corpus vocabulary covers the full sub-lattice.
    let bits = bits_for_dataset(dataset);
    let storage = TokenLutStorage::from_tokens(
        dataset.chars().iter().map(|c| c.to_string().into_bytes()),
    );
    let coverage = storage.confidence(&HLLSet::from_tokens(
        dataset.chars().iter().map(|c| c.to_string().into_bytes()),
    ));
    assert_eq!(coverage, 1.0, "coverage invariant violated before Phase 2");

    let baseline = Transformer::new(cfg.clone(), 42);
    let baseline_loss = train_to_loss(&baseline, data, block, steps, lr, 7);

    let address = Transformer::new_address(cfg, 42, &bits);
    let address_loss = train_to_loss(&address, data, block, steps, lr, 7);

    let gap_ratio = (address_loss - baseline_loss) / baseline_loss;
    Phase2Report {
        steps,
        baseline_loss,
        address_loss,
        gap_ratio,
        passes: gap_ratio <= MAX_GAP,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_cover_vocabulary_in_range() {
        let ds = CharDataset::from_text("abc");
        let bits = bits_for_dataset(&ds);
        assert_eq!(bits.len(), 3);
        assert!(bits.iter().all(|&b| (b as usize) < crate::model::TOTAL_CELLS));
    }

    #[test]
    fn address_model_approaches_baseline() {
        // Natural corpus, short run: the address-key model must stay within
        // a loose relative margin of the learned-K baseline.
        let ds = CharDataset::from_text(crate::corpus::CORPUS);
        let report = run_phase2(&ds, 150, 1e-3);
        assert!(report.baseline_loss.is_finite());
        assert!(report.address_loss.is_finite());
        assert!(
            report.gap_ratio <= 0.5,
            "gap too large: baseline={} address={} gap={}",
            report.baseline_loss,
            report.address_loss,
            report.gap_ratio
        );
    }
}
