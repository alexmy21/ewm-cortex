//! Phase 2 harness — address-key attention (Mode A).
//!
//! Run with:
//!
//! ```text
//! cargo run -p hllset-repro --bin phase2 --release
//! ```

use hllset_repro::{bits_for_dataset, run_phase2, CharDataset, CORPUS};

fn main() {
    let dataset = CharDataset::from_text(CORPUS);
    let bits = bits_for_dataset(&dataset);

    println!("Phase 2 — address-key attention (Mode A)");
    println!("  vocab = {} chars, bits per token = {}", dataset.vocab_size(), bits.len());
    println!("  sample bits: {:?}", &bits[..8.min(bits.len())]);
    println!();

    let steps = 800usize;
    let lr = 1e-3f32;
    let report = run_phase2(&dataset, steps, lr);

    println!("  steps          : {}", report.steps);
    println!("  baseline loss  : {:.4}  (learned K, Phase 0)", report.baseline_loss);
    println!("  address loss   : {:.4}  (K = E_bit[bit(t)] + K_pos[t])", report.address_loss);
    println!(
        "  gap ratio      : {:+.2}%  {}",
        report.gap_ratio * 100.0,
        if report.passes { "✓ within 5%" } else { "✗ exceeds 5%" }
    );
}
