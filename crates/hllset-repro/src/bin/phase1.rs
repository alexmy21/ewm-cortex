//! Phase 1 harness — K-storage attach.
//!
//! Run with:
//!
//! ```text
//! cargo run -p hllset-repro --bin phase1
//! ```

use hllset_attn::{ContextVocabulary, KStorage, TokenLutStorage};
use hllset_core::HLLSet;
use hllset_repro::{attach, CharDataset, CORPUS};

fn main() {
    let dataset = CharDataset::from_text(CORPUS);
    let report = attach(&dataset);

    println!("Phase 1 — K-storage attach");
    println!("  corpus vocab          : {} characters", report.vocab_size);
    println!("  corpus bits           : {}", report.corpus_bits);
    println!("  coverage (invariant)  : {:.3} {}", report.coverage, if report.coverage == 1.0 { "✓" } else { "✗" });
    println!();
    println!("  corpus collisions:");
    println!("    occupied cells      : {}", report.corpus.occupied_cells);
    println!("    max group size      : {}", report.corpus.max_group_size);
    println!("    observed pairs      : {}", report.corpus.observed_pairs);
    println!("    expected pairs      : {:.2}", report.corpus.expected_pairs);
    println!();
    println!("  synthetic 5000-token vocabulary:");
    println!("    occupied cells      : {}", report.synthetic.occupied_cells);
    println!("    observed pairs      : {}", report.synthetic.observed_pairs);
    println!("    expected pairs      : {:.2}", report.synthetic.expected_pairs);
    println!("    observed prob       : {:.6}", report.synthetic.observed_prob);
    println!("    theory prob         : {:.6}", 1.0 / 3072.0);
    println!("    matches theory      : {}", if report.synthetic_matches_theory { "✓" } else { "✗" });

    // Demo: instrument a few tokens with their bit addresses.
    println!();
    println!("  token instrumentation (key_of → bit address):");
    let storage = TokenLutStorage::from_tokens(dataset.chars().iter().map(|c| c.to_string().into_bytes()));
    for token in ["t", "h", "e", " ", "."] {
        if let hllset_attn::KeyRef::Cell(bit) = storage.key_of(token.as_bytes()) {
            let (reg, tz) = (bit / 32, bit % 32);
            println!("    {token:>3} -> bit {bit:>5}  (reg {reg}, tz {tz})");
        }
    }

    // Demo: bounded vocabulary grows monotonically over the corpus sentences.
    println!();
    println!("  bounded vocabulary over corpus sentences (ContextVocabulary::accumulate):");
    let mut vocab = ContextVocabulary::new(TokenLutStorage::new());
    for (i, sentence) in CORPUS.lines().enumerate() {
        let tokens: Vec<Vec<u8>> = sentence
            .split_whitespace()
            .map(|w| w.as_bytes().to_vec())
            .collect();
        for t in &tokens {
            vocab.register(t.clone());
        }
        vocab.accumulate(&HLLSet::from_tokens(tokens.iter()));
        if i % 5 == 4 {
            println!("    after {:>2} sentences: |V(t)| = {:>2}  coverage = {:.2}",
                i + 1, vocab.len(), vocab.coverage());
        }
    }
    println!("    final |V(t)| = {}  coverage = {:.2}", vocab.len(), vocab.coverage());
}
