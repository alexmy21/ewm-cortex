//! Phase 0 training harness.
//!
//! Trains the hand-rolled reproduction transformer on a small original
//! character-level corpus and prints loss, final train accuracy, and a
//! greedy sample. Run with:
//!
//! ```text
//! cargo run -p hllset-repro --release
//! ```

use hllset_repro::{Adam, CharDataset, Config, Transformer, CORPUS};

fn main() {
    let block = 32usize;
    let steps = 800usize;
    let lr = 1e-3f32;
    let report_every = 100usize;

    let dataset = CharDataset::from_text(CORPUS);
    let cfg = Config::small(dataset.vocab_size(), block);
    let model = Transformer::new(cfg.clone(), 42);
    let mut opt = Adam::new(lr);

    println!("Phase 0 reproduction transformer");
    println!("  vocab={} chars={:?}", dataset.vocab_size(), dataset.chars());
    println!("  corpus_chars={}  d={}  heads={}  layers={}  block={}  ffn={}",
        dataset.data().len(), cfg.d, cfg.heads, cfg.layers, cfg.block, cfg.ffn);
    println!("  steps={steps}  lr={lr}\n");

    let data = dataset.data();
    let max_start = data.len() - block - 1;

    let mut rng = hllset_repro::XorShift::new(7);
    for step in 0..steps {
        let start = rng.index(max_start);
        let inputs = data[start..start + block].to_vec();
        let targets = data[start + 1..start + 1 + block].to_vec();
        let loss = model.train_step(&mut opt, &inputs, &targets);

        if step == 0 || (step + 1) % report_every == 0 {
            println!("step {:>4}: loss {:.4}", step + 1, loss);
        }
    }

    // Final train accuracy on a fixed window.
    let start = 0;
    let inputs = data[start..start + block].to_vec();
    let targets = data[start + 1..start + 1 + block].to_vec();
    let logits = model.forward(&inputs);
    let logits_data = logits.borrow().data.clone();
    let n = dataset.vocab_size();
    let mut correct = 0usize;
    for (i, &t) in targets.iter().enumerate() {
        let row = &logits_data[i * n..(i + 1) * n];
        let pred = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(j, _)| j)
            .unwrap_or(0);
        if pred == t {
            correct += 1;
        }
    }
    println!("\ntrain accuracy (first window): {correct}/{block}");

    // Greedy sample.
    let prompt = dataset.encode("the cat");
    let sampled = model.sample(&prompt, 80);
    println!("\nsample: {}", dataset.decode(&sampled));
}
