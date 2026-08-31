//! Phase 3 harness — MoE / ETT / resolution + hybrid gate.
//!
//! Run with:
//!
//! ```text
//! cargo run -p hllset-repro --bin phase3
//! ```

use hllset_attn::MoE;
use hllset_core::HLLSet;
use hllset_repro::{bits_for_dataset, run_phase3, Adam, CharDataset, Config, Transformer, XorShift, CORPUS};

fn top_k(scores: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
    idx.truncate(k);
    idx.into_iter().map(|i| (i, scores[i])).collect()
}

fn main() {
    let width = 3usize;
    let stride = 2usize;
    let tau_min = 0.3f64;
    let recent_window = 2usize;

    let report = run_phase3(CORPUS, width, stride, tau_min, recent_window);

    println!("Phase 3 — MoE / ETT / resolution (A+B)");
    println!("  observations : {}", report.observations);
    println!("  experts      : {}", report.experts);
    println!("  leader cid   : {}…", &report.leader_cid[..8.min(report.leader_cid.len())]);
    println!("  leader ρ     : {:.3}", report.leader_relevance);
    println!("  F(t) bits    : {}", report.resolved_bits);
    println!("  F(t) tokens  : {}", report.resolved_tokens);
    println!("  coverage     : {:.2} {}", report.coverage, if report.coverage == 1.0 { "✓" } else { "✗" });
    println!("  top tokens (TF-ranked, resolution B):");
    for (token, count) in &report.top_tokens {
        println!("    {token:>10} ×{count}");
    }

    // ── Hybrid gate (Mode C): transformer logits ⊕ HLLSet gate ──────────────
    println!();
    println!("  hybrid gate (Mode C): logits + λ · 1[bit ∈ F(t)]");

    let dataset = CharDataset::from_text(CORPUS);
    let bits = bits_for_dataset(&dataset);
    let cfg = Config::small(dataset.vocab_size(), 32);
    let model = Transformer::new_address(cfg, 42, &bits);
    let mut opt = Adam::new(1e-3);
    let data = dataset.data();
    let mut rng = XorShift::new(7);
    for step in 0..400 {
        let start = rng.index(data.len() - 33);
        let inputs = data[start..start + 32].to_vec();
        let targets = data[start + 1..start + 33].to_vec();
        model.train_step(&mut opt, &inputs, &targets);
    }

    // Rebuild the resolved F(t) bitmap for the gate — at CHARACTER level,
    // so the gate's bits live in the same vocabulary as the transformer.
    let char_observations: Vec<HLLSet> = CORPUS
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let chars: Vec<Vec<u8>> = l.chars().map(|c| c.to_string().into_bytes()).collect();
            HLLSet::from_tokens(chars.iter())
        })
        .collect();
    let recent_start = char_observations.len().saturating_sub(recent_window);
    let mut recent = HLLSet::new();
    for obs in &char_observations[recent_start..] {
        recent.merge(obs);
    }
    let moe = MoE::from_observations(&char_observations, width, stride, None);
    let resolved = moe.resolve(&recent, tau_min);

    let prompt = dataset.encode("the cat");
    let logits = model.forward(&prompt);
    let logits_data = logits.borrow().data.clone();
    let n = dataset.vocab_size();
    let base: Vec<f32> = logits_data[(prompt.len() - 1) * n..].to_vec();

    for lambda in [0.0f32, 2.0] {
        let mut gated = base.clone();
        for (tok, &bit) in bits.iter().enumerate() {
            if resolved.bitmap().contains(bit) {
                gated[tok] += lambda;
            }
        }
        let top = top_k(&gated, 5);
        let tokens: Vec<String> = top
            .iter()
            .map(|&(t, s)| format!("{} ({:.2})", dataset.decode(&[t]), s))
            .collect();
        println!("    λ={lambda:.1}: {}", tokens.join(", "));
    }
}
