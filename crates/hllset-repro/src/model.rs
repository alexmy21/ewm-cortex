//! The reproduction transformer (token realm).
//!
//! A standard small decoder-only transformer: learned token + positional
//! embeddings, stacked blocks of causal multi-head attention and an MLP,
//! LayerNorm, tied output embeddings. Everything is composed from the
//! [`autograd`] ops, so the whole model is differentiable and inspectable.
//!
//! Two attention modes are supported:
//!
//! - **Learned K** (Phase 0 baseline): `k = x W_K` per head.
//! - **Address K** (Phase 2, Mode A): `k_t = E_bit[bit(t)] + K_pos[t]` per
//!   head, where `bit(t)` is the token's HLLSet cell address — the key is
//!   served by the HLLSet realm and lifted by a learned `E_bit` table of
//!   `TOTAL_CELLS × d_head` entries. Q and V stay contextual and learned.

use crate::autograd::{
    add, add_bias, backward, gather, layernorm, leaf, matmul, matmul_bt, relu, scale, softmax,
    Tensor, XorShift,
};

/// Number of HLLSet bit cells: `1024 registers × 32 trailing-zero states`.
pub const TOTAL_CELLS: usize = 32768;

/// Transformer configuration.
#[derive(Clone, Debug)]
pub struct Config {
    pub vocab: usize,
    pub d: usize,
    pub heads: usize,
    pub layers: usize,
    pub block: usize,
    pub ffn: usize,
}

impl Config {
    /// Small reproduction-scale defaults.
    pub fn small(vocab: usize, block: usize) -> Self {
        Self {
            vocab,
            d: 64,
            heads: 4,
            layers: 2,
            block,
            ffn: 128,
        }
    }
}

/// One attention head.
enum Head {
    /// Phase 0: learned Q/K/V projections.
    Learned {
        wq: Tensor,
        wk: Tensor,
        wv: Tensor,
        wo: Tensor,
    },
    /// Phase 2 (Mode A): Q/V learned; K = `E_bit[bit(t)] + K_pos[t]`.
    Address {
        wq: Tensor,
        wv: Tensor,
        wo: Tensor,
        /// Learned lift of the HLLSet partition: one key per bit cell.
        ebit: Tensor,
        /// Learned positional key term.
        posk: Tensor,
    },
}

struct Block {
    heads: Vec<Head>,
    ln1_w: Tensor,
    ln1_b: Tensor,
    ln2_w: Tensor,
    ln2_b: Tensor,
    ffn_w1: Tensor,
    ffn_b1: Tensor,
    ffn_w2: Tensor,
    ffn_b2: Tensor,
}

/// A decoder-only transformer built from [`autograd`] tensors.
pub struct Transformer {
    cfg: Config,
    token_emb: Tensor,
    pos_emb: Tensor,
    blocks: Vec<Block>,
    lnf_w: Tensor,
    lnf_b: Tensor,
    /// Token id → HLLSet bit cell (empty in learned mode).
    bits: Vec<u32>,
    params: Vec<Tensor>,
}

impl Transformer {
    /// Build a Phase 0 model with learned K projections.
    pub fn new(cfg: Config, seed: u32) -> Self {
        Self::build(cfg, seed, &[])
    }

    /// Build a Phase 2 (Mode A) model with address keys.
    ///
    /// `bits[tok]` must be the HLLSet cell of token id `tok`
    /// (`0..TOTAL_CELLS`).
    pub fn new_address(cfg: Config, seed: u32, bits: &[u32]) -> Self {
        assert_eq!(bits.len(), cfg.vocab, "bits must cover the vocabulary");
        assert!(
            bits.iter().all(|&b| (b as usize) < TOTAL_CELLS),
            "bit out of range"
        );
        Self::build(cfg, seed, bits)
    }

    fn build(cfg: Config, seed: u32, bits: &[u32]) -> Self {
        let mut rng = XorShift::new(seed);
        let dh = cfg.d / cfg.heads;
        assert!(cfg.d % cfg.heads == 0, "d must divide by heads");
        let address_mode = !bits.is_empty();

        let token_emb = leaf(&[cfg.vocab, cfg.d], |_| rng.uniform(0.1));
        let pos_emb = leaf(&[cfg.block, cfg.d], |_| rng.uniform(0.1));
        let mut params = vec![token_emb.clone(), pos_emb.clone()];

        let mut blocks = Vec::new();
        for _ in 0..cfg.layers {
            let mut heads = Vec::new();
            for _ in 0..cfg.heads {
                let head = if address_mode {
                    let h = Head::Address {
                        wq: leaf(&[cfg.d, dh], |_| rng.uniform(0.1)),
                        wv: leaf(&[cfg.d, dh], |_| rng.uniform(0.1)),
                        wo: leaf(&[dh, cfg.d], |_| rng.uniform(0.1)),
                        ebit: leaf(&[TOTAL_CELLS, dh], |_| rng.uniform(0.1)),
                        posk: leaf(&[cfg.block, dh], |_| rng.uniform(0.1)),
                    };
                    if let Head::Address { wq, wv, wo, ebit, posk } = &h {
                        params.extend([
                            wq.clone(),
                            wv.clone(),
                            wo.clone(),
                            ebit.clone(),
                            posk.clone(),
                        ]);
                    }
                    h
                } else {
                    let h = Head::Learned {
                        wq: leaf(&[cfg.d, dh], |_| rng.uniform(0.1)),
                        wk: leaf(&[cfg.d, dh], |_| rng.uniform(0.1)),
                        wv: leaf(&[cfg.d, dh], |_| rng.uniform(0.1)),
                        wo: leaf(&[dh, cfg.d], |_| rng.uniform(0.1)),
                    };
                    if let Head::Learned { wq, wk, wv, wo } = &h {
                        params.extend([wq.clone(), wk.clone(), wv.clone(), wo.clone()]);
                    }
                    h
                };
                heads.push(head);
            }
            let block = Block {
                heads,
                ln1_w: leaf(&[cfg.d], |_| 1.0),
                ln1_b: leaf(&[cfg.d], |_| 0.0),
                ln2_w: leaf(&[cfg.d], |_| 1.0),
                ln2_b: leaf(&[cfg.d], |_| 0.0),
                ffn_w1: leaf(&[cfg.d, cfg.ffn], |_| rng.uniform(0.1)),
                ffn_b1: leaf(&[cfg.ffn], |_| 0.0),
                ffn_w2: leaf(&[cfg.ffn, cfg.d], |_| rng.uniform(0.1)),
                ffn_b2: leaf(&[cfg.d], |_| 0.0),
            };
            params.extend([
                block.ln1_w.clone(),
                block.ln1_b.clone(),
                block.ln2_w.clone(),
                block.ln2_b.clone(),
                block.ffn_w1.clone(),
                block.ffn_b1.clone(),
                block.ffn_w2.clone(),
                block.ffn_b2.clone(),
            ]);
            blocks.push(block);
        }

        let lnf_w = leaf(&[cfg.d], |_| 1.0);
        let lnf_b = leaf(&[cfg.d], |_| 0.0);
        params.push(lnf_w.clone());
        params.push(lnf_b.clone());

        Self {
            cfg,
            token_emb,
            pos_emb,
            blocks,
            lnf_w,
            lnf_b,
            bits: bits.to_vec(),
            params,
        }
    }

    /// All trainable parameters.
    pub fn params(&self) -> Vec<Tensor> {
        self.params.clone()
    }

    /// Whether the model uses address keys (Mode A).
    pub fn address_mode(&self) -> bool {
        !self.bits.is_empty()
    }

    /// Forward pass: token ids → logits `[T, vocab]`.
    pub fn forward(&self, idx: &[usize]) -> Tensor {
        let t = idx.len();
        let cfg = &self.cfg;
        assert!(t <= cfg.block, "sequence longer than block size");

        // Embeddings via row gathers (sparse: no dense one-hot matmuls).
        let x_tok = gather(&self.token_emb, idx);
        let pos_idx: Vec<usize> = (0..t).collect();
        let x_pos = gather(&self.pos_emb, &pos_idx);
        let mut x = add(&x_tok, &x_pos);

        let mask = causal_mask(t);

        // Address-mode bit cells per position (built once, reused by heads).
        let bits_at_pos: Vec<usize> = if self.address_mode() {
            idx.iter().map(|&tok| self.bits[tok] as usize).collect()
        } else {
            Vec::new()
        };

        for block in &self.blocks {
            // --- Causal multi-head attention ---
            let xl = layernorm(&x, &block.ln1_w, &block.ln1_b);
            let dh = cfg.d / cfg.heads;
            let scale_factor = 1.0 / (dh as f32).sqrt();

            let mut head_outs: Vec<Tensor> = Vec::new();
            for head in &block.heads {
                let head_out = match head {
                    Head::Learned { wq, wk, wv, wo } => {
                        let q = matmul(&xl, wq);
                        let k = matmul(&xl, wk);
                        let v = matmul(&xl, wv);
                        let scores = scale(&matmul_bt(&q, &k), scale_factor);
                        let attn = softmax(&add(&scores, &mask));
                        let ctx = matmul(&attn, &v);
                        matmul(&ctx, wo)
                    }
                    Head::Address { wq, wv, wo, ebit, posk } => {
                        let q = matmul(&xl, wq);
                        let k = add(&gather(ebit, &bits_at_pos), &gather(posk, &pos_idx));
                        let v = matmul(&xl, wv);
                        let scores = scale(&matmul_bt(&q, &k), scale_factor);
                        let attn = softmax(&add(&scores, &mask));
                        let ctx = matmul(&attn, &v);
                        matmul(&ctx, wo)
                    }
                };
                head_outs.push(head_out);
            }
            let mut attn_out = head_outs.remove(0);
            for h in head_outs {
                attn_out = add(&attn_out, &h);
            }
            x = add(&x, &attn_out);

            // --- MLP ---
            let xl = layernorm(&x, &block.ln2_w, &block.ln2_b);
            let hidden = relu(&add_bias(&matmul(&xl, &block.ffn_w1), &block.ffn_b1));
            let ffn_out = add_bias(&matmul(&hidden, &block.ffn_w2), &block.ffn_b2);
            x = add(&x, &ffn_out);
        }

        // Final norm + tied output projection.
        let xl = layernorm(&x, &self.lnf_w, &self.lnf_b);
        matmul_bt(&xl, &self.token_emb)
    }

    /// Train one step on a batch; returns the loss value.
    pub fn train_step(
        &self,
        opt: &mut crate::autograd::Adam,
        inputs: &[usize],
        targets: &[usize],
    ) -> f32 {
        let logits = self.forward(inputs);
        let loss = crate::autograd::cross_entropy(&logits, targets);
        backward(&loss);
        let value = loss.borrow().data[0];
        opt.step(&self.params);
        value
    }

    /// Greedy sample: continue `idx` for `steps` tokens.
    pub fn sample(&self, idx: &[usize], steps: usize) -> Vec<usize> {
        let mut out = idx.to_vec();
        for _ in 0..steps {
            let ctx: Vec<usize> = if out.len() >= self.cfg.block {
                out[out.len() - self.cfg.block..].to_vec()
            } else {
                out.clone()
            };
            let logits = self.forward(&ctx);
            let data = logits.borrow().data.clone();
            let n = self.cfg.vocab;
            let last = &data[(ctx.len() - 1) * n..ctx.len() * n];
            let next = argmax(last);
            out.push(next);
        }
        out
    }

    /// Average cross-entropy over every window of `data` (no training).
    /// Used to evaluate on held-out (unknown) text.
    pub fn eval_loss(&self, data: &[usize], block: usize) -> f32 {
        assert!(data.len() > block, "not enough data for evaluation");
        let mut total = 0.0f32;
        let mut count = 0usize;
        for start in 0..=data.len() - block - 1 {
            let inputs = &data[start..start + block];
            let targets = &data[start + 1..start + 1 + block];
            let logits = self.forward(inputs);
            let loss = crate::autograd::cross_entropy(&logits, targets);
            total += loss.borrow().data[0];
            count += 1;
        }
        total / count as f32
    }
}

fn causal_mask(t: usize) -> Tensor {
    let mut values = vec![0.0f32; t * t];
    for i in 0..t {
        for j in 0..t {
            if j > i {
                values[i * t + j] = -1e9;
            }
        }
    }
    crate::autograd::const_(&[t, t], values)
}

fn argmax(xs: &[f32]) -> usize {
    xs.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::Adam;

    fn tiny_cfg() -> Config {
        Config {
            vocab: 10,
            d: 16,
            heads: 2,
            layers: 1,
            block: 8,
            ffn: 32,
        }
    }

    fn tiny_bits(vocab: usize) -> Vec<u32> {
        (0..vocab).map(|i| (i * 137 % TOTAL_CELLS) as u32).collect()
    }

    #[test]
    fn forward_shapes_and_loss_finite() {
        let model = Transformer::new(tiny_cfg(), 1);
        let inputs = [1usize, 2, 3, 4, 5, 6, 7, 8];
        let targets = [2usize, 3, 4, 5, 6, 7, 8, 9];
        let logits = model.forward(&inputs);
        assert_eq!(logits.borrow().shape, vec![8, 10]);
        let loss = crate::autograd::cross_entropy(&logits, &targets);
        assert!(loss.borrow().data[0].is_finite());
    }

    #[test]
    fn address_mode_forward_shapes_and_loss_finite() {
        let cfg = tiny_cfg();
        let model = Transformer::new_address(cfg, 1, &tiny_bits(10));
        assert!(model.address_mode());
        let inputs = [1usize, 2, 3, 4, 5, 6, 7, 8];
        let targets = [2usize, 3, 4, 5, 6, 7, 8, 9];
        let logits = model.forward(&inputs);
        assert_eq!(logits.borrow().shape, vec![8, 10]);
        let loss = crate::autograd::cross_entropy(&logits, &targets);
        assert!(loss.borrow().data[0].is_finite());
    }

    #[test]
    fn training_reduces_loss_on_repetitive_sequence() {
        // A tiny deterministic language: "ab " repeated.
        let text: String = (0..80)
            .map(|i| if i % 3 == 0 { 'a' } else if i % 3 == 1 { 'b' } else { ' ' })
            .collect();
        let cfg = Config {
            vocab: 3,
            d: 16,
            heads: 2,
            layers: 1,
            block: 8,
            ffn: 32,
        };
        let model = Transformer::new(cfg, 7);
        let mut opt = Adam::new(0.02);
        let seq: Vec<usize> = text
            .chars()
            .map(|c| match c {
                'a' => 0,
                'b' => 1,
                _ => 2,
            })
            .collect();

        let mut losses = Vec::new();
        for step in 0..300 {
            let start = step % (seq.len() - 8 - 1);
            let inputs = seq[start..start + 8].to_vec();
            let targets = seq[start + 1..start + 9].to_vec();
            let loss = model.train_step(&mut opt, &inputs, &targets);
            if step % 50 == 0 {
                losses.push(loss);
            }
        }
        let first = losses[0];
        let last = *losses.last().unwrap();
        assert!(last < first * 0.5, "loss did not decrease: {first} -> {last}");
    }

    #[test]
    fn address_mode_training_reduces_loss() {
        let text: String = (0..80)
            .map(|i| if i % 3 == 0 { 'a' } else if i % 3 == 1 { 'b' } else { ' ' })
            .collect();
        let cfg = Config {
            vocab: 3,
            d: 16,
            heads: 2,
            layers: 1,
            block: 8,
            ffn: 32,
        };
        let bits = tiny_bits(3);
        let model = Transformer::new_address(cfg, 7, &bits);
        let mut opt = Adam::new(0.02);
        let seq: Vec<usize> = text
            .chars()
            .map(|c| match c {
                'a' => 0,
                'b' => 1,
                _ => 2,
            })
            .collect();

        let mut losses = Vec::new();
        for step in 0..300 {
            let start = step % (seq.len() - 8 - 1);
            let inputs = seq[start..start + 8].to_vec();
            let targets = seq[start + 1..start + 9].to_vec();
            let loss = model.train_step(&mut opt, &inputs, &targets);
            if step % 50 == 0 {
                losses.push(loss);
            }
        }
        let first = losses[0];
        let last = *losses.last().unwrap();
        assert!(last < first * 0.5, "address-mode loss did not decrease: {first} -> {last}");
    }

    #[test]
    fn eval_loss_is_finite_and_positive() {
        let cfg = tiny_cfg();
        let model = Transformer::new(cfg, 1);
        let data: Vec<usize> = (0..40).map(|i| i % 10).collect();
        let loss = model.eval_loss(&data, 8);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }
}
