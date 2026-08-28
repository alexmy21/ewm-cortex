//! Minimal tensor autograd — micrograd-style, at tensor granularity.
//!
//! A tiny tape-based engine sized for the Phase 0 reproduction transformer
//! (d ≤ 256, block ≤ 256). Tensors are 2-D (plus scalars as `[1]`), ops are
//! recorded on a parent list, and [`backward`] runs reverse-mode
//! accumulation. Gradients are computed with the same naive GEMM kernels as
//! the forward pass — slow but transparent, which is the point.

use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::rc::Rc;

/// Recorded operation of a tensor node.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    Leaf,
    Const,
    MatMul,
    MatMulBt,
    Add,
    AddBias,
    ReLU,
    Scale(f32),
    Softmax,
    LayerNorm,
    /// Reduce sum over all elements → scalar-shaped `[1]`.
    Sum,
    /// Row gather: `out[i] = table[indices[i]]`. Carries the indices.
    Gather(Vec<usize>),
    /// Cross-entropy over the last dim; carries the target indices.
    CrossEntropy(Vec<usize>),
}

/// A differentiable tensor node.
#[derive(Clone, Debug)]
pub struct Tensor_ {
    pub data: Vec<f32>,
    pub grad: Vec<f32>,
    pub shape: Vec<usize>,
    pub requires_grad: bool,
    op: Op,
    parents: Vec<Tensor>,
    /// Forward values needed by the backward pass (copies, small scale).
    saved: Vec<Vec<f32>>,
}

pub type Tensor = Rc<RefCell<Tensor_>>;

// ── Constructors ───────────────────────────────────────────────────────────

/// A trainable leaf parameter.
pub fn leaf(shape: &[usize], mut init: impl FnMut(usize) -> f32) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(&mut init).collect();
    Rc::new(RefCell::new(Tensor_ {
        grad: vec![0.0; n],
        data,
        shape: shape.to_vec(),
        requires_grad: true,
        op: Op::Leaf,
        parents: Vec::new(),
        saved: Vec::new(),
    }))
}

/// A constant tensor (no gradient flows through it).
pub fn const_(shape: &[usize], values: Vec<f32>) -> Tensor {
    let n: usize = shape.iter().product();
    assert_eq!(n, values.len(), "const_ shape/data mismatch");
    Rc::new(RefCell::new(Tensor_ {
        grad: vec![0.0; n],
        data: values,
        shape: shape.to_vec(),
        requires_grad: false,
        op: Op::Const,
        parents: Vec::new(),
        saved: Vec::new(),
    }))
}

/// A zero constant tensor.
pub fn zeros(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    const_(shape, vec![0.0; n])
}

/// A one-hot constant tensor of shape `[rows, cols]`.
pub fn one_hot(rows: usize, cols: usize, indices: &[usize]) -> Tensor {
    assert_eq!(rows, indices.len(), "one_hot row count mismatch");
    let mut values = vec![0.0f32; rows * cols];
    for (i, &idx) in indices.iter().enumerate() {
        assert!(idx < cols, "one_hot index out of range");
        values[i * cols + idx] = 1.0;
    }
    const_(&[rows, cols], values)
}

// ── Ops ────────────────────────────────────────────────────────────────────

/// `A @ B`, `A: [m, n]`, `B: [n, k]` → `[m, k]`.
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let (m, n) = dims2(a);
    let (n2, k) = dims2(b);
    assert_eq!(n, n2, "matmul inner dim mismatch");
    let a_data = a.borrow().data.clone();
    let b_data = b.borrow().data.clone();
    let data = gemm(&a_data, &b_data, m, n, k);
    make(
        vec![m, k],
        data,
        Op::MatMul,
        vec![a.clone(), b.clone()],
        vec![a_data, b_data],
    )
}

/// `A @ Bᵀ`, `A: [m, n]`, `B: [k, n]` → `[m, k]`.
pub fn matmul_bt(a: &Tensor, b: &Tensor) -> Tensor {
    let (m, n) = dims2(a);
    let (k, n2) = dims2(b);
    assert_eq!(n, n2, "matmul_bt inner dim mismatch");
    let a_data = a.borrow().data.clone();
    let b_data = b.borrow().data.clone();
    let data = gemm_bt(&a_data, &b_data, m, n, k);
    make(
        vec![m, k],
        data,
        Op::MatMulBt,
        vec![a.clone(), b.clone()],
        vec![a_data, b_data],
    )
}

/// Elementwise add, same shape.
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let sa = a.borrow().shape.clone();
    let sb = b.borrow().shape.clone();
    assert_eq!(sa, sb, "add shape mismatch");
    let a_data = a.borrow().data.clone();
    let b_data = b.borrow().data.clone();
    let data: Vec<f32> = a_data.iter().zip(&b_data).map(|(x, y)| x + y).collect();
    make(sa, data, Op::Add, vec![a.clone(), b.clone()], Vec::new())
}

/// Row-wise bias add: `A: [m, n] + b: [n]` → `[m, n]`.
pub fn add_bias(a: &Tensor, b: &Tensor) -> Tensor {
    let (m, n) = dims2(a);
    assert_eq!(b.borrow().shape.as_slice(), &[n], "add_bias shape mismatch");
    let a_data = a.borrow().data.clone();
    let b_data = b.borrow().data.clone();
    let mut data = a_data.clone();
    for i in 0..m {
        for j in 0..n {
            data[i * n + j] += b_data[j];
        }
    }
    make(
        vec![m, n],
        data,
        Op::AddBias,
        vec![a.clone(), b.clone()],
        Vec::new(),
    )
}

/// Elementwise ReLU.
pub fn relu(a: &Tensor) -> Tensor {
    let shape = a.borrow().shape.clone();
    let a_data = a.borrow().data.clone();
    let data: Vec<f32> = a_data.iter().map(|&x| x.max(0.0)).collect();
    make(shape, data, Op::ReLU, vec![a.clone()], vec![a_data])
}

/// Multiply by a scalar.
pub fn scale(a: &Tensor, s: f32) -> Tensor {
    let shape = a.borrow().shape.clone();
    let a_data = a.borrow().data.clone();
    let data: Vec<f32> = a_data.iter().map(|&x| x * s).collect();
    make(shape, data, Op::Scale(s), vec![a.clone()], Vec::new())
}

/// Softmax over the last dim.
pub fn softmax(a: &Tensor) -> Tensor {
    let (m, n) = dims2(a);
    let a_data = a.borrow().data.clone();
    let probs = softmax_rows(&a_data, m, n);
    make(
        vec![m, n],
        probs.clone(),
        Op::Softmax,
        vec![a.clone()],
        vec![probs],
    )
}

/// LayerNorm over the last dim with learnable scale `w` and bias `b`.
pub fn layernorm(a: &Tensor, w: &Tensor, b: &Tensor) -> Tensor {
    let (m, n) = dims2(a);
    assert_eq!(w.borrow().shape.as_slice(), &[n], "layernorm w shape");
    assert_eq!(b.borrow().shape.as_slice(), &[n], "layernorm b shape");
    let a_data = a.borrow().data.clone();
    let w_data = w.borrow().data.clone();
    let b_data = b.borrow().data.clone();
    let (y, mean, var, inv_std, xhat) = layernorm_rows(&a_data, &w_data, &b_data, m, n);
    make(
        vec![m, n],
        y,
        Op::LayerNorm,
        vec![a.clone(), w.clone(), b.clone()],
        vec![a_data, w_data, mean, var, inv_std, xhat],
    )
}

/// Reduce sum over all elements. Returns a scalar-shaped `[1]` tensor.
pub fn sum(a: &Tensor) -> Tensor {
    let total: f32 = a.borrow().data.iter().sum();
    make(vec![1], vec![total], Op::Sum, vec![a.clone()], Vec::new())
}

/// Row gather: `out[i] = table[indices[i]]`. `table: [N, D]`, `indices` has
/// length `T` → output `[T, D]`. The sparse equivalent of one-hot matmul —
/// the backward scatters row gradients into the table.
pub fn gather(table: &Tensor, indices: &[usize]) -> Tensor {
    let (n, d) = dims2(table);
    assert!(
        indices.iter().all(|&i| i < n),
        "gather index out of range"
    );
    let mut data = Vec::with_capacity(indices.len() * d);
    for &i in indices {
        data.extend_from_slice(&table.borrow().data[i * d..(i + 1) * d]);
    }
    make(
        vec![indices.len(), d],
        data,
        Op::Gather(indices.to_vec()),
        vec![table.clone()],
        Vec::new(),
    )
}

/// Mean cross-entropy over the last dim. Returns a scalar-shaped `[1]` tensor.
pub fn cross_entropy(logits: &Tensor, targets: &[usize]) -> Tensor {
    let (m, n) = dims2(logits);
    assert_eq!(m, targets.len(), "cross_entropy target count mismatch");
    let logits_data = logits.borrow().data.clone();
    let probs = softmax_rows(&logits_data, m, n);
    let mut loss = 0.0f32;
    for (i, &t) in targets.iter().enumerate() {
        assert!(t < n, "cross_entropy target out of range");
        loss -= probs[i * n + t].ln();
    }
    loss /= m as f32;
    make(
        vec![1],
        vec![loss],
        Op::CrossEntropy(targets.to_vec()),
        vec![logits.clone()],
        vec![probs],
    )
}

// ── Backward ───────────────────────────────────────────────────────────────

/// Reverse-mode autodiff: fills `.grad` on every reachable leaf.
pub fn backward(root: &Tensor) {
    // Topological order.
    let mut topo: Vec<Tensor> = Vec::new();
    let mut visited: HashSet<usize> = HashSet::new();
    fn build(t: &Tensor, topo: &mut Vec<Tensor>, visited: &mut HashSet<usize>) {
        let key = Rc::as_ptr(t) as usize;
        if visited.insert(key) {
            for p in &t.borrow().parents {
                build(p, topo, visited);
            }
            topo.push(t.clone());
        }
    }
    build(root, &mut topo, &mut visited);

    for t in &topo {
        t.borrow_mut().grad.fill(0.0);
    }
    root.borrow_mut().grad[0] = 1.0;

    for t in topo.iter().rev() {
        let (op, parents, saved, grad) = {
            let node = t.borrow();
            (
                node.op.clone(),
                node.parents.clone(),
                node.saved.clone(),
                node.grad.clone(),
            )
        };
        let dparents: Vec<Vec<f32>> = match &op {
            Op::Leaf | Op::Const => Vec::new(),
            Op::MatMul => {
                let (m, n) = dims2(&parents[0]);
                let k = parents[1].borrow().shape[1];
                let (a, b) = (&saved[0], &saved[1]);
                vec![
                    gemm_bt(&grad, b, m, k, n), // dA = dC @ Bᵀ
                    gemm_at_b(a, &grad, m, n, k), // dB = Aᵀ @ dC
                ]
            }
            Op::MatMulBt => {
                let (m, n) = dims2(&parents[0]);
                let k = parents[1].borrow().shape[0];
                let (a, b) = (&saved[0], &saved[1]);
                vec![
                    gemm(&grad, b, m, k, n),       // dA = dC @ B
                    gemm_at_b(&grad, a, m, k, n),  // dB = dCᵀ @ A
                ]
            }
            Op::Add => vec![grad.clone(), grad.clone()],
            Op::AddBias => {
                let (m, n) = dims2(&parents[0]);
                let mut db = vec![0.0f32; n];
                for i in 0..m {
                    for j in 0..n {
                        db[j] += grad[i * n + j];
                    }
                }
                vec![grad.clone(), db]
            }
            Op::ReLU => {
                let a = &saved[0];
                let d = a
                    .iter()
                    .zip(&grad)
                    .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 })
                    .collect();
                vec![d]
            }
            Op::Scale(s) => {
                let d = grad.iter().map(|&g| g * s).collect();
                vec![d]
            }
            Op::Softmax => {
                let (m, n) = dims2(&parents[0]);
                let probs = &saved[0];
                vec![softmax_backward(probs, &grad, m, n)]
            }
            Op::LayerNorm => {
                let (m, n) = dims2(&parents[0]);
                let x = &saved[0];
                let w = &saved[1];
                let mean = &saved[2];
                let var = &saved[3];
                let inv_std = &saved[4];
                let xhat = &saved[5];
                layernorm_backward(&grad, x, w, mean, var, inv_std, xhat, m, n)
            }
            Op::Sum => {
                let n = parents[0].borrow().data.len();
                vec![vec![grad[0]; n]]
            }
            Op::Gather(indices) => {
                let (n, d) = dims2(&parents[0]);
                let mut dx = vec![0.0f32; n * d];
                for (i, &row) in indices.iter().enumerate() {
                    for dim in 0..d {
                        dx[row * d + dim] += grad[i * d + dim];
                    }
                }
                vec![dx]
            }
            Op::CrossEntropy(targets) => {
                let (m, n) = dims2(&parents[0]);
                let probs = &saved[0];
                let mut d = probs.clone();
                for (i, &t) in targets.iter().enumerate() {
                    d[i * n + t] -= 1.0;
                }
                for v in d.iter_mut() {
                    *v /= m as f32;
                }
                vec![d]
            }
        };

        for (p, dp) in parents.iter().zip(&dparents) {
            if p.borrow().requires_grad {
                let mut parent = p.borrow_mut();
                for (g, &d) in parent.grad.iter_mut().zip(dp) {
                    *g += d;
                }
            }
        }
    }
}

// ── Adam ───────────────────────────────────────────────────────────────────

/// Adam optimizer over a fixed parameter list.
pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: usize,
    m: HashMap<usize, Vec<f32>>,
    v: HashMap<usize, Vec<f32>>,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    /// Update every parameter in place and zero its gradient.
    pub fn step(&mut self, params: &[Tensor]) {
        self.t += 1;
        let (b1, b2, lr, eps, t) = (self.beta1, self.beta2, self.lr, self.eps, self.t);
        for p in params {
            let key = Rc::as_ptr(p) as usize;
            let mut node = p.borrow_mut();
            let n = node.data.len();
            let m = self.m.entry(key).or_insert_with(|| vec![0.0f32; n]);
            let v = self.v.entry(key).or_insert_with(|| vec![0.0f32; n]);
            for i in 0..n {
                let g = node.grad[i];
                m[i] = b1 * m[i] + (1.0 - b1) * g;
                v[i] = b2 * v[i] + (1.0 - b2) * g * g;
                let mhat = m[i] / (1.0 - b1.powi(t as i32));
                let vhat = v[i] / (1.0 - b2.powi(t as i32));
                node.data[i] -= lr * mhat / (vhat.sqrt() + eps);
            }
            node.grad.fill(0.0);
        }
    }
}

// ── Internal helpers ───────────────────────────────────────────────────────

fn dims2(t: &Tensor) -> (usize, usize) {
    let s = t.borrow().shape.clone();
    assert_eq!(s.len(), 2, "expected 2-D tensor, got {s:?}");
    (s[0], s[1])
}

fn make(
    shape: Vec<usize>,
    data: Vec<f32>,
    op: Op,
    parents: Vec<Tensor>,
    saved: Vec<Vec<f32>>,
) -> Tensor {
    let n = data.len();
    Rc::new(RefCell::new(Tensor_ {
        data,
        grad: vec![0.0; n],
        shape,
        requires_grad: parents.iter().any(|p| p.borrow().requires_grad),
        op,
        parents,
        saved,
    }))
}

/// `A @ B`: `A: [m, n]`, `B: [n, k]` → `[m, k]`.
fn gemm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * k];
    for i in 0..m {
        for j in 0..k {
            let mut s = 0.0;
            for p in 0..n {
                s += a[i * n + p] * b[p * k + j];
            }
            out[i * k + j] = s;
        }
    }
    out
}

/// `A @ Bᵀ`: `A: [m, n]`, `B: [k, n]` → `[m, k]`.
fn gemm_bt(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * k];
    for i in 0..m {
        for j in 0..k {
            let mut s = 0.0;
            for p in 0..n {
                s += a[i * n + p] * b[j * n + p];
            }
            out[i * k + j] = s;
        }
    }
    out
}

/// `Aᵀ @ B`: `A: [m, n]`, `B: [m, k]` → `[n, k]`.
fn gemm_at_b(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * k];
    for i in 0..n {
        for j in 0..k {
            let mut s = 0.0;
            for p in 0..m {
                s += a[p * n + i] * b[p * k + j];
            }
            out[i * k + j] = s;
        }
    }
    out
}

fn softmax_rows(x: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        let row = &x[i * n..(i + 1) * n];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for j in 0..n {
            let e = (row[j] - max).exp();
            out[i * n + j] = e;
            sum += e;
        }
        for j in 0..n {
            out[i * n + j] /= sum;
        }
    }
    out
}

fn softmax_backward(probs: &[f32], grad: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        let row = &probs[i * n..(i + 1) * n];
        let g = &grad[i * n..(i + 1) * n];
        let dot: f32 = row.iter().zip(g).map(|(p, g)| p * g).sum();
        for j in 0..n {
            out[i * n + j] = row[j] * (g[j] - dot);
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn layernorm_rows(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let eps = 1e-5f32;
    let mut y = vec![0.0f32; m * n];
    let mut mean = vec![0.0f32; m];
    let mut var = vec![0.0f32; m];
    let mut inv_std = vec![0.0f32; m];
    let mut xhat = vec![0.0f32; m * n];
    for i in 0..m {
        let row = &x[i * n..(i + 1) * n];
        let mu: f32 = row.iter().sum::<f32>() / n as f32;
        let v: f32 = row.iter().map(|&x| (x - mu) * (x - mu)).sum::<f32>() / n as f32;
        let is = 1.0 / (v + eps).sqrt();
        mean[i] = mu;
        var[i] = v;
        inv_std[i] = is;
        for j in 0..n {
            let xh = (row[j] - mu) * is;
            xhat[i * n + j] = xh;
            y[i * n + j] = xh * w[j] + b[j];
        }
    }
    (y, mean, var, inv_std, xhat)
}

#[allow(clippy::too_many_arguments)]
fn layernorm_backward(
    grad: &[f32],
    x: &[f32],
    w: &[f32],
    mean: &[f32],
    var: &[f32],
    inv_std: &[f32],
    xhat: &[f32],
    m: usize,
    n: usize,
) -> Vec<Vec<f32>> {
    let eps = 1e-5f32;
    let mut dx = vec![0.0f32; m * n];
    let mut dw = vec![0.0f32; n];
    let mut db = vec![0.0f32; n];
    for i in 0..m {
        let g = &grad[i * n..(i + 1) * n];
        let xh = &xhat[i * n..(i + 1) * n];
        let xr = &x[i * n..(i + 1) * n];
        let is = inv_std[i];
        let mu = mean[i];
        let v = var[i];

        // db, dw
        for j in 0..n {
            db[j] += g[j];
            dw[j] += g[j] * xh[j];
        }
        // dxhat = g * w
        let dxhat: Vec<f32> = g.iter().zip(w).map(|(&g, &w)| g * w).collect();
        // dvar = sum(dxhat * (x-mu)) * (-0.5) * (v+eps)^(-3/2)
        let mut dvar = 0.0f32;
        for j in 0..n {
            dvar += dxhat[j] * (xr[j] - mu);
        }
        dvar *= -0.5 * (v + eps).powf(-1.5);
        // dmu = -inv_std * sum(dxhat) + dvar * mean(-2(x-mu))  [second term is 0]
        let mut dmu = 0.0f32;
        for j in 0..n {
            dmu -= is * dxhat[j];
        }
        for j in 0..n {
            dx[i * n + j] = dxhat[j] * is + dvar * 2.0 * (xr[j] - mu) / n as f32 + dmu / n as f32;
        }
    }
    vec![dx, dw, db]
}

// ── Small utilities for the harness ────────────────────────────────────────

/// A deterministic xorshift PRNG (no external deps).
pub struct XorShift(u32);

impl XorShift {
    pub fn new(seed: u32) -> Self {
        Self(seed.max(1))
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        (x as f32) / (u32::MAX as f32 + 1.0)
    }

    /// Uniform in `[-scale, scale]`.
    pub fn uniform(&mut self, scale: f32) -> f32 {
        (self.next_f32() * 2.0 - 1.0) * scale
    }

    /// Index in `[0, n)`.
    pub fn index(&mut self, n: usize) -> usize {
        (self.next_f32() * n as f32) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaf_and_const_shapes() {
        let a = leaf(&[2, 3], |_| 1.0);
        assert_eq!(a.borrow().shape, vec![2, 3]);
        assert_eq!(a.borrow().data.len(), 6);
        assert!(a.borrow().requires_grad);

        let c = const_(&[2], vec![3.0, 4.0]);
        assert!(!c.borrow().requires_grad);
    }

    #[test]
    fn gemm_matches_naive() {
        // A [2,3] @ B [3,2]
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let out = gemm(&a, &b, 2, 3, 2);
        assert_eq!(out, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn matmul_backward_shapes() {
        let a = leaf(&[2, 3], |i| i as f32);
        let b = leaf(&[3, 2], |i| (i as f32) - 2.0);
        let c = matmul(&a, &b);
        let loss = scale(&c, 1.0); // sum would need a reduce; use backward on c directly
        backward(&loss);
        assert_eq!(a.borrow().grad.len(), 6);
        assert_eq!(b.borrow().grad.len(), 6);
    }

    #[test]
    fn softmax_rows_sums_to_one() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let p = softmax_rows(&x, 1, 4);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
