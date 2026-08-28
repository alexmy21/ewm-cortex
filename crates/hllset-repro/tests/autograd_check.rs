//! Gradient checks for the hand-rolled autograd engine.
//!
//! Each check compares reverse-mode grads against central finite
//! differences. This is the safety net that makes hand-rolled backprop
//! trustworthy before it is used to train the Phase 0 transformer.

use hllset_repro::autograd::{
    add_bias, backward, cross_entropy, gather, layernorm, leaf, matmul, matmul_bt, relu, softmax,
    sum, Tensor,
};

/// Central finite-difference gradient of `leaf` w.r.t. `loss_fn()`.
fn numeric_grad(leaf: &Tensor, loss_fn: impl Fn() -> Tensor, eps: f32) -> Vec<f32> {
    let n = leaf.borrow().data.len();
    let mut grads = vec![0.0f32; n];
    for i in 0..n {
        let orig = leaf.borrow().data[i];
        leaf.borrow_mut().data[i] = orig + eps;
        let lp = loss_fn().borrow().data[0];
        leaf.borrow_mut().data[i] = orig - eps;
        let lm = loss_fn().borrow().data[0];
        leaf.borrow_mut().data[i] = orig;
        grads[i] = (lp - lm) / (2.0 * eps);
    }
    grads
}

fn assert_close(analytic: &[f32], numeric: &[f32], tol: f32) {
    assert_eq!(analytic.len(), numeric.len());
    for (i, (a, n)) in analytic.iter().zip(numeric).enumerate() {
        let diff = (a - n).abs();
        let scale = 1.0f32.max(a.abs()).max(n.abs());
        assert!(
            diff <= tol * scale,
            "grad[{i}] analytic={a} numeric={n} diff={diff}"
        );
    }
}

/// Run a gradient check on `leaf` using `loss_fn` (which must rebuild the
/// graph from shared `Rc` clones of the leaves).
fn grad_check(leaf: &Tensor, loss_fn: impl Fn() -> Tensor, tol: f32) {
    let loss = loss_fn();
    backward(&loss);
    let analytic = leaf.borrow().grad.clone();
    let numeric = numeric_grad(leaf, loss_fn, 1e-3);
    assert_close(&analytic, &numeric, tol);
}

#[test]
fn matmul_grad() {
    let a = leaf(&[2, 3], |i| (i as f32 + 1.0) * 0.7);
    let b = leaf(&[3, 2], |i| (i as f32 - 2.0) * 0.5);
    {
        let (a2, b2) = (a.clone(), b.clone());
        grad_check(&a, move || sum(&matmul(&a2, &b2)), 1e-3);
    }
    {
        let (a2, b2) = (a.clone(), b.clone());
        grad_check(&b, move || sum(&matmul(&a2, &b2)), 1e-3);
    }
}

#[test]
fn matmul_bt_grad() {
    let a = leaf(&[2, 3], |i| (i as f32 + 1.0) * 0.6);
    let b = leaf(&[2, 3], |i| (i as f32 - 1.0) * 0.4);
    {
        let (a2, b2) = (a.clone(), b.clone());
        grad_check(&a, move || sum(&matmul_bt(&a2, &b2)), 1e-3);
    }
    {
        let (a2, b2) = (a.clone(), b.clone());
        grad_check(&b, move || sum(&matmul_bt(&a2, &b2)), 1e-3);
    }
}

#[test]
fn relu_and_add_bias_grad() {
    let a = leaf(&[2, 3], |i| (i as f32 - 2.0) * 0.8);
    let b = leaf(&[3], |i| (i as f32 + 1.0) * 0.3);
    {
        let (a2, b2) = (a.clone(), b.clone());
        grad_check(&a, move || sum(&relu(&add_bias(&a2, &b2))), 1e-3);
    }
    {
        let (a2, b2) = (a.clone(), b.clone());
        grad_check(&b, move || sum(&relu(&add_bias(&a2, &b2))), 1e-3);
    }
}

#[test]
fn softmax_cross_entropy_grad() {
    let logits = leaf(&[2, 4], |i| (i as f32 - 3.0) * 0.9);
    let targets = vec![1usize, 3];
    {
        let l = logits.clone();
        let t = targets.clone();
        grad_check(&logits, move || cross_entropy(&l, &t), 1e-3);
    }
}

#[test]
fn layernorm_grad() {
    let x = leaf(&[2, 4], |i| (i as f32 - 1.0) * 0.7);
    let w = leaf(&[4], |i| 1.0 + 0.2 * i as f32);
    let b = leaf(&[4], |i| -0.1 * i as f32);

    {
        let (x2, w2, b2) = (x.clone(), w.clone(), b.clone());
        grad_check(&x, move || sum(&layernorm(&x2, &w2, &b2)), 1e-3);
    }
    {
        let (x2, w2, b2) = (x.clone(), w.clone(), b.clone());
        grad_check(&w, move || sum(&layernorm(&x2, &w2, &b2)), 1e-3);
    }
    {
        let (x2, w2, b2) = (x.clone(), w.clone(), b.clone());
        grad_check(&b, move || sum(&layernorm(&x2, &w2, &b2)), 1e-3);
    }
}

#[test]
fn softmax_grad() {
    let a = leaf(&[1, 5], |i| (i as f32 - 2.0) * 0.6);
    {
        let a2 = a.clone();
        grad_check(&a, move || sum(&softmax(&a2)), 1e-3);
    }
}

#[test]
fn gather_grad() {
    let table = leaf(&[4, 3], |i| (i as f32 - 1.0) * 0.5);
    let indices = [0usize, 2, 0]; // repeated row exercises scatter-add
    {
        let t = table.clone();
        grad_check(&table, move || sum(&gather(&t, &indices)), 1e-3);
    }
}
