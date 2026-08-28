//! Mode C: hybrid attention scores.
//!
//! The learned dot-product attention remains the workhorse; the HLLSet term
//! injects a content-addressed bias:
//!
//! ```text
//! score(i, j) = (q_i · k_j) / √d  +  λ · φ(Q_hll, K_hll)
//! ```

/// Configuration for the hybrid mode.
#[derive(Clone, Copy, Debug)]
pub struct HybridConfig {
    /// Weight of the lattice score relative to the dense dot product.
    pub lambda: f32,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self { lambda: 1.0 }
    }
}

/// Combine one dense dot-product score with one lattice score.
pub fn hybrid_score(dense: f32, lattice: f64, lambda: f32) -> f32 {
    dense + lambda * lattice as f32
}

/// Combine parallel dense and lattice logit vectors elementwise.
///
/// Panics if the slices have different lengths.
pub fn hybrid_logits(dense: &[f32], lattice: &[f64], lambda: f32) -> Vec<f32> {
    assert_eq!(dense.len(), lattice.len(), "logit vectors must align");
    dense
        .iter()
        .zip(lattice)
        .map(|(&d, &l)| hybrid_score(d, l, lambda))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_score_scales_lattice_term() {
        assert_eq!(hybrid_score(1.0, 0.5, 2.0), 2.0);
        assert_eq!(hybrid_score(1.0, 0.5, 0.0), 1.0);
    }

    #[test]
    fn hybrid_logits_combines_elementwise() {
        let dense = [1.0, 2.0, 3.0];
        let lattice = [0.5, 1.0, -1.0];
        let out = hybrid_logits(&dense, &lattice, 2.0);
        assert_eq!(out, vec![2.0, 4.0, 1.0]);
    }

    #[test]
    #[should_panic(expected = "logit vectors must align")]
    fn hybrid_logits_mismatched_lengths_panic() {
        let _ = hybrid_logits(&[1.0], &[1.0, 2.0], 1.0);
    }
}
