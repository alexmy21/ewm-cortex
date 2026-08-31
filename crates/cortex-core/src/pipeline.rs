//! `CortexPipeline` — the black-box interface (enhanced hllset-cortex).
//!
//! ```text
//! encoding IDs → HLLSet → ∩ gate_TF → TF-LUT → materialize → restored IDs
//! ```
//!
//! The two-space discipline of the reference is preserved: the cortex only
//! sees opaque `tid{n}` encoding IDs; real tokens never cross the boundary.
//! The enhancement (MoE/ETT context layer) lives in `hllset-attn` and will
//! be wired on top of this pipeline in the next milestone.

use hllset_core::HLLSet;

use crate::gate::Gate;
use crate::lut::TfLut;

/// Result of one pipeline pass.
#[derive(Clone, Debug)]
pub struct PipelineResult {
    /// The ingested encoding IDs (in input order).
    pub input_ids: Vec<Vec<u8>>,
    /// Bits set by the raw document HLLSet.
    pub doc_bits: u64,
    /// Bits surviving the gate intersection.
    pub gated_bits: u64,
    /// Materialized IDs, TF-ranked (ties: byte order).
    pub restored_ids: Vec<Vec<u8>>,
    /// Restored IDs that are NOT in the decoder vocabulary (sketch-gate
    /// leaks; the exact-LUT check reports them).
    pub leaks: Vec<Vec<u8>>,
    /// LUT coverage over the gated HLLSet (1.0 = full resolution).
    pub coverage: f64,
}

impl PipelineResult {
    /// Whether the pass is leak-free.
    pub fn ok(&self) -> bool {
        self.leaks.is_empty()
    }
}

/// The black-box cortex pipeline.
#[derive(Clone, Debug, Default)]
pub struct CortexPipeline {
    gate: Option<Gate>,
    lut: TfLut,
    processed: u64,
}

impl CortexPipeline {
    /// Create an empty pipeline (no gate until [`Self::set_gate`]).
    pub fn new() -> Self {
        Self::default()
    }

    /// Build the `gate_TF` from the decoder vocabulary.
    pub fn set_gate<I, B>(&mut self, vocab: I)
    where
        I: IntoIterator<Item = B>,
        B: AsRef<[u8]>,
    {
        self.gate = Some(Gate::from_vocab(vocab));
    }

    /// Whether a gate is configured.
    pub fn has_gate(&self) -> bool {
        self.gate.is_some()
    }

    /// Number of processed passes.
    pub fn processed(&self) -> u64 {
        self.processed
    }

    /// Number of interned encoding IDs in the TF-LUT.
    pub fn lut_len(&self) -> usize {
        self.lut.len()
    }

    /// Process one document of encoding IDs.
    ///
    /// TF accumulates **pre-gate** (monotonic); materialization resolves the
    /// gated HLLSet. Unknown IDs are registered by `observe` and may survive
    /// the sketch gate only by hash collision — reported as `leaks`.
    pub fn process(&mut self, ids: &[Vec<u8>]) -> PipelineResult {
        let doc = HLLSet::from_tokens(ids.iter());
        let gated = match &self.gate {
            Some(gate) => gate.apply(&doc),
            None => doc.clone(),
        };

        self.lut.observe(ids);

        let restored_ids = self.lut.materialize(&gated);
        let leaks: Vec<Vec<u8>> = match &self.gate {
            Some(gate) => restored_ids
                .iter()
                .filter(|id| !gate.exact_known(id))
                .cloned()
                .collect(),
            None => Vec::new(),
        };
        let coverage = self.lut.confidence(&gated);

        self.processed += 1;

        PipelineResult {
            input_ids: ids.to_vec(),
            doc_bits: doc.popcount(),
            gated_bits: gated.popcount(),
            restored_ids,
            leaks,
            coverage,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_ids_roundtrip_through_gate() {
        let mut pipeline = CortexPipeline::new();
        pipeline.set_gate(["tid0", "tid1", "tid2"]);
        let result = pipeline.process(&[b"tid0".to_vec(), b"tid1".to_vec()]);
        assert!(result.ok(), "leaks: {:?}", result.leaks);
        assert_eq!(result.restored_ids, vec![b"tid0".to_vec(), b"tid1".to_vec()]);
        assert_eq!(result.coverage, 1.0);
    }

    #[test]
    fn unknown_ids_are_reported_not_silently_dropped() {
        let mut pipeline = CortexPipeline::new();
        pipeline.set_gate(["tid0"]);
        // tid999 is not in the gate vocabulary; whether it survives the
        // sketch gate or not, the exact-LUT check must report any leak.
        let result = pipeline.process(&[b"tid0".to_vec(), b"tid999".to_vec()]);
        for leak in &result.leaks {
            assert_ne!(leak, b"tid0", "valid id must not leak");
        }
        assert!(result.restored_ids.contains(&b"tid0".to_vec()));
    }

    #[test]
    fn tf_accumulates_across_passes() {
        let mut pipeline = CortexPipeline::new();
        pipeline.set_gate(["tid0", "tid1"]);
        pipeline.process(&[b"tid0".to_vec()]);
        let result = pipeline.process(&[b"tid0".to_vec(), b"tid1".to_vec()]);
        // tid0 was observed twice overall → ranked first.
        assert_eq!(result.restored_ids.first(), Some(&b"tid0".to_vec()));
    }
}
