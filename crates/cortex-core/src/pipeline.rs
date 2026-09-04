//! `CortexPipeline` — the black-box interface (enhanced hllset-cortex).
//!
//! ```text
//! tokens → hash → tokenLUT → HLLSet → materialize → gate_TF → restored → decoder
//! ```
//!
//! The LUT and TF-LUT are **never gated** — they register and resolve every
//! ingested ID. Only the **output** passes through `gate_TF`, the decoder
//! vocabulary limit (TokenGate). Out-of-vocab IDs are reported, never hidden.
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
    /// IDs materialized from the full (ungated) HLLSet: collection
    /// intersection per bit, TF only on collision ties (bit order).
    pub materialized_ids: Vec<Vec<u8>>,
    /// Materialized IDs that survive the output TokenGate (decoder vocab).
    pub restored_ids: Vec<Vec<u8>>,
    /// Materialized IDs that are NOT in the decoder vocabulary (out-of-vocab;
    /// reported by the output gate).
    pub leaks: Vec<Vec<u8>>,
    /// LUT coverage over the full document HLLSet (1.0 = full resolution).
    pub coverage: f64,
}

impl PipelineResult {
    /// Whether the pass is leak-free (every materialized ID is in-vocab).
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

    /// Term frequency of an encoding ID (0 if never observed).
    pub fn tf(&self, id: &[u8]) -> u64 {
        self.lut.tf(id)
    }

    /// Process one document of encoding IDs.
    ///
    /// The TF-LUT is ungated: TF accumulates monotonically for every ingested
    /// ID, and materialization resolves the **full** HLLSet. Only the output
    /// is gated — out-of-vocab IDs are reported as `leaks`, never hidden.
    pub fn process(&mut self, ids: &[Vec<u8>]) -> PipelineResult {
        // 1. Ingest: the full document HLLSet. The LUT and TF-LUT are NEVER
        //    gated — they register every ingested ID.
        let doc = HLLSet::from_tokens(ids.iter());
        self.lut.observe(ids);

        // 2. Materialize the full HLLSet through the (ungated) TF-LUT.
        let materialized_ids = self.lut.materialize(&doc);

        // 3. Gate ONLY the output: the TokenGate is the decoder-vocabulary
        //    limit. Out-of-vocab IDs are reported as leaks, never hidden.
        let restored_ids = match &self.gate {
            Some(gate) => gate.filter_tokens(&materialized_ids),
            None => materialized_ids.clone(),
        };
        let leaks = match &self.gate {
            Some(gate) => gate.out_of_vocab(&materialized_ids),
            None => Vec::new(),
        };
        let coverage = self.lut.confidence(&doc);

        self.processed += 1;

        PipelineResult {
            input_ids: ids.to_vec(),
            doc_bits: doc.popcount(),
            materialized_ids,
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
    fn unknown_ids_are_reported_not_silently_dropped() {
        let mut pipeline = CortexPipeline::new();
        pipeline.set_gate(["tid0"]);
        // tid999 is out-of-vocab. The LUT is NOT gated: it must register and
        // materialize tid999 anyway; the output gate filters it and reports it.
        let result = pipeline.process(&[b"tid0".to_vec(), b"tid999".to_vec()]);
        assert!(
            result.materialized_ids.contains(&b"tid999".to_vec()),
            "LUT must not be gated: OOV ids are still materialized"
        );
        assert!(
            !result.restored_ids.contains(&b"tid999".to_vec()),
            "output gate must filter OOV ids"
        );
        assert!(result.restored_ids.contains(&b"tid0".to_vec()));
        assert!(result.leaks.contains(&b"tid999".to_vec()));
    }

    #[test]
    fn known_ids_roundtrip_materialized_equals_restored() {
        let mut pipeline = CortexPipeline::new();
        pipeline.set_gate(["tid0", "tid1", "tid2"]);
        let result = pipeline.process(&[b"tid0".to_vec(), b"tid1".to_vec()]);
        assert!(result.ok(), "leaks: {:?}", result.leaks);
        assert_eq!(result.materialized_ids, result.restored_ids);
        // Bit order, not TF order: both in-vocab ids resolve as singletons.
        let mut expected = vec![b"tid0".to_vec(), b"tid1".to_vec()];
        let mut actual = result.restored_ids.clone();
        expected.sort();
        actual.sort();
        assert_eq!(actual, expected);
        assert_eq!(result.coverage, 1.0);
    }

    #[test]
    fn tf_accumulates_across_passes() {
        let mut pipeline = CortexPipeline::new();
        pipeline.set_gate(["tid0", "tid1"]);
        pipeline.process(&[b"tid0".to_vec()]);
        let result = pipeline.process(&[b"tid0".to_vec(), b"tid1".to_vec()]);
        // TF accumulates: tid0 was observed twice, tid1 once.
        assert_eq!(pipeline.tf(b"tid0"), 2);
        assert_eq!(pipeline.tf(b"tid1"), 1);
        // Materialization resolves both as single-candidate bits.
        assert!(result.restored_ids.contains(&b"tid0".to_vec()));
        assert!(result.restored_ids.contains(&b"tid1".to_vec()));
    }
}
