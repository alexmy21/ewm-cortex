//! Ingest — the forward morphism (symmetric to `materialize.rs`).
//!
//! Ingestion is the more complex morphism: we stream data **only once**, so
//! every token must be fully processed in a single pass:
//!
//! 1. update **all n-gram/seed-n token LUTs** (the multi-seed `CatalogLUT`);
//! 2. update the **working HLLSets** — built internally inside the loop and
//!    never exposed until the pass ends;
//! 3. update **TF in all LUTs** (per-token TF + the 32K bit-TF vector);
//! 4. report each final original HLLSet to an [`IngestSink`] so the caller
//!    can update its HLLSet-LUT (`<SHA1, TH>`).
//!
//! n-gram construction happens upstream (the stream supplies 1-, 2- or
//! 3-gram byte strings); the ingestor is n-gram agnostic and treats every
//! token through all seeds.

use std::collections::HashMap;

use hllset_core::core::hashing::{murmur3_hash_seeded, sha1_hex};
use hllset_core::core::tfvec::TFVec;
use hllset_core::HLLSet;

use crate::materialize::CatalogLUT;

/// Receiver for the original HLLSets produced by a pass (point 4).
///
/// The canonical receiver is the HLLSet-LUT (`<SHA1, TH>`): register the
/// original and count the touch.
pub trait IngestSink {
    /// An original HLLSet produced by the just-finished pass, with its
    /// content address.
    fn on_original(&mut self, hllset: &HLLSet, sha1: String);
}

/// A no-op sink (for passes that don't need HLLSet-LUT updates).
#[derive(Clone, Debug, Default)]
pub struct NoopSink;

impl IngestSink for NoopSink {
    fn on_original(&mut self, _hllset: &HLLSet, _sha1: String) {}
}

/// Per-pass statistics.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IngestStats {
    /// Tokens seen in this pass.
    pub tokens: u64,
    /// Tokens that were new to the LUTs.
    pub new_lut_entries: u64,
    /// Number of seed channels.
    pub channels: usize,
    /// Largest channel popcount of the pass.
    pub active_bits: u64,
}

/// The result of one pass: stats plus the final channel HLLSets.
#[derive(Clone, Debug, Default)]
pub struct IngestOutput {
    pub stats: IngestStats,
    /// One HLLSet per seed channel (G1, G2, G3, …).
    pub channels: Vec<HLLSet>,
}

/// The streaming ingestor.
#[derive(Clone, Debug)]
pub struct Ingestor {
    seeds: Vec<u64>,
    catalog: CatalogLUT,
    token_tf: HashMap<Vec<u8>, u64>,
    bit_tf: TFVec,
    total_tokens: u64,
    total_new_entries: u64,
}

impl Ingestor {
    /// Create an ingestor with the given seeds (default `[0, 1, 2]`).
    pub fn new(seeds: &[u64]) -> Self {
        assert!(seeds.len() >= 2, "need at least 2 seeds for consensus");
        Self {
            seeds: seeds.to_vec(),
            catalog: CatalogLUT::new().with_seeds(seeds),
            token_tf: HashMap::new(),
            bit_tf: TFVec::new(),
            total_tokens: 0,
            total_new_entries: 0,
        }
    }

    /// The multi-seed catalog LUT (all seed-n token LUTs).
    pub fn catalog(&self) -> &CatalogLUT {
        &self.catalog
    }

    /// Per-token TF (0 if never seen).
    pub fn token_tf(&self, token: &[u8]) -> u64 {
        self.token_tf.get(token).copied().unwrap_or(0)
    }

    /// The cumulative 32K bit-TF vector over all ingested HLLSets.
    pub fn bit_tf(&self) -> &TFVec {
        &self.bit_tf
    }

    /// Total tokens ingested across all passes.
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// Total new LUT entries across all passes.
    pub fn total_new_entries(&self) -> u64 {
        self.total_new_entries
    }

    /// One streaming pass over the tokens.
    ///
    /// Each token is fully processed before the next is read: LUT update,
    /// working-HLLSet update, and TF update all happen inside the loop. The
    /// working HLLSets are internal and only leave the loop when the pass
    /// ends (as `IngestOutput::channels`, and via `sink.on_original`).
    pub fn ingest_stream<I, B>(&mut self, tokens: I, sink: &mut impl IngestSink) -> IngestOutput
    where
        I: IntoIterator<Item = B>,
        B: AsRef<[u8]>,
    {
        // Working HLLSets — internal to this pass, never exposed mid-stream.
        let mut working: Vec<HLLSet> = (0..self.seeds.len()).map(|_| HLLSet::new()).collect();

        let mut pass_tokens = 0u64;
        let mut new_entries = 0u64;

        for token in tokens {
            let t = token.as_ref();

            // 1. Update all n-gram/seed-n token LUTs (deduplicated).
            if self.catalog.positions_of(t).is_none() {
                self.catalog.insert(t.to_vec());
                new_entries += 1;
            }

            // 2. Update the working HLLSets (one bit per seed channel).
            for (i, &seed) in self.seeds.iter().enumerate() {
                working[i].add_hash(murmur3_hash_seeded(t, seed));
            }

            // 3. Update per-token TF.
            *self.token_tf.entry(t.to_vec()).or_insert(0) += 1;
            pass_tokens += 1;
        }

        // Finalize: bit-TF accumulates one touch per channel HLLSet, then
        // each original is emitted to the sink (point 4 — HLLSet-LUT).
        for hll in &working {
            self.bit_tf.increment_from_hllset(hll, 1.0);
        }
        for hll in &working {
            let cid = sha1_hex(&hll.to_bytes());
            sink.on_original(hll, cid);
        }

        self.total_tokens += pass_tokens;
        self.total_new_entries += new_entries;

        let stats = IngestStats {
            tokens: pass_tokens,
            new_lut_entries: new_entries,
            channels: self.seeds.len(),
            active_bits: working.iter().map(|h| h.popcount()).max().unwrap_or(0),
        };

        IngestOutput {
            stats,
            channels: working,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct CollectSink {
        originals: Vec<(String, u64)>,
    }

    impl IngestSink for CollectSink {
        fn on_original(&mut self, hllset: &HLLSet, sha1: String) {
            self.originals.push((sha1, hllset.popcount()));
        }
    }

    #[test]
    fn one_pass_updates_luts_hllsets_tf_and_sink() {
        let mut ingestor = Ingestor::new(&[0, 1, 2]);
        let mut sink = CollectSink::default();

        let out = ingestor.ingest_stream(["alpha", "beta", "alpha"], &mut sink);

        assert_eq!(out.stats.tokens, 3);
        assert_eq!(out.stats.channels, 3);
        assert_eq!(out.stats.new_lut_entries, 2, "alpha counted once");
        assert_eq!(ingestor.catalog().len(), 2);
        assert_eq!(ingestor.token_tf(b"alpha"), 2);
        assert_eq!(ingestor.token_tf(b"beta"), 1);
        assert_eq!(out.channels.len(), 3);

        // Each channel HLLSet was emitted to the sink.
        assert_eq!(sink.originals.len(), 3);
        assert_eq!(sink.originals[0].1, out.channels[0].popcount());

        // bit-TF: one touch per channel HLLSet.
        let expected_total: f64 = out.channels.iter().map(|h| h.popcount() as f64).sum();
        assert_eq!(ingestor.bit_tf().total(), expected_total);
    }

    #[test]
    fn working_hllsets_are_not_exposed_mid_stream() {
        // The Ingestor has no public accessor for the working sets — the
        // only HLLSets visible after a pass are `IngestOutput::channels`
        // and the sink originals. Verified by API shape: nothing to assert
        // beyond the pass completing and channels being consistent.
        let mut ingestor = Ingestor::new(&[0, 1]);
        let out = ingestor.ingest_stream(["x", "y"], &mut NoopSink);
        assert_eq!(out.channels.len(), 2);
        assert_eq!(out.stats.active_bits, out.channels[0].popcount().max(out.channels[1].popcount()));
    }

    #[test]
    fn passes_accumulate_tf_monotonically() {
        let mut ingestor = Ingestor::new(&[0, 1, 2]);
        let first = ingestor.ingest_stream(["a", "b"], &mut NoopSink);
        let second = ingestor.ingest_stream(["b", "c"], &mut NoopSink);
        assert_eq!(ingestor.total_tokens(), 4);
        assert_eq!(first.stats.tokens + second.stats.tokens, 4);
        assert_eq!(ingestor.token_tf(b"b"), 2);
        assert!(ingestor.bit_tf().total() >= first.channels.iter().map(|h| h.popcount() as f64).sum());
    }
}
