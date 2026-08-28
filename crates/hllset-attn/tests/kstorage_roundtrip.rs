//! Integration tests: K-storage round-trips and the K-bridge.
//!
//! These mirror the invariants of
//! `_DOCS/arch/HLLSET_LUT_TRANSFORMER_ARCHITECTURE.md` §8:
//! key_of is total and deterministic; candidates recover the V side;
//! confidence is 1.0 exactly when the LUT covers the sub-lattice.

use hllset_attn::bit_table::BitKeyTable;
use hllset_attn::bridge::KBridge;
use hllset_attn::kstorage::{CatalogLutStorage, KStorage, KeyRef, TokenLutStorage};
use hllset_attn::setkey::{bss_coverage, jaccard, overlap, retrieve};
use hllset_core::core::hashing::{murmur3_hash_seeded, token_to_position};
use hllset_core::{BITS_PER_REG, HLLSet};

#[test]
fn token_lut_key_of_matches_hash_decomposition() {
    let storage = TokenLutStorage::from_tokens(&["alpha", "beta"]);
    let (reg, zeros) = token_to_position(b"alpha");
    assert_eq!(
        storage.key_of(b"alpha"),
        KeyRef::Cell(reg * BITS_PER_REG + zeros)
    );
}

#[test]
fn token_lut_candidates_recover_all_tokens() {
    let storage = TokenLutStorage::from_tokens(&["alpha", "beta", "gamma"]);
    let hllset = HLLSet::from_tokens(&["alpha", "beta", "gamma"]);
    let candidates = storage.candidates(&hllset);
    for token in ["alpha", "beta", "gamma"] {
        assert!(
            candidates.iter().any(|c| c == token.as_bytes()),
            "missing candidate: {token}"
        );
    }
}

#[test]
fn token_lut_confidence_is_full_on_covered_set() {
    let storage = TokenLutStorage::from_tokens(&["alpha", "beta", "gamma"]);
    let hllset = HLLSet::from_tokens(&["alpha", "beta", "gamma"]);
    assert_eq!(storage.confidence(&hllset), 1.0);
}

#[test]
fn token_lut_confidence_drops_on_unknown_bits() {
    let storage = TokenLutStorage::from_tokens(&["alpha"]);
    let hllset = HLLSet::from_tokens(&["alpha", "unregistered-token-xyz"]);
    let confidence = storage.confidence(&hllset);
    assert!(confidence < 1.0, "confidence={confidence}");
    assert!(confidence > 0.0, "confidence={confidence}");
}

#[test]
fn catalog_lut_key_is_multi_seed_with_quorum() {
    let storage = CatalogLutStorage::from_values(&["alice@example.com"]);
    match storage.key_of(b"alice@example.com") {
        KeyRef::Cells(cells, quorum) => {
            assert_eq!(cells.len(), 3);
            assert_eq!(quorum, 2);
        }
        other => panic!("expected Cells, got {other:?}"),
    }
}

#[test]
fn catalog_lut_consensus_roundtrip() {
    let values = ["alice", "bob", "carol"];
    let storage = CatalogLutStorage::from_values(&values);

    // Build the 3-seed HLLSet exactly as homogeneous ingestion would.
    let mut hllset = HLLSet::new();
    for seed in [0u64, 1, 2] {
        for value in values {
            let hash = murmur3_hash_seeded(value.as_bytes(), seed);
            hllset.add_hash(hash);
        }
    }

    assert_eq!(storage.confidence(&hllset), 1.0);
    let candidates = storage.candidates(&hllset);
    for value in values {
        assert!(
            candidates.iter().any(|c| c == value.as_bytes()),
            "missing candidate: {value}"
        );
    }
}

#[test]
fn bit_table_and_bridge_roundtrip() {
    let storage = TokenLutStorage::from_tokens(&["hello"]);
    let bridge = KBridge::new(storage);
    let bit = match bridge.key_ref(b"hello") {
        KeyRef::Cell(b) => b,
        other => panic!("expected Cell, got {other:?}"),
    };

    let mut table = BitKeyTable::new(4);
    table.set(bit, &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(bridge.key_vector(b"hello", &table), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn setkey_scores_and_retrieval() {
    let query = HLLSet::from_tokens(&["alpha"]);
    let context = HLLSet::from_tokens(&["alpha", "beta", "gamma"]);

    assert!((jaccard(&query, &query) - 1.0).abs() < 1e-9);
    assert_eq!(overlap(&query, &context), query.popcount() as f64);
    assert!((bss_coverage(&context, &query) - 1.0).abs() < 1e-9);

    let retrieval = retrieve(&context, &query, vec![b"alpha".to_vec()]);
    assert!((retrieval.score - 1.0).abs() < 1e-9);
    assert_eq!(retrieval.candidates, vec![b"alpha".to_vec()]);
}
