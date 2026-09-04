//! Hashing utilities for HLLSet.
//!
//! Uses MurmurHash3-x64-128 via the `murmur3` crate.
//!
//! ## Seed convention
//!
//! - Seed 0: default for heterogeneous (n-gram) tokenization — the **G1
//!   compatibility layer** shared between heterogeneous and homogeneous data.
//! - Seeds 0/1/2: used in 3-seed cross-validation for homogeneous catalog data.

use sha1::{Digest, Sha1};
use std::io::Cursor;

/// MurmurHash3 64-bit with seed 0.
///
/// Identical to `murmur3_hash_seeded(data, 0)`. All n-gram tokens in
/// heterogeneous datasets use this hash.
pub fn murmur3_hash(data: &[u8]) -> u64 {
    murmur3_hash_seeded(data, 0)
}

/// Seeded MurmurHash3 64-bit (lower 64 bits of x64-128).
///
/// Seed is truncated to `u32` (the algorithm's native seed size).
/// Homogeneous datasets use multiple seeds; seed 0 is shared with
/// heterogeneous n-grams via the G1 layer.
pub fn murmur3_hash_seeded(data: &[u8], seed: u64) -> u64 {
    let hash_128 = murmur3::murmur3_x64_128(&mut Cursor::new(data), seed as u32)
        .expect("in-memory Cursor cannot fail");
    hash_128 as u64
}

/// SHA-1 hash of arbitrary bytes, returned as a hex string.
pub fn sha1_hex(data: &[u8]) -> String {
    let mut hasher = Sha1::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

/// Decompose a 64-bit hash into (register, trailing_zeros) position.
///
/// Used for populating the TokenLUT (reverse index) during materialization.
pub fn hash_to_position(hash: u64) -> (u32, u32) {
    use crate::core::hllset::{M, P};
    let reg = (hash as u32) & ((M as u32) - 1);
    let remaining = hash >> P;
    let zeros = (remaining.trailing_zeros() as u32).min(31);
    (reg, zeros)
}

/// Decompose a token into its (register, trailing_zeros) position.
pub fn token_to_position(token: &[u8]) -> (u32, u32) {
    hash_to_position(murmur3_hash(token))
}

/// Decompose a token into its position using a specific seed.
///
/// Used for multi-seed catalog hashing (homogeneous consensus).
pub fn token_to_position_seeded(token: &[u8], seed: u64) -> (u32, u32) {
    hash_to_position(murmur3_hash_seeded(token, seed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_murmur3_deterministic() {
        assert_eq!(murmur3_hash(b"hello"), murmur3_hash(b"hello"));
    }

    #[test]
    fn test_murmur3_seeded_differs() {
        let h0 = murmur3_hash_seeded(b"x", 0);
        let h1 = murmur3_hash_seeded(b"x", 1);
        assert_ne!(h0, h1);
    }

    #[test]
    fn test_seed_zero_equals_unseeded() {
        assert_eq!(murmur3_hash(b"test"), murmur3_hash_seeded(b"test", 0));
    }

    #[test]
    fn test_sha1_hex_length() {
        let h = sha1_hex(b"hello");
        assert_eq!(h.len(), 40);
    }
}
