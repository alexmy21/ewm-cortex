//! `TokenMask` — a gate over the LUT's canonical token store.
//!
//! The mask stores **content addresses** (murmur3 seed-0 hashes) instead of
//! token bytes. Bytes live only in the LUT; the mask references them by
//! address, so the vocabulary view can never drift out of sync with the
//! K-storage. Resolving an address back to bytes is a boundary crossing
//! ([`crate::kstorage::KStorage::resolve`]).
//!
//! Two gates sit over one LUT:
//!
//! - the **bit gate** — the HLLSet context sub-lattice (compact, lossy:
//!   collision groups share a bit);
//! - the **hash gate** — this mask (exact token identity, content-addressed).

use std::collections::HashSet;

use hllset_core::core::hashing::murmur3_hash;

/// A gate of content addresses (murmur3 seed-0 hashes) over a K-storage.
#[derive(Clone, Debug, Default)]
pub struct TokenMask {
    hashes: HashSet<u64>,
}

impl TokenMask {
    /// Create an empty mask.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a content address. Returns `true` if it was newly inserted.
    pub fn insert(&mut self, hash: u64) -> bool {
        self.hashes.insert(hash)
    }

    /// Remove a content address. Returns `true` if it was present.
    pub fn remove(&mut self, hash: u64) -> bool {
        self.hashes.remove(&hash)
    }

    /// Whether the address is in the mask.
    pub fn contains_hash(&self, hash: u64) -> bool {
        self.hashes.contains(&hash)
    }

    /// Whether the token (by content address) is in the mask.
    pub fn contains_token(&self, token: &[u8]) -> bool {
        self.hashes.contains(&murmur3_hash(token))
    }

    /// Number of gated tokens.
    pub fn len(&self) -> usize {
        self.hashes.len()
    }

    /// Whether the mask is empty.
    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }

    /// Sorted content addresses (deterministic iteration).
    pub fn sorted_hashes(&self) -> Vec<u64> {
        let mut v: Vec<u64> = self.hashes.iter().copied().collect();
        v.sort_unstable();
        v
    }

    /// Iterate over the content addresses.
    pub fn hashes(&self) -> impl Iterator<Item = u64> + '_ {
        self.hashes.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_holds_addresses_not_bytes() {
        let mut mask = TokenMask::new();
        let hash = murmur3_hash(b"hello");
        assert!(mask.insert(hash));
        assert!(mask.contains_hash(hash));
        assert!(mask.contains_token(b"hello"));
        assert!(!mask.contains_token(b"world"));
        assert_eq!(mask.len(), 1);
        assert!(mask.remove(hash));
        assert!(mask.is_empty());
    }
}
