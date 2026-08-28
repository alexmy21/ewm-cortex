//! `KBridge` — routes tokens through HLLSet K-storage into key vectors.
//!
//! The bridge is the single point of contact between the token realm and the
//! HLLSet realm. The transformer never hashes tokens inside the model graph;
//! it asks the bridge for a key, and the bridge consults the storage.

use hllset_core::HLLSet;

use crate::bit_table::BitKeyTable;
use crate::kstorage::{KStorage, KeyRef};

/// Routes tokens through a [`KStorage`] into key vectors.
#[derive(Clone, Debug)]
pub struct KBridge<S: KStorage> {
    storage: S,
}

impl<S: KStorage> KBridge<S> {
    /// Wrap a K-storage.
    pub fn new(storage: S) -> Self {
        Self { storage }
    }

    /// The underlying K-storage.
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// The key reference for a token (K direction).
    pub fn key_ref(&self, token: &[u8]) -> KeyRef {
        self.storage.key_of(token)
    }

    /// Mode A address-key: the key vector of a token.
    ///
    /// Tier 1 keys read their single cell from the table; tier 2 keys return
    /// the centroid of their multi-seed cells.
    pub fn key_vector(&self, token: &[u8], table: &BitKeyTable) -> Vec<f32> {
        let d = table.dim();
        let mut out = vec![0.0; d];
        let cells = self.key_ref(token).cells();
        for cell in cells {
            let row = table.get(cell);
            for (out_i, &v) in row.iter().enumerate() {
                out[out_i] += v;
            }
        }
        let count = self.key_ref(token).cells().len().max(1) as f32;
        for v in &mut out {
            *v /= count;
        }
        out
    }

    /// Key vector plus a residual (positional/contextual), added in place.
    pub fn key_vector_with(
        &self,
        token: &[u8],
        table: &BitKeyTable,
        residual: &[f32],
    ) -> Vec<f32> {
        let mut key = self.key_vector(token, table);
        assert_eq!(
            key.len(),
            residual.len(),
            "residual length must match table dimension"
        );
        for (k, &r) in key.iter_mut().zip(residual) {
            *k += r;
        }
        key
    }

    /// Materialize the candidate tokens for a sub-lattice HLLSet (V direction).
    pub fn candidates(&self, hllset: &HLLSet) -> Vec<Vec<u8>> {
        self.storage.candidates(hllset)
    }

    /// Coverage gauge for a sub-lattice HLLSet.
    pub fn confidence(&self, hllset: &HLLSet) -> f64 {
        self.storage.confidence(hllset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kstorage::{CatalogLutStorage, TokenLutStorage};

    #[test]
    fn address_key_reads_single_cell() {
        let bridge = KBridge::new(TokenLutStorage::from_tokens(&["hello"]));
        let bit = match bridge.key_ref(b"hello") {
            KeyRef::Cell(b) => b,
            other => panic!("expected Cell, got {other:?}"),
        };
        let mut table = BitKeyTable::new(4);
        table.set(bit, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(bridge.key_vector(b"hello", &table), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn multi_seed_key_is_cell_centroid() {
        let bridge = KBridge::new(CatalogLutStorage::from_values(&["alice"]));
        let cells = bridge.key_ref(b"alice").cells();
        assert_eq!(cells.len(), 3);
        let mut table = BitKeyTable::new(2);
        for (i, &c) in cells.iter().enumerate() {
            table.set(c, &[i as f32, (2 * i) as f32]);
        }
        let key = bridge.key_vector(b"alice", &table);
        // mean of rows [0,0], [1,2], [2,4] = [1,2]
        assert_eq!(key, vec![1.0, 2.0]);
    }

    #[test]
    fn key_vector_with_adds_residual() {
        let bridge = KBridge::new(TokenLutStorage::from_tokens(&["hello"]));
        let bit = match bridge.key_ref(b"hello") {
            KeyRef::Cell(b) => b,
            other => panic!("expected Cell, got {other:?}"),
        };
        let mut table = BitKeyTable::new(3);
        table.set(bit, &[1.0, 1.0, 1.0]);
        let key = bridge.key_vector_with(b"hello", &table, &[0.5, -1.0, 2.0]);
        assert_eq!(key, vec![1.5, 0.0, 3.0]);
    }
}
