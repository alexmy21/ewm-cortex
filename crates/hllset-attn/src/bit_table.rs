//! `BitKeyTable` — the learned key table of Mode A.
//!
//! One key vector per HLLSet bit cell: `E_bit[bit]` is the key of every
//! token whose hash decomposes to `bit`. The table has `TOTAL_CELLS =
//! 1024 × 32 = 32,768` rows of dimension `d`. Tokens in the same cell share
//! the same key; attention can route only to cells, and V plus context
//! disambiguate within a cell.

use hllset_core::{BITS_PER_REG, M};

/// Number of bit cells in the HLLSet lattice (1024 registers × 32 tz states).
pub const TOTAL_CELLS: usize = M * BITS_PER_REG as usize;

/// A learned `TOTAL_CELLS × d` key table, row-major.
///
/// The table is deliberately backend-agnostic: it is a flat `Vec<f32>` that
/// the Phase 0/2 training harness can read and update directly.
#[derive(Clone, Debug)]
pub struct BitKeyTable {
    d: usize,
    data: Vec<f32>,
}

impl BitKeyTable {
    /// Create a zero-initialised table of dimension `d`.
    pub fn new(d: usize) -> Self {
        Self {
            d,
            data: vec![0.0; TOTAL_CELLS * d],
        }
    }

    /// Create a table and fill each entry `(bit, dim)` with `f(bit, dim)`.
    pub fn from_fn(d: usize, mut f: impl FnMut(usize, usize) -> f32) -> Self {
        let mut table = Self::new(d);
        for bit in 0..TOTAL_CELLS {
            for dim in 0..d {
                table.data[bit * d + dim] = f(bit, dim);
            }
        }
        table
    }

    /// Key dimension.
    pub fn dim(&self) -> usize {
        self.d
    }

    /// Number of rows (always [`TOTAL_CELLS`]).
    pub fn len(&self) -> usize {
        TOTAL_CELLS
    }

    /// Whether the table is empty (never, for `TOTAL_CELLS > 0`).
    pub fn is_empty(&self) -> bool {
        TOTAL_CELLS == 0
    }

    /// Immutable view of row `bit`.
    pub fn get(&self, bit: u32) -> &[f32] {
        let start = bit as usize * self.d;
        &self.data[start..start + self.d]
    }

    /// Mutable view of row `bit`.
    pub fn get_mut(&mut self, bit: u32) -> &mut [f32] {
        let start = bit as usize * self.d;
        &mut self.data[start..start + self.d]
    }

    /// Copy `row` into row `bit` (panics if `row.len() != d`).
    pub fn set(&mut self, bit: u32, row: &[f32]) {
        assert_eq!(row.len(), self.d, "row length must match table dimension");
        self.get_mut(bit).copy_from_slice(row);
    }

    /// Iterate over all rows.
    pub fn rows(&self) -> impl Iterator<Item = &[f32]> {
        (0..TOTAL_CELLS).map(|bit| self.get(bit as u32))
    }

    /// Underlying flat storage (row-major).
    pub fn data(&self) -> &[f32] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_zeroed_and_sized() {
        let t = BitKeyTable::new(8);
        assert_eq!(t.dim(), 8);
        assert_eq!(t.len(), TOTAL_CELLS);
        assert_eq!(t.data().len(), TOTAL_CELLS * 8);
        assert!(t.get(0).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn set_get_roundtrip() {
        let mut t = BitKeyTable::new(4);
        let row: Vec<f32> = (0..4).map(|i| i as f32).collect();
        t.set(12_345, &row);
        assert_eq!(t.get(12_345), row.as_slice());
    }

    #[test]
    fn from_fn_fills_all_entries() {
        let t = BitKeyTable::from_fn(2, |bit, dim| (bit + dim) as f32);
        assert_eq!(t.get(0), &[0.0, 1.0]);
        assert_eq!(t.get(1), &[1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "row length must match table dimension")]
    fn set_wrong_length_panics() {
        let mut t = BitKeyTable::new(4);
        t.set(0, &[1.0, 2.0]);
    }
}
