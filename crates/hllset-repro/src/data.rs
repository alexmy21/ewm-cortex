//! Character-level dataset for the Phase 0 reproduction harness.
//!
//! Zero-dependency tokenization: the vocabulary is the set of distinct
//! characters in the corpus. This is the smallest possible token realm and
//! keeps the harness fully inspectable.

use std::collections::HashMap;

/// A small character-level dataset.
#[derive(Clone, Debug)]
pub struct CharDataset {
    /// Vocabulary (sorted distinct characters).
    chars: Vec<char>,
    stoi: HashMap<char, usize>,
    /// The full corpus encoded as token ids.
    data: Vec<usize>,
}

impl CharDataset {
    /// Build a dataset from a text corpus.
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort_unstable();
        chars.dedup();
        let stoi: HashMap<char, usize> =
            chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let data: Vec<usize> = text.chars().map(|c| stoi[&c]).collect();
        Self { chars, stoi, data }
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.chars.len()
    }

    /// The sorted vocabulary.
    pub fn chars(&self) -> &[char] {
        &self.chars
    }

    /// Encoded corpus.
    pub fn data(&self) -> &[usize] {
        &self.data
    }

    /// Encode a string to token ids (unknown chars panic — corpus-consistent).
    pub fn encode(&self, s: &str) -> Vec<usize> {
        s.chars().map(|c| self.stoi[&c]).collect()
    }

    /// Decode token ids back to a string.
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|&i| self.chars[i]).collect()
    }
}

/// A single training window: `block` input tokens and `block` targets
/// (the same sequence shifted left by one).
#[derive(Clone, Debug)]
pub struct Batch {
    pub inputs: Vec<usize>,
    pub targets: Vec<usize>,
}

/// Extract a batch from the corpus at the given start offset.
pub fn batch_at(data: &[usize], start: usize, block: usize) -> Batch {
    assert!(start + block < data.len(), "batch window out of range");
    Batch {
        inputs: data[start..start + block].to_vec(),
        targets: data[start + 1..start + 1 + block].to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_and_roundtrip() {
        let ds = CharDataset::from_text("ab ca");
        assert_eq!(ds.vocab_size(), 4); // 'a','b','c',' '
        let ids = ds.encode("ab c");
        assert_eq!(ds.decode(&ids), "ab c");
    }

    #[test]
    fn batch_targets_are_shifted() {
        let ds = CharDataset::from_text("abcdef");
        let batch = batch_at(ds.data(), 0, 3);
        assert_eq!(batch.inputs, ds.encode("abc"));
        assert_eq!(batch.targets, ds.encode("bcd"));
    }
}
