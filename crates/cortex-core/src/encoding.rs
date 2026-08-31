//! Simulated DeepSeek-OCR encoder/decoder over opaque encoding IDs.
//!
//! The boundary is encoding-agnostic: the cortex never sees real tokens,
//! only `tid{n}` IDs. This module simulates both sides of the boundary for
//! tests and the CLI:
//!
//! ```text
//! text --encode--> tid{n} IDs --(cortex black box)--> restored IDs --decode--> text
//! ```

use std::collections::HashMap;

/// The canonical encoding-ID form: `tid{n}`.
pub fn tid(n: usize) -> String {
    format!("tid{n}")
}

/// A deterministic word ↔ `tid{n}` codec for a fixed vocabulary.
#[derive(Clone, Debug)]
pub struct SimCodec {
    /// Valid encoding IDs, in word-sorted order (`tid0`, `tid1`, …).
    vocab: Vec<Vec<u8>>,
    id_to_word: HashMap<Vec<u8>, Vec<u8>>,
    word_to_id: HashMap<Vec<u8>, Vec<u8>>,
}

impl SimCodec {
    /// Build a codec from the distinct whitespace tokens of `text`.
    pub fn from_text(text: &str) -> Self {
        let mut words: Vec<Vec<u8>> = text
            .split_whitespace()
            .map(|w| w.as_bytes().to_vec())
            .collect();
        words.sort();
        words.dedup();
        Self::from_words(words)
    }

    /// Build a codec from an explicit word list (sorted, deduplicated).
    pub fn from_words<I, B>(words: I) -> Self
    where
        I: IntoIterator<Item = B>,
        B: AsRef<[u8]>,
    {
        let mut ws: Vec<Vec<u8>> = words.into_iter().map(|w| w.as_ref().to_vec()).collect();
        ws.sort();
        ws.dedup();
        let vocab: Vec<Vec<u8>> = (0..ws.len()).map(|i| tid(i).into_bytes()).collect();
        let word_to_id: HashMap<Vec<u8>, Vec<u8>> = ws
            .iter()
            .cloned()
            .zip(vocab.iter().cloned())
            .collect();
        let id_to_word: HashMap<Vec<u8>, Vec<u8>> = vocab
            .iter()
            .cloned()
            .zip(ws.iter().cloned())
            .collect();
        Self {
            vocab,
            id_to_word,
            word_to_id,
        }
    }

    /// Valid encoding IDs (the decoder vocabulary).
    pub fn vocab(&self) -> &[Vec<u8>] {
        &self.vocab
    }

    /// Encode one word to its `tid{n}` ID (None if out of vocabulary).
    pub fn encode_word(&self, word: &[u8]) -> Option<Vec<u8>> {
        self.word_to_id.get(word).cloned()
    }

    /// Encode text to encoding IDs, dropping out-of-vocabulary words.
    pub fn encode_text(&self, text: &str) -> Vec<Vec<u8>> {
        text.split_whitespace()
            .filter_map(|w| self.encode_word(w.as_bytes()))
            .collect()
    }

    /// Decode one encoding ID to its word (None if unknown).
    pub fn decode_id(&self, id: &[u8]) -> Option<Vec<u8>> {
        self.id_to_word.get(id).cloned()
    }

    /// Decode a sequence of IDs to text (unknown IDs render as `<id>`).
    pub fn decode_ids(&self, ids: &[Vec<u8>]) -> String {
        ids.iter()
            .map(|id| match self.decode_id(id) {
                Some(word) => String::from_utf8_lossy(&word).to_string(),
                None => format!("<{}>", String::from_utf8_lossy(id)),
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_roundtrip() {
        let codec = SimCodec::from_text("the cat sat the dog ran");
        let ids = codec.encode_text("the cat sat");
        assert_eq!(ids.len(), 3);
        assert_eq!(codec.decode_ids(&ids), "the cat sat");
    }

    #[test]
    fn oov_words_are_dropped() {
        let codec = SimCodec::from_text("alpha beta");
        let ids = codec.encode_text("alpha gamma");
        assert_eq!(ids.len(), 1);
        assert_eq!(codec.decode_id(&ids[0]).unwrap(), b"alpha");
    }

    #[test]
    fn vocab_is_sorted_and_tid_indexed() {
        let codec = SimCodec::from_words(["b", "a"]);
        assert_eq!(codec.vocab()[0], b"tid0");
        assert_eq!(codec.decode_id(&codec.vocab()[0]).unwrap(), b"a");
        assert_eq!(codec.decode_id(&codec.vocab()[1]).unwrap(), b"b");
    }
}
