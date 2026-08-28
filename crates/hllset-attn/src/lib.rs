//! # hllset-attn — HLLSet K-storage for attention
//!
//! This crate implements the K side of attention on top of the HLLSet
//! lattice, per `_DOCS/arch/HLLSET_LUT_TRANSFORMER_ARCHITECTURE.md`.
//!
//! ## Two realms
//!
//! - **Token realm** (not in this crate): embeddings, Q/V projections,
//!   FFN, softmax — the traditional transformer matrix machinery.
//! - **HLLSet realm** (this crate): the 32K-bit lattice and the LUTs that
//!   store it. A key is a bit address; the label of the key is its
//!   collision group.
//!
//! ## Modules
//!
//! | Module | Role |
//! |--------|------|
//! | [`kstorage`] | `KStorage` trait + `TokenLUT`/`CatalogLUT` wrappers (tier 1/2) |
//! | [`bit_table`] | `BitKeyTable` — the learned 32,768 × d key table (Mode A) |
//! | [`bridge`] | `KBridge` — routes a token through K-storage into a key vector |
//! | [`setkey`] | HLLSet-native attention scores: overlap, Jaccard, BSS (Mode B) |
//! | [`hybrid`] | hybrid dense + lattice score (Mode C) |
//!
//! ## The K-storage interface
//!
//! ```rust
//! use hllset_attn::{KStorage, TokenLutStorage, KeyRef};
//! use hllset_core::HLLSet;
//!
//! let storage = TokenLutStorage::from_tokens(&["hello", "world"]);
//! let key = storage.key_of(b"hello");       // KeyRef::Cell(reg * 32 + tz)
//! let hllset = HLLSet::from_tokens(&["hello", "world"]);
//! let candidates = storage.candidates(&hllset);   // the V side
//! let confidence = storage.confidence(&hllset);    // 1.0 iff LUT covers all bits
//! ```

pub mod bit_table;
pub mod bridge;
pub mod context_vocab;
pub mod hybrid;
pub mod kstorage;
pub mod setkey;
pub mod token_mask;

pub use bit_table::BitKeyTable;
pub use bridge::KBridge;
pub use context_vocab::{ContextVocabulary, VocabDelta};
pub use hybrid::{hybrid_logits, hybrid_score, HybridConfig};
pub use kstorage::{CatalogLutStorage, KStorage, KeyRef, TokenLutStorage};
pub use setkey::{bss_coverage, bss_symmetric, jaccard, overlap};
pub use token_mask::TokenMask;
