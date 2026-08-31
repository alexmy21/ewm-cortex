//! # hllset-repro — Phase 0 reproduction transformer (token realm)
//!
//! A zero-dependency, hand-rolled character-level transformer used as the
//! Phase 0 baseline of `_DOCS/arch/HLLSET_LUT_TRANSFORMER_ARCHITECTURE.md`.
//! Later phases replace the K path with `hllset-attn` K-storage while this
//! crate keeps owning the traditional matrix machinery (Q, V, FFN, head).
//!
//! Modules:
//!
//! - [`autograd`] — micrograd-style tensor autograd (matmul, softmax,
//!   layernorm, cross-entropy, Adam).
//! - [`data`] — character-level dataset and batch extraction.
//! - [`model`] — the decoder-only transformer.

pub mod autograd;
pub mod corpus;
pub mod data;
pub mod model;
pub mod phase1;
pub mod phase2;
pub mod phase3;

pub use autograd::{Adam, Tensor, XorShift};
pub use corpus::CORPUS;
pub use data::{batch_at, CharDataset};
pub use model::{Config, Transformer, TOTAL_CELLS};
pub use phase1::{attach, CollisionStats, Phase1Report};
pub use phase2::{bits_for_dataset, run_phase2, Phase2Report};
pub use phase3::{run_phase3, Phase3Report};
