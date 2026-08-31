//! # cortex-core — enhanced hllset-cortex pipeline (Rust port)
//!
//! The black-box interface between the DeepSeek-OCR encoder and decoder,
//! ported from the reference implementation and built on the tested
//! ewm-cortex primitives:
//!
//! ```text
//! encoding IDs → HLLSet → ∩ gate_TF → TF-LUT → materialize → restored IDs
//! ```
//!
//! Modules:
//!
//! - [`encoding`] — simulated `tid{n}` encoder/decoder (opaque IDs only)
//! - [`gate`] — `gate_TF` sketch gate + exact membership
//! - [`lut`] — monotonic TF reverse index (TF-ranked materialization)
//! - [`pipeline`] — the [`CortexPipeline`] black box

pub mod encoding;
pub mod gate;
pub mod lut;
pub mod pipeline;

pub use encoding::{tid, SimCodec};
pub use gate::Gate;
pub use lut::TfLut;
pub use pipeline::{CortexPipeline, PipelineResult};
