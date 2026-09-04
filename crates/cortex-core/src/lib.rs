//! # cortex-core — enhanced hllset-cortex pipeline (Rust port)
//!
//! The black-box interface between the DeepSeek-OCR encoder and decoder,
//! ported from the reference implementation and built on the tested
//! ewm-cortex primitives:
//!
//! ```text
//! tokens → hash → tokenLUT → HLLSet → materialize → gate_TF → restored → decoder
//! ```
//!
//! Modules:
//!
//! - [`encoding`] — simulated `tid{n}` encoder/decoder (opaque IDs only)
//! - [`gate`] — `gate_TF` output TokenGate (decoder-vocabulary limit) + exact membership
//! - [`lut`] — monotonic TF reverse index (collection intersection per bit; TF only on collision ties)
//! - [`pipeline`] — the [`CortexPipeline`] black box

pub mod encoding;
pub mod gate;
pub mod lut;
pub mod pipeline;

pub use encoding::{tid, SimCodec};
pub use gate::Gate;
pub use lut::TfLut;
pub use pipeline::{CortexPipeline, PipelineResult};
