//! # hllset-materialize — the two morphisms of the HLLSet bridge
//!
//! - **Ingest** (`ingest.rs`) — tokens → HLLSets: one streaming pass that
//!   updates all n-gram/seed-n LUTs, builds the working HLLSets internally,
//!   updates TF, and reports the originals for the HLLSet-LUT.
//! - **Materialize** (`materialize.rs`) — HLLSets → tokens: `TokenLUT`
//!   (single-seed reverse index), `CatalogLUT` (multi-seed consensus),
//!   and the `MaterializeEngine` trait that future backends implement.
//! - **FPGA delegation** (`fpga.rs`, feature `fpga-sim`) — HLLSet operations
//!   executed by the `hllset-fpga-simulator` golden model through the
//!   `fpga-hostif` protocol.

pub mod ingest;
pub mod materialize;

#[cfg(feature = "fpga-sim")]
pub mod fpga;

#[cfg(feature = "fpga-sim")]
pub use fpga::{FpgaSim, SimFpgaDriver};
pub use ingest::{IngestOutput, IngestSink, IngestStats, Ingestor, NoopSink};
pub use materialize::{
    materialize_homogeneous_consensus, materialize_inlut, CatalogLUT, MaterializedResult, TokenLUT,
};

use hllset_core::HLLSet;

/// Abstract materialization backend.
///
/// Future engines (chunked, FPGA sim, physical) implement this trait and
/// must produce bit-exact results matching the in-memory reference.
pub trait MaterializeEngine {
    fn materialize(&self, hllset: &HLLSet) -> Result<Vec<Vec<u8>>, MaterializeError>;

    fn name(&self) -> &str;

    fn lut_count(&self) -> usize;

    fn is_hardware(&self) -> bool {
        false
    }
}

/// Materialization errors.
#[derive(Debug)]
pub enum MaterializeError {
    LutNotLoaded(String),
    Query(String),
    IO(std::io::Error),
}

impl std::fmt::Display for MaterializeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MaterializeError::LutNotLoaded(k) => write!(f, "LUT not loaded: {k}"),
            MaterializeError::Query(e) => write!(f, "query error: {e}"),
            MaterializeError::IO(e) => write!(f, "io: {e}"),
        }
    }
}

impl std::error::Error for MaterializeError {}
