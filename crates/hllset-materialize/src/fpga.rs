//! FPGA delegation (feature `fpga-sim`) — HLLSet operations executed by the
//! `hllset-fpga-simulator` golden model through the `fpga-hostif` protocol.
//!
//! This module is the host-side socket for the substitution rule:
//!
//! ```text
//! ewm-cortex ── fpga-hostif::{Command, FpgaDriver} ──┬─ SimFpgaDriver (here,
//!                                                    │   golden model)
//!                                                    └─ future board driver
//! ```
//!
//! [`SimFpgaDriver`] executes commands eagerly against
//! [`fpga_hllset::DensePlane`] (the bit-exact soldered core). [`FpgaSim`] is
//! the ergonomic HLLSet-level wrapper used by the rest of the workspace:
//! every operation submits real `Command`s and drains `Response`s, so host
//! code written against `FpgaSim` runs unchanged against a physical driver.
//!
//! Plane upload/download is a simulator-side bootstrap (there is no wire
//! command for it yet); all *processing* goes through the protocol.

use std::collections::{HashMap, VecDeque};

use fpga_hostif::{
    AlgebraOp, BoardSpec, Command, FpgaDriver, FpgaError, FpgaResult, PlaneId, Reg, Response, Tz,
};
use fpga_hllset::DensePlane;
use hllset_core::HLLSet;

use crate::materialize::TokenLUT;

const TF_LANES: usize = 32_768;

/// A functional simulator driver: executes every [`Command`] eagerly against
/// the `fpga-hllset` golden model and queues the resulting [`Response`].
///
/// The simulator's own `fpga-board::SimDriver` is currently a Phase-0 stub;
/// this driver is the Phase-8-style execution model. When `fpga-board`
/// matures, swap this type for it — host code only sees [`FpgaDriver`].
#[derive(Debug)]
pub struct SimFpgaDriver {
    name: String,
    planes: HashMap<PlaneId, DensePlane>,
    next_plane: PlaneId,
    tf: HashMap<PlaneId, Vec<i64>>,
    lut: HashMap<(Reg, Tz), Vec<Vec<u8>>>,
    responses: VecDeque<Response>,
    submitted: u64,
    spec: Option<BoardSpec>,
}

impl SimFpgaDriver {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            planes: HashMap::new(),
            next_plane: 1,
            tf: HashMap::new(),
            lut: HashMap::new(),
            responses: VecDeque::new(),
            submitted: 0,
            spec: None,
        }
    }

    fn alloc(&mut self, plane: DensePlane) -> PlaneId {
        let id = self.next_plane;
        self.next_plane += 1;
        self.planes.insert(id, plane);
        id
    }

    fn plane(&self, id: PlaneId) -> FpgaResult<&DensePlane> {
        self.planes
            .get(&id)
            .ok_or(FpgaError::UnknownPlane(id))
    }

    /// Simulator bootstrap: make a host HLLSet resident in the FPGA.
    /// (No wire command exists yet; physical transport will add one.)
    pub fn upload(&mut self, hllset: &HLLSet) -> PlaneId {
        let plane =
            DensePlane::from_roaring_bytes(&hllset.to_bytes()).unwrap_or_default();
        self.alloc(plane)
    }

    /// Simulator bootstrap: read a resident plane back as a host HLLSet.
    pub fn download(&self, id: PlaneId) -> FpgaResult<HLLSet> {
        let plane = self.plane(id)?;
        HLLSet::from_bytes(&plane.to_roaring_bytes())
            .ok_or_else(|| FpgaError::Protocol("plane bytes did not parse as HLLSet".into()))
    }

    /// Load a token LUT into the FPGA BRAM model (host-side bootstrap for
    /// [`Command::Lookup`]).
    pub fn load_lut(&mut self, lut: &TokenLUT) {
        for reg in 0..1024u32 {
            for tz in 0..32u32 {
                if let Some(tokens) = lut.get(reg, tz) {
                    if !tokens.is_empty() {
                        self.lut
                            .insert((reg as Reg, tz as Tz), tokens.clone());
                    }
                }
            }
        }
    }

    fn exec(&mut self, cmd: Command) -> FpgaResult<Response> {
        match cmd {
            Command::Ingest { tokens, seeds } => {
                let seeds = if seeds.is_empty() { vec![0u64] } else { seeds };
                let mut plane = DensePlane::new();
                for token in &tokens {
                    for &seed in &seeds {
                        plane.add_hash(fpga_hllset::murmur3_low64(token, seed));
                    }
                }
                Ok(Response::Plane {
                    id: self.alloc(plane),
                })
            }
            Command::Gram { tokens } => {
                let mut plane = DensePlane::new();
                for start in 0..tokens.len().max(1) {
                    let end = (start + 3).min(tokens.len());
                    if start == end {
                        break;
                    }
                    let mut gram = Vec::new();
                    for (i, token) in tokens[start..end].iter().enumerate() {
                        if i > 0 {
                            gram.push(0u8);
                        }
                        gram.extend_from_slice(token);
                    }
                    plane.add_hash(fpga_hllset::murmur3_low64(&gram, 0));
                }
                Ok(Response::Plane {
                    id: self.alloc(plane),
                })
            }
            Command::Algebra { op, a, b } => {
                let pa = self.plane(a)?.clone();
                let pb = self.plane(b)?.clone();
                let result = match op {
                    AlgebraOp::Or => pa.or(&pb),
                    AlgebraOp::And => pa.and(&pb),
                    AlgebraOp::AndNot => pa.and_not(&pb),
                    AlgebraOp::Xor => pa.xor(&pb),
                };
                Ok(Response::Plane {
                    id: self.alloc(result),
                })
            }
            Command::Drn { prev, curr } => {
                let pp = self.plane(prev)?.clone();
                let pc = self.plane(curr)?.clone();
                let departed = self.alloc(pp.and_not(&pc));
                let retained = self.alloc(pp.and(&pc));
                let new = self.alloc(pc.and_not(&pp));
                Ok(Response::Drn {
                    departed,
                    retained,
                    new,
                })
            }
            Command::Popcount { plane } => {
                let value = self.plane(plane)?.popcount();
                Ok(Response::Popcount { value })
            }
            Command::Hist { plane } => {
                let counts = self.plane(plane)?.bit_counts();
                Ok(Response::Hist { counts })
            }
            Command::TfAdd { plane, delta } => {
                let positions: Vec<u32> = self
                    .plane(plane)?
                    .active_positions()
                    .into_iter()
                    .map(|(reg, tz)| reg * 32 + tz)
                    .collect();
                let lanes = self.tf.entry(plane).or_insert_with(|| vec![0i64; TF_LANES]);
                for pos in positions {
                    lanes[pos as usize] += delta;
                }
                Ok(Response::Ack)
            }
            Command::TfMerge { a, b } => {
                let bl = self.tf.entry(b).or_insert_with(|| vec![0i64; TF_LANES]).clone();
                let al = self.tf.entry(a).or_insert_with(|| vec![0i64; TF_LANES]);
                for i in 0..TF_LANES {
                    al[i] = al[i].max(bl[i]);
                }
                Ok(Response::Ack)
            }
            Command::Lookup { positions } => {
                let mut tokens = Vec::new();
                for pos in positions {
                    if let Some(candidates) = self.lut.get(&pos) {
                        tokens.extend(candidates.iter().cloned());
                    }
                }
                Ok(Response::Candidates { tokens })
            }
            Command::Forth { .. } => Err(FpgaError::Unsupported("Forth core".into())),
            Command::Configure { spec } => {
                self.spec = Some(spec);
                Ok(Response::Ack)
            }
            Command::Status => Ok(Response::Status {
                counters: vec![
                    ("submitted".into(), self.submitted),
                    ("resident_planes".into(), self.planes.len() as u64),
                    ("tf_accumulators".into(), self.tf.len() as u64),
                    ("lut_entries".into(), self.lut.len() as u64),
                    ("configured".into(), self.spec.is_some() as u64),
                ],
            }),
        }
    }
}

impl FpgaDriver for SimFpgaDriver {
    fn submit(&mut self, cmd: Command) -> FpgaResult<()> {
        self.submitted += 1;
        let response = self.exec(cmd)?;
        self.responses.push_back(response);
        Ok(())
    }

    fn drain(&mut self) -> FpgaResult<Vec<Response>> {
        Ok(self.responses.drain(..).collect())
    }

    fn status(&mut self) -> FpgaResult<Vec<(String, u64)>> {
        Ok(vec![
            ("submitted".to_string(), self.submitted),
            ("resident_planes".to_string(), self.planes.len() as u64),
        ])
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// HLLSet-level FPGA delegation wrapper.
///
/// Every method submits real `fpga-hostif` commands and drains responses, so
/// this is the same code path a physical board would see. Upload/download are
/// simulator bootstrap helpers; processing runs in the golden model.
#[derive(Debug)]
pub struct FpgaSim {
    driver: SimFpgaDriver,
}

impl FpgaSim {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            driver: SimFpgaDriver::new(name),
        }
    }

    /// The underlying driver (for direct protocol access).
    pub fn driver(&mut self) -> &mut SimFpgaDriver {
        &mut self.driver
    }

    fn one(&mut self) -> FpgaResult<Response> {
        let mut responses = self.driver.drain()?;
        responses
            .pop()
            .ok_or_else(|| FpgaError::Protocol("no response from FPGA".into()))
    }

    fn plane_from(&mut self, response: Response) -> FpgaResult<HLLSet> {
        match response {
            Response::Plane { id } => self.driver.download(id),
            other => Err(FpgaError::Protocol(format!(
                "expected plane response, got {other:?}"
            ))),
        }
    }

    /// Hash `tokens` with all `seeds` on the FPGA and return the plane.
    pub fn ingest(&mut self, tokens: &[Vec<u8>], seeds: &[u64]) -> FpgaResult<HLLSet> {
        self.driver.submit(Command::Ingest {
            tokens: tokens.to_vec(),
            seeds: seeds.to_vec(),
        })?;
        {
            let resp = self.one()?;
            self.plane_from(resp)
        }
    }

    /// 3-gram structural fingerprint plane.
    pub fn gram(&mut self, tokens: &[Vec<u8>]) -> FpgaResult<HLLSet> {
        self.driver
            .submit(Command::Gram { tokens: tokens.to_vec() })?;
        {
            let resp = self.one()?;
            self.plane_from(resp)
        }
    }

    /// `a ∪ b` on the FPGA.
    pub fn union(&mut self, a: &HLLSet, b: &HLLSet) -> FpgaResult<HLLSet> {
        self.algebra(AlgebraOp::Or, a, b)
    }

    /// `a ∩ b` on the FPGA.
    pub fn intersection(&mut self, a: &HLLSet, b: &HLLSet) -> FpgaResult<HLLSet> {
        self.algebra(AlgebraOp::And, a, b)
    }

    /// `a \ b` on the FPGA.
    pub fn difference(&mut self, a: &HLLSet, b: &HLLSet) -> FpgaResult<HLLSet> {
        self.algebra(AlgebraOp::AndNot, a, b)
    }

    /// `a Δ b` on the FPGA.
    pub fn symmetric_difference(&mut self, a: &HLLSet, b: &HLLSet) -> FpgaResult<HLLSet> {
        self.algebra(AlgebraOp::Xor, a, b)
    }

    fn algebra(&mut self, op: AlgebraOp, a: &HLLSet, b: &HLLSet) -> FpgaResult<HLLSet> {
        let pa = self.driver.upload(a);
        let pb = self.driver.upload(b);
        self.driver.submit(Command::Algebra { op, a: pa, b: pb })?;
        {
            let resp = self.one()?;
            self.plane_from(resp)
        }
    }

    /// `D/R/N` decomposition of `prev` → `curr` on the FPGA.
    pub fn drn(
        &mut self,
        prev: &HLLSet,
        curr: &HLLSet,
    ) -> FpgaResult<(HLLSet, HLLSet, HLLSet)> {
        let pp = self.driver.upload(prev);
        let pc = self.driver.upload(curr);
        self.driver
            .submit(Command::Drn { prev: pp, curr: pc })?;
        match self.one()? {
            Response::Drn {
                departed,
                retained,
                new,
            } => Ok((
                self.driver.download(departed)?,
                self.driver.download(retained)?,
                self.driver.download(new)?,
            )),
            other => Err(FpgaError::Protocol(format!(
                "expected drn response, got {other:?}"
            ))),
        }
    }

    /// Popcount on the FPGA.
    pub fn popcount(&mut self, hllset: &HLLSet) -> FpgaResult<u32> {
        let plane = self.driver.upload(hllset);
        self.driver.submit(Command::Popcount { plane })?;
        match self.one()? {
            Response::Popcount { value } => Ok(value),
            other => Err(FpgaError::Protocol(format!(
                "expected popcount response, got {other:?}"
            ))),
        }
    }

    /// 32-lane HT histogram on the FPGA.
    pub fn hist(&mut self, hllset: &HLLSet) -> FpgaResult<[u32; 32]> {
        let plane = self.driver.upload(hllset);
        self.driver.submit(Command::Hist { plane })?;
        match self.one()? {
            Response::Hist { counts } => Ok(counts),
            other => Err(FpgaError::Protocol(format!(
                "expected hist response, got {other:?}"
            ))),
        }
    }

    /// TF accumulation on the FPGA: add `delta` to every set bit of `hllset`.
    pub fn tf_add(&mut self, hllset: &HLLSet, delta: i64) -> FpgaResult<()> {
        let plane = self.driver.upload(hllset);
        self.driver.submit(Command::TfAdd { plane, delta })?;
        match self.one()? {
            Response::Ack => Ok(()),
            other => Err(FpgaError::Protocol(format!(
                "expected ack, got {other:?}"
            ))),
        }
    }

    /// Status counters from the FPGA.
    pub fn status(&mut self) -> FpgaResult<Vec<(String, u64)>> {
        self.driver.status()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hll(tokens: &[&str]) -> HLLSet {
        HLLSet::from_tokens(tokens)
    }

    #[test]
    fn algebra_crosschecks_against_host() {
        let mut sim = FpgaSim::new("crosscheck");
        let a = hll(&["alpha", "beta", "gamma"]);
        let b = hll(&["beta", "gamma", "delta"]);

        assert_eq!(
            sim.union(&a, &b).unwrap().popcount(),
            a.union(&b).popcount()
        );
        assert_eq!(
            sim.intersection(&a, &b).unwrap().popcount(),
            a.intersection(&b).popcount()
        );
        assert_eq!(
            sim.difference(&a, &b).unwrap().popcount(),
            a.difference(&b).popcount()
        );
        assert_eq!(
            sim.symmetric_difference(&a, &b).unwrap().popcount(),
            a.symmetric_difference(&b).popcount()
        );
    }

    #[test]
    fn drn_decomposes_exactly() {
        let mut sim = FpgaSim::new("drn");
        let prev = hll(&["a", "b", "c"]);
        let curr = hll(&["b", "c", "d"]);
        let (departed, retained, new) = sim.drn(&prev, &curr).unwrap();
        assert_eq!(departed.popcount(), prev.difference(&curr).popcount());
        assert_eq!(retained.popcount(), prev.intersection(&curr).popcount());
        assert_eq!(new.popcount(), curr.difference(&prev).popcount());
    }

    #[test]
    fn ingest_and_popcount_match_host() {
        let mut sim = FpgaSim::new("ingest");
        let tokens: Vec<Vec<u8>> = ["x", "y", "z"].iter().map(|t| t.as_bytes().to_vec()).collect();
        let plane = sim.ingest(&tokens, &[0]).unwrap();
        assert_eq!(plane.popcount(), hll(&["x", "y", "z"]).popcount());
        assert_eq!(sim.popcount(&plane).unwrap() as u64, plane.popcount());
    }

    #[test]
    fn multi_seed_ingest_matches_host_hllsets() {
        let mut sim = FpgaSim::new("multi-seed");
        let tokens: Vec<Vec<u8>> = ["m", "n"].iter().map(|t| t.as_bytes().to_vec()).collect();

        let g1 = sim.ingest(&tokens, &[0]).unwrap();
        let g2 = sim.ingest(&tokens, &[1]).unwrap();
        let g3 = sim.ingest(&tokens, &[2]).unwrap();
        let all = sim.ingest(&tokens, &[0, 1, 2]).unwrap();

        // Independent channels are different planes; the union of the three
        // single-seed planes equals the multi-seed plane.
        let host_g123 = g1.union(&g2).union(&g3);
        assert_eq!(all.popcount(), host_g123.popcount());
    }

    #[test]
    fn tf_add_and_status_report() {
        let mut sim = FpgaSim::new("tf");
        let plane = hll(&["tf", "test"]);
        sim.tf_add(&plane, 3).unwrap();
        let status = sim.status().unwrap();
        assert!(status.iter().any(|(k, v)| k == "resident_planes" && *v >= 1));
    }
}
