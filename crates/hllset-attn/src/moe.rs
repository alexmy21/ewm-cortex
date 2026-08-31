//! MoE / Expert Think Tank — the formal definitions of
//! `HLLSET_K_SPACE_MATH.md` §8.
//!
//! ```text
//! π    = sort_by_sha1( { Sᵢ : recent } )      // IICA shuffle, time-unbiased
//! F_k  = { S_π(i) : i ∈ [k·s, k·s + w) }      // convolution frame
//! E_k  = ∪ F_k                                 // Expert
//! MoE  = { E_k }
//! ρ(E) = BSSτ(E, R(t))                         // relevance to recent observation
//! ETT  = { (E_k, ρ_k) } sorted by ρ desc
//! EL   = argmax ρ
//! F(t) = EL ∪ { E_k : ρ_k ≥ τ_min }            // resolution A (bitmap)
//! ```
//!
//! The temporal pyramid is used only as a *retrieval index* for the recent
//! candidates; frames are formed over the original observations after a
//! content-address shuffle, so grouping is content-driven, not time-driven.

use hllset_core::core::hashing::sha1_hex;
use hllset_core::HLLSet;

use crate::setkey::bss_coverage;

/// One original observation with its content address.
#[derive(Clone, Debug)]
pub struct Observation {
    pub hllset: HLLSet,
    /// SHA1 of the serialized HLLSet (the `h:` content key material).
    pub cid: String,
}

impl Observation {
    pub fn new(hllset: HLLSet) -> Self {
        let cid = sha1_hex(&hllset.to_bytes());
        Self { hllset, cid }
    }
}

/// One expert: the union of a convolution frame.
#[derive(Clone, Debug)]
pub struct Expert {
    pub hllset: HLLSet,
    /// Content address of the expert union.
    pub cid: String,
    /// Original observation indices that fell into this frame.
    pub frame: Vec<usize>,
}

/// An expert with its relevance to the recent observation.
#[derive(Clone, Debug)]
pub struct RankedExpert {
    pub expert: Expert,
    pub relevance: f64,
}

/// Mixture of Experts — the convolution result over a candidate collection.
#[derive(Clone, Debug, Default)]
pub struct MoE {
    pub experts: Vec<Expert>,
}

impl MoE {
    /// Convolve original observations after an IICA shuffle by content
    /// address. `salt` enables multi-shuffle (independent MoE partitions).
    ///
    /// The temporal pyramid is *not* used for grouping — only the original
    /// HLLSets enter the frames, ordered by SHA1.
    pub fn from_observations(
        observations: &[HLLSet],
        width: usize,
        stride: usize,
        salt: Option<&str>,
    ) -> Self {
        assert!(width > 0, "frame width must be positive");
        assert!(stride > 0, "stride must be positive");

        // IICA shuffle: order by SHA1(content ‖ salt), keeping the original index.
        let mut keyed: Vec<(String, usize, HLLSet)> = observations
            .iter()
            .enumerate()
            .map(|(idx, s)| {
                let cid = match salt {
                    Some(salt) => {
                        let mut bytes = s.to_bytes();
                        bytes.extend_from_slice(salt.as_bytes());
                        sha1_hex(&bytes)
                    }
                    None => sha1_hex(&s.to_bytes()),
                };
                (cid, idx, s.clone())
            })
            .collect();
        keyed.sort_by(|a, b| a.0.cmp(&b.0));

        let mut experts = Vec::new();
        let mut start = 0;
        while start < keyed.len() {
            let end = (start + width).min(keyed.len());
            let frame_obs = &keyed[start..end];
            let mut union = HLLSet::new();
            let mut frame = Vec::with_capacity(end - start);
            for (_, idx, s) in frame_obs {
                union.merge(s);
                frame.push(*idx);
            }
            let cid = sha1_hex(&union.to_bytes());
            experts.push(Expert {
                hllset: union,
                cid,
                frame,
            });
            start += stride;
        }

        Self { experts }
    }

    /// Expert Think Tank: experts ranked by pure BSS coverage of the recent
    /// observation, descending. Ties are left in stable frame order.
    pub fn think_tank(&self, recent: &HLLSet) -> Vec<RankedExpert> {
        let mut ranked: Vec<RankedExpert> = self
            .experts
            .iter()
            .map(|e| RankedExpert {
                relevance: bss_coverage(&e.hllset, recent),
                expert: e.clone(),
            })
            .collect();
        ranked.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked
    }

    /// Expert Leader: the most relevant expert (empty if no experts).
    pub fn expert_leader(&self, recent: &HLLSet) -> Option<Expert> {
        self.think_tank(recent)
            .into_iter()
            .next()
            .map(|r| r.expert)
    }

    /// Resolution A: `F(t) = EL ∪ { E_k : ρ_k ≥ τ_min }`.
    ///
    /// The leader is always included; the remaining experts join above the
    /// threshold. Monotone and IICA by construction.
    pub fn resolve(&self, recent: &HLLSet, tau_min: f64) -> HLLSet {
        let mut f = HLLSet::new();
        for ranked in self.think_tank(recent) {
            if f.is_empty() || ranked.relevance >= tau_min {
                f.merge(&ranked.expert.hllset);
            } else {
                break; // ranked descending — the rest are below threshold
            }
        }
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(tokens: &[&str]) -> HLLSet {
        HLLSet::from_tokens(tokens.iter())
    }

    #[test]
    fn shuffle_is_deterministic() {
        let observations = vec![obs(&["a"]), obs(&["b"]), obs(&["c"]), obs(&["d"])];
        let m1 = MoE::from_observations(&observations, 2, 1, None);
        let m2 = MoE::from_observations(&observations, 2, 1, None);
        let c1: Vec<String> = m1.experts.iter().map(|e| e.cid.clone()).collect();
        let c2: Vec<String> = m2.experts.iter().map(|e| e.cid.clone()).collect();
        assert_eq!(c1, c2, "IICA shuffle must be deterministic");
    }

    #[test]
    fn leader_covers_recent_observation() {
        let observations = vec![
            obs(&["x", "y"]),
            obs(&["a", "b"]),
            obs(&["c", "d"]),
            obs(&["a", "b", "c"]),
        ];
        let recent = obs(&["a", "b", "c"]);
        let moe = MoE::from_observations(&observations, 2, 1, None);
        let leader = moe.expert_leader(&recent).expect("non-empty MoE");
        // The leader must explain a large share of the recent observation.
        let rho = bss_coverage(&leader.hllset, &recent);
        assert!(rho > 0.0);
        // And it must be at least as relevant as any other expert.
        for ranked in moe.think_tank(&recent) {
            assert!(ranked.relevance <= rho + 1e-9);
        }
    }

    #[test]
    fn resolution_contains_leader_and_is_monotone() {
        let observations = vec![
            obs(&["a", "b"]),
            obs(&["b", "c"]),
            obs(&["c", "d"]),
            obs(&["d", "e"]),
        ];
        let recent = obs(&["a", "b"]);
        let moe = MoE::from_observations(&observations, 2, 1, None);
        let leader = moe.expert_leader(&recent).unwrap();
        let resolved = moe.resolve(&recent, 0.5);
        assert!(resolved.popcount() >= leader.hllset.popcount());
        // Every leader bit is in the resolution (EL ⊆ F(t)).
        let inter = leader.hllset.intersection(&resolved);
        assert_eq!(inter.popcount(), leader.hllset.popcount());
    }

    #[test]
    fn multi_shuffle_gives_different_partitions() {
        // 10 singletons: two salts produce the same sorted order with
        // probability 1/10!, so this is deterministic in practice.
        let observations: Vec<HLLSet> = (0..10)
            .map(|i| obs(&[&format!("w{i}")]))
            .collect();
        let m1 = MoE::from_observations(&observations, 2, 1, Some("salt-1"));
        let m2 = MoE::from_observations(&observations, 2, 1, Some("salt-2"));
        let c1: Vec<String> = m1.experts.iter().map(|e| e.cid.clone()).collect();
        let c2: Vec<String> = m2.experts.iter().map(|e| e.cid.clone()).collect();
        assert_ne!(c1, c2, "different salts should shuffle differently");
    }
}
