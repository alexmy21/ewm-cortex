# Reproduction Transformer with HLLSet LUT-based K-Storage — Architecture

> **Status:** Proposed design — implementation roadmap (ewm-cortex workspace).
> **Scope:** A basic reproduction transformer LLM in which the K side of
> attention is **driven by HLLSet K-storage** (`TokenLUT` / `CatalogLUT`),
> while Q, V, the feed-forward network, and the output head remain
> **traditional matrix machinery in the token realm**.
> **Companions:** [`HLLSET_K_SPACE_MATH.md`](HLLSET_K_SPACE_MATH.md)
> (theory, same directory). Canonical framework docs:
> [`CONVERSATION_CONTEXT_LLM.md`](../../hllset-next/_DOCS/arch/CONVERSATION_CONTEXT_LLM.md)
> (context sub-lattice design), [`STANDARD.md`](../../hllset-next/_DOCS/dev/STANDARD.md)
> (platform specification).

---

## 1. Goal and principles

**Goal.** Prove that the HLLSet partition — the `<reg, tz>` bit cells and
the LUTs that store them — can serve as the K-representation of an LLM
attention mechanism, by building a small transformer that reproduces
standard next-token behavior while its routing structure is anchored in
HLLSet K-storage.

**Principles (from the design session):**

1. **IICA everywhere.** The morphism chain
   `{tokens} → ingestion → hashing → HLLSet → materialization → {tokens}`
   is deterministic, idempotent, immutable, and content-addressed.
2. **Two realms, strictly separated.**
   - *Token realm*: embeddings, Q/V projections, FFN, softmax — the LLM's
     matrix machinery.
   - *HLLSet realm*: the 32K-bit lattice, LUTs, CIDs — relations and K.
   - The bridge is the adjunction pair `ingest (I) ⊣ materialize (M)`.
3. **Context is a sub-lattice.** The LLM context is a bit-closed subset of
   the 32K lattice; restriction is a lattice homomorphism, so local work
   never breaks global consistency.
4. **Coverage invariant.** `LUT coverage ⊇ active bits of the managed
   sub-lattice`; verified by `confidence = 1.0`.
5. **Precision is tiered.** Single-seed collision groups (tier 1) →
   multi-seed quorum (tier 2) → structural 2-/3-gram disambiguation
   (tier 3).

---

## 2. System overview

```text
╔════════════════════════════ TOKEN REALM (traditional LLM) ═════════════════════════════════╗
║                                                                                            ║
║  tokens ──▶ Embedding E ──▶ Transformer blocks ──▶ LM head ──▶ next-token distribution ║
║                 │                   ▲                                                      ║
║                 │                   │ Q (learned)  V (learned)  FFN (learned)              ║
║                 │                   │                                                      ║
║                 └──── K-bridge ─────┘                                                      ║
║                       (bit(t) via TokenLUT.forward)                                        ║
╚═══════════════════════════════════════╤════════════════════════════════════════════════════╝
                                        │  I (ingest)              M (materialize)
╔═══════════════════════════════════════╧═════════════════════════════════════════════════════╗
║                              HLLSET REALM (K-storage)                                       ║
║                                                                                             ║
║  HLLSet 32K-bit lattice  (join = OR, meet = AND, CIDs = SHA-1)                              ║
║   ├─ TokenLUT     : forward token→(reg,tz) ; reverse (reg,tz)→[tokens]   (ordered streams)  ║
║   ├─ CatalogLUT   : forward value→3 positions ; reverse + 2-of-3 quorum (unordered streams) ║
║   ├─ DenseLUT     : 1024×32 array — key IS the address (FPGA path)                          ║
║   ├─ Engines      : InMemory / DuckDB (register-range chunks) / FPGASim + Registry          ║
║   └─ Context      : ConversationContext sub-lattice (top.hll_1/2/3, matrix, TokenIndex)     ║
║                                                                                             ║
║  K-storage interface: key_of(token) → KeyRef ; candidates(hllset) → tokens ;                ║
║                       confidence(hllset) → [0,1]                                            ║
╚═════════════════════════════════════════════════════════════════════════════════════════════╝
```

The transformer never hashes tokens inside the model graph. It asks the
K-bridge for a key; the bridge consults the LUT and returns either an
address (tier 1) or a materialized candidate set (tier 2/3).

---

## 3. Token realm — traditional transformer machinery

Keep the standard small-transformer stack unchanged (the "reproduction"
baseline):

```text
config:
  vocab          = sub-vocabulary of the corpus (e.g. 2K–50K BPE/word tokens)
  d_model        = 128–256
  n_heads        = 4–8
  n_layers       = 4–6
  context_len    = 128–256
  attention      = standard scaled dot-product (with the K-bridge, §5)
  ffn            = 2-layer MLP, GELU
  norm           = LayerNorm / RMSNorm
  head           = tied or untied softmax
```

Responsibilities that **stay in the token realm**:

- `Q = x W_Q`, `V = x W_V` — learned, contextual.
- FFN, norms, positional encoding, output head — learned.
- Token generation (sampling/argmax) and prompt handling.

The token realm owns **all differentiable parameters except the K path**.
This is the cleanest way to test the hypothesis: if the model can reproduce
baseline behavior with K anchored in the LUTs, the HLLSet partition is a
valid K-representation.

---

## 4. HLLSet realm — K-storage

Reuse existing crates without modification:

| Component | Crate | Role |
| ----------- | ------- | ------ |
| `HLLSet`, hashing, `token_to_position` | `hllset-core` | partition assignment |
| `TokenLUT`, `DenseLUT` | `hllset-dsl::materialize` | tier-1 K-storage (ordered) |
| `CatalogLUT`, `materialize_homogeneous_consensus` | `hllset-dsl::materialize` | tier-2 K-storage (unordered) |
| `MaterializeEngine`, `MaterializeRegistry` | `hllset-materialize` | pluggable engines |
| `ConversationContext`, `SparseAdjacencyMatrix`, `TokenIndex` | `hllset-context` | context sub-lattice |

**K-storage interface** (new thin trait, implemented by thin wrappers over
the existing LUTs):

```rust
/// A key reference into the HLLSet realm.
pub enum KeyRef {
    /// One bit cell: reg * 32 + tz (tier 1).
    Cell(u32),
    /// Multi-seed cell set + quorum rule (tier 2).
    Cells(Vec<u32>, usize /* min seeds */),
}

pub trait KStorage {
    /// The key(s) a token maps to under this storage.
    fn key_of(&self, token: &[u8]) -> KeyRef;
    /// Materialized candidates for a sub-lattice HLLSet (the V side).
    fn candidates(&self, hllset: &HLLSet) -> Vec<Vec<u8>>;
    /// Coverage gauge: resolved_bits / total_bits of `hllset`.
    fn confidence(&self, hllset: &HLLSet) -> f64;
}
```

`TokenLUT` implements `key_of` via `forward`; `CatalogLUT` via its
multi-seed `forward`; both implement `candidates`/`confidence` via the
existing materialization strategies.

**Coverage invariant (runtime-checked):**

```text
for the managed context sub-lattice C:
    KStorage::confidence(C) == 1.0
```

---

## 5. The K-bridge — how K enters attention

Three modes, ordered by implementation risk. Mode A is the first target.

### Mode A — Address-key (token-level, tier 1)

```text
bit(t)  = 32 * reg(t) + tz(t)              (via TokenLUT.forward)
k_t     = E_bit[ bit(t) ]                  (learned 32,768 × d table)
```

- `E_bit` is a learned lift of the HLLSet partition: **one key vector per
  bit cell**. Tokens in the same cell share the same key, so attention can
  route only to cells; `V` and context disambiguate within a cell.
- `Q`, `V`, FFN are standard and contextual. The model trains end-to-end;
  hashing happens outside the graph.
- Expected behavior: for a corpus vocabulary of a few thousand tokens,
  most cells are singletons or small groups (§4 of the math notes), so the
  address-key attention is a faithful reproduction of token-level routing.
- Variant (pure LSH, no learning): `k_t = one-hot(bit(t)) W_K` with `W_K`
  fixed random — the Reformer-style bucketed attention; useful as an
  ablation and for the FPGA path.

### Mode B — Set-key (context-level, tier 2/3)

```text
Q_hll = I(prefix tokens)                    (ingest the query)
K_hll = context sub-lattice C               (stored, content-addressed)
score = BSS inclusion τ(Q_hll ⊆ K_hll)      (or Jaccard / overlap popcount)
V     = M(K_hll ∩ Q_hll)                     (materialized candidate tokens)
```

Attention becomes lattice retrieval: no dot product over learned vectors,
but a set morphism plus LUT extraction. This is the HLLSet-native attention
already exercised by CAAL-LLM and `hllset-context::suggest_continuations`.

### Mode C — Hybrid (reproduction with HLLSet bias)

```text
score(i,j) = (q_i · k_j) / √d  +  λ · φ(Q_hll, K_hll)
```

where `φ` is a lattice similarity over the context sub-lattice and `λ` is
learned or scheduled. The learned dot-product attention remains the
workhorse; the HLLSet term injects content-addressed bias. This mode keeps
the reproduction guarantee while making the HLLSet contribution measurable
by ablation (`λ = 0` vs `λ > 0`).

---

## 6. Implementation phases

### Phase 0 — Reproduction baseline (token realm only)

- Stand up a standard small transformer on a fixed corpus (start with the
  CAAL driving-rules corpus, then a small text corpus).
- Metric: next-token perplexity / accuracy on a held-out set.
- No HLLSet involvement. Freeze this as the reference baseline.

**Exit criteria:** baseline trains to a stable loss; eval harness works.

### Phase 1 — K-storage attach

- Build `TokenLUT` (and `CatalogLUT` for the unordered/catalog slice of the
  corpus) for the corpus vocabulary.
- Add the `KStorage` trait and wrappers.
- Instrument every training token with `bit(t)`; assert the coverage
  invariant (`confidence == 1.0`) on the corpus sub-lattice.
- Record the collision statistics (expected ≈ `C(n,2)/3072`) and compare
  with the observed LUT group sizes.

**Exit criteria:** `key_of`/`candidates`/`confidence` pass for 100% of the
vocabulary; collision statistics match the theory within sampling error.

### Phase 2 — Address-key attention (Mode A)

- Add `E_bit` (32,768 × d) and the K-bridge lookup.
- Replace the learned `W_K` path with `k_t = E_bit[bit(t)]` (plus
  positional encoding as in the baseline).
- Train to convergence; compare with Phase 0 baseline.

**Exit criteria:** Mode A reaches within a small margin (e.g. ≤ 5% relative
perplexity gap) of the Phase 0 baseline on the same corpus. A smaller gap
is a direct measurement of "the HLLSet partition serves as K."

### Phase 3 — Set-key and hybrid (Modes B, C)

The context side follows the formal definitions of
`HLLSET_K_SPACE_MATH.md` §8:

- **Candidates** — original observation HLLSets from recent history
  (temporal pyramid used as retrieval index only).
- **MoE** — convolve the candidates after an IICA shuffle by SHA1:
  `π = sort_by_sha1(Sᵢ)`, `E_k = ∪ frame_k` (width `w`, stride `s`).
- **ETT / EL** — rank experts by pure BSS coverage of the recent
  observation `R(t)`: `ρ(E) = BSSτ(E, R(t))`; `EL = argmax`.
- **Resolution (A+B)** — `F(t) = EL ∪ { E_k : ρ_k ≥ τ_min }` (bitmap),
  then materialize by collection intersection (TF only on collisions)
  and rank the output so the leader dominates.

Implementation:

- Add `hllset-attn::moe` (`Expert`, `MoE`, `ETT`, `expert_leader`,
  `resolve`) with the SHA1 shuffle.
- Implement `φ` (BSS/Jaccard) over `F(t)` and the hybrid score
  `q·k + λ·φ(Q_hll, F(t))`.
- Harness (`bin/phase3`): build a conversation history of sentence
  HLLSets, compute MoE/ETT/EL/resolution, materialize `F(t)`, and blend
  transformer next-token logits with BSS-ranked candidates.
- Run ablations: `λ = 0` vs `λ > 0`; pure Mode B (retrieval) vs Mode A.

**Exit criteria:** the hybrid ranking matches or exceeds Mode A on
next-token prediction, and Mode B alone achieves non-trivial retrieval
accuracy (reproduction of CAAL results).

### Phase 4 — Multi-seed, backends, persistence

- Route unordered/catalog inputs through `CatalogLUT` consensus (tier 2).
- Exercise `DuckDBEngine` (register-range chunked LUT) and `FPGASimEngine`
  through `MaterializeRegistry`; measure lookup latency.
- Persist the context sub-lattice as CIDs (`o:/h:/r:/d:/n:` taxonomy) and
  verify sub-lattice extraction from a persisted LUT.

**Exit criteria:** all three precision tiers demonstrable; sub-lattice
extraction from persisted storage is bit-exact (tier 2/3) or
collision-group-exact (tier 1) with `confidence == 1.0`.

---

## 7. Module layout (proposed)

```text
crates/hllset-attn/                     (new crate)
  Cargo.toml
  src/
    lib.rs            // re-exports
    kstorage.rs       // KeyRef, KStorage trait, TokenLUT/CatalogLUT wrappers
    bit_table.rs      // E_bit: 32,768 × d learned key table (Mode A)
    bridge.rs         // token -> KeyRef -> key vector (Mode A/C plumbing)
    setkey.rs         // HLLSet-native attention scores: BSS, Jaccard, overlap (Mode B)
    hybrid.rs         // combined score with λ gate (Mode C)
  tests/
    kstorage_roundtrip.rs     // key_of/candidates/confidence on corpus LUTs
    collision_stats.rs        // observed vs C(n,2)/3072
    mode_a_reproduction.rs    // Phase 2 acceptance test
    mode_b_retrieval.rs       // CAAL-style retrieval test
```

Dependencies: `hllset-core`, `hllset-dsl` (LUTs + strategies),
`hllset-materialize` (engines), `hllset-context` (sub-lattice), plus a
minimal tensor backend for the token-realm transformer (e.g. `candle` or a
hand-rolled f32 matmul for the reproduction scale).

---

## 8. Invariants and test plan

| # | Invariant | Check |
| --- | ----------- | ------- |
| 1 | Coverage: LUT ⊇ active bits of managed sub-lattice | `confidence(C) == 1.0` on every context |
| 2 | Partition: every token has exactly one `(reg, tz)` | `TokenLUT.forward` total and deterministic |
| 3 | Collision rate matches theory | observed group sizes ≈ `C(n,2)/3072` |
| 4 | Multi-seed quorum rejects weak matches | existing `test_consensus_rejects_weak_matches` |
| 5 | Round-trip composition law | `materialize → ingest` per-layer bit-exactness (`roundtrip.rs`) |
| 6 | Matrix index is vocabulary sub-lattice | `matrix_index_is_vocabulary()` |
| 7 | Sub-lattice extraction is local | extraction touches only bits of `C` (instrumented) |
| 8 | Reproduction gap | Mode A perplexity within margin of Phase 0 baseline |

---

## 9. Risks and mitigations

| Risk | Mitigation |
| ------ | ------------ |
| Address-key resolution too coarse for exact-token routing | `E_bit` gives one vector per cell; V and context disambiguate; multi-seed (`CatalogLUT`) restores exactness where needed |
| Static `k_t = E_bit[bit(t)]` loses contextuality of K | add positional encoding (baseline already has it); Mode C hybrid restores learned contextual K |
| Learned `E_bit` could "cheat" by re-deriving token identity | acceptable: the hypothesis is about the partition's sufficiency, not about freezing vectors; add Mode A pure-LSH variant as control |
| `tz` precision is semantics-blind | accepted by design (math notes §4); frequency-aware remapping is a future refinement, not a blocker |
| Small-corpus overfitting hides the routing question | hold-out eval; collision-group ablation (merge cells, measure loss delta) |

---

## 10. Open questions

1. **Tensor backend.** Reproduction scale is small (d ≤ 256, ≤ 6 layers);
   hand-rolled matmul is feasible and keeps the stack Rust-only. At what
   scale does a real backend (`candle`) become necessary?
2. **K-table size.** `E_bit` is 32,768 × d — larger than the token
   embedding for a 2K vocabulary. Is a low-rank factorisation
   (`E_bit = U_bit U_shared`) acceptable without weakening the result?
3. **Where does frequency live?** The lattice is multiplicity-free; the
   matrix carries counts. Should the K-bridge ever read `matrix` counts,
   or stay purely on the bit level?
4. **Sub-lattice selection policy.** Which bits form the managed context
   sub-lattice at inference time — the last `k` exchanges, the top union,
   or a BSS-relevant projection?
