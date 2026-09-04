# ewm-cortex — Enhanced HLLSet Cortex Architecture

> **Status:** Milestone 1 implemented (`cortex-core`).
> **Reference:** [`hllset_cortex`](https://github.com/alexmy21/DeepSeek-OCR/tree/feature/hllset-cortex-git/hllset_cortex) —
> the DeepSeek-OCR black-box reference implementation.
> **POC foundation:** this workspace's `hllset-attn` and `hllset-repro` crates
> (Phases 0–3, all tested).

## Goal

Reimplement hllset-cortex as an **enhanced** Rust version, adding the MoE /
Expert Think Tank context layer and aligning with the EWM spec
(Emergent Ontology, IICA morphisms, Ashby-Bootstrap, Noether evolution).
DeepSeek-OCR's encoder feeds encoding IDs in; the decoder receives restored
IDs; the cortex never sees real tokens.

## Two spaces, two morphisms (unchanged from the reference)

```text
Token space (encoding IDs tid{n})          HLLSet space (32,768-bit sketches)
        │                                           ▲
        │ ingest (tokens → HLLSet)                  │ materialize (HLLSet → tokens)
        ▼                                           │
```

All structural work happens on HLLSets; every interchange with the LLM /
OCR happens in tokens. The only crossings are `ingest` and `materialize`.

Both morphisms now have dedicated modules in `hllset-materialize`:

- **`ingest.rs`** — the forward morphism. One streaming pass per token:
  1. update all n-gram/seed-n token LUTs (multi-seed `CatalogLUT`);
  2. update the working HLLSets — internal to the loop, never exposed
     until the pass ends;
  3. update TF (per-token TF + the 32K bit-TF vector);
  4. report each original HLLSet to an `IngestSink` so the caller updates
     its HLLSet-LUT (`<SHA1, TH>`).
- **`materialize.rs`** — the backward morphism: collection intersection
  per bit (multi-seed consensus), TF only on collision ties.

`Repository::ingest` wires the two together: ingest-and-commit, with the
HLLSet-LUT updates applied by the sink.

## Enhanced pipeline

```text
DeepSeek-OCR Encoder                     DeepSeek-OCR Decoder
      │ encoding IDs                          ▲ restored IDs
      ▼                                       │
╔════════════════════════════════════════════════════════════╗
║                    ewm-cortex (Rust workspace)             ║
║                                                            ║
║  cortex-core   (milestone 1 — reference port)              ║
║  encoding IDs → hash → tokenLUT → HLLSet → materialize     ║
║    → gate_TF (output) → restored IDs                       ║  
║                                                            ║
║  cortex-context (milestone 2 — the enhancement)            ║
║    Context Sub-Lattice → SHA1-shuffle MoE → ETT → EL       ║
║      → Resolution A+B → F(t) → hybrid gate                 ║
║                                                            ║
║  grounding/search (milestone 3 — from reference)           ║
║    token + structural hallucination diagnostics,           ║
║    page-granular search, precedents                        ║
║                                                            ║
║  reuses: hllset-attn (K-storage, TokenMask, MoE),          ║
║          hllset-repro (transformer POC),                   ║
║          hllset-core + hllset-materialize (first-party)    ║
╚════════════════════════════════════════════════════════════╝
```

## Crate layout

```text
crates/
├── hllset-core/      # vendored algebra: HLLSet, hashing, ops, TFVec
├── hllset-materialize/# TokenLUT / CatalogLUT / consensus + engine trait
│                      #   + fpga.rs (feature `fpga-sim`): FPGA delegation
├── cortex-core/      # milestone 1: the black-box pipeline (this doc §2)
│     src/encoding.rs   SimCodec — simulated tid{n} encoder/decoder
│     src/gate.rs       gate_TF output TokenGate + exact membership
│     src/lut.rs        TF-LUT — monotonic TF reverse index
│     src/pipeline.rs   CortexPipeline — process(ids) → PipelineResult
│     src/main.rs       cortex-cli demo
├── ewm-git/          # milestone E1: 2005-style Git evolution store
│     src/object.rs     ObjectId, Commit, Blob (deterministic serialization)
│     src/store.rs      ObjectStore trait, MemoryStore, LooseStore (+ HEAD)
│     src/repo.rs       Repository — commit, state, merge, log, gc
│     src/view.rs       CommitView — H(t) = (S(t), H(t-1), D, R, N)
│     src/main.rs       ewm-git CLI demo
├── hllset-attn/      # K-storage, TokenMask, ContextVocabulary, MoE/ETT
└── hllset-repro/     # transformer POC + Phase 0–3 harnesses
```

## FPGA delegation (feature `fpga-sim`)

Some HLLSet processings can be delegated to the
`hllset-fpga-simulator` through the `fpga-hostif` protocol, behind the
optional `fpga-sim` feature of `hllset-materialize` (default builds stay
self-contained — the external simulator appears only when the feature is
enabled, exactly like the simulator's own `crosscheck` pattern):

```text
ewm-cortex ── fpga-hostif::{Command, FpgaDriver} ──┬─ SimFpgaDriver (fpga.rs,
                                                    │   golden model)
                                                    └─ future board driver
```

- **`SimFpgaDriver`** executes `Ingest`, `Gram`, `Algebra` (union /
  intersection / difference / xor), `Drn`, `Popcount`, `Hist`, `TfAdd`,
  `TfMerge`, and `Lookup` against the simulator's bit-exact
  `fpga_hllset::DensePlane` golden model.
- **`FpgaSim`** is the HLLSet-level wrapper: every operation submits real
  commands and drains responses, so swapping to a physical driver is a
  driver change, not a host-code change (the substitution rule).
- Plane upload/download is a simulator-side bootstrap (no wire command
  yet); all processing runs through the protocol.
- Crosscheck tests verify bit-exactness against the host HLLSet algebra:
  `cargo test -p hllset-materialize --features fpga-sim`.

The `ewm-fpga-bridge` (`ModuleDriver`: Configure/Feed/Step) is the next
level up — whole module-graph pipelines — and will sit on top of this
socket once pipeline nodes move to the bridge.

## Global channels — G1 / G2 / G3

`G1`, `G2`, `G3` are special prefixes of the current **global** HLLSet
state:

| Channel | Source |
|---|---|
| `G1` | bits mapped from 1-gram or seed-0 |
| `G2` | bits mapped from 2-gram or seed-1 |
| `G3` | bits mapped from 3-gram or seed-2 |

n-grams and seeds are interchangeable — they are just bits, and it does
not matter where they come from. All three channels are **commit-linked**:
one commit carries all three snapshots (`Commit.trees = [G1, G2, G3]`).

Because every commit carries the three channels, any HLLSet can be
projected onto **any previous commit state** — time travel:

```text
project(H, t, GX) = H ∩ GX(t)
```

i.e. "which bits of H already existed in channel GX at commit t". This is
the multi-seed / multi-gram memory: G1, G2, G3 remember the same history
through three independent hashings, and the 2-of-3 quorum across channels
gives exact membership.

## Commit discipline — ingestion only

A commit makes sense only when the lattice gains **new information**:

- **Ingestion is the only automatic commit.** `Repository::ingest` commits
  only if the pass brings bits that are not already covered by the lattice
  top; a no-change pass returns `Ok(None)`.
- **Compound HLLSets do not commit.** A union of committed originals is
  derivable — committing it stores nothing new. `Repository::merge` is a
  pure per-channel lattice join (+ TF merge, + HLLSet-LUT touches) that
  returns the derived `LatticeState` and never updates `HEAD`.
  `merge_commit` exists for callers that explicitly want two-parent
  topology.
- **Context growth → warn, don't commit.** `Repository::context_warning`
  reports when the G1 lattice-top outgrows a threshold and suggests
  compression. The lattice top is maintained as an O(1) per-channel cache.

## Ranking consistency — TF everywhere, no lattice degree

Lattice-degree ranking (HLLSet degrees in an explicit lattice) is dropped:
maintaining the lattice explicitly is computationally and resource
expensive. Instead, **all three rankings are frequency-based**:

| Object | Rank measure | Stored in |
|---|---|---|
| tokens | TF (term frequency) | token/catalog LUT |
| registers (bits) | bit-TF (32K vector) | `BitTf` — snapshot in every commit |
| original HLLSets | TH (touch count) | `HllsetLut` — `<SHA1, TH>` |

- **`HllsetLut`** — `<SHA1, TH>`: how often a given original HLLSet was
  part of a compound HLLSet or used in any other way; **each touch
  counts**. Managed exactly like the token/catalog LUTs (append-only,
  idempotent registration, monotonic TH). Compound HLLSets are never
  stored: the union of all original HLLSets is the lattice **Top**, and
  the original HLLSets are the natural **bottom** (atoms).
- **`BitTf`** — the 32K integer vector holding TF for each bit of the
  lattice top (the union of all original HLLSets). Monotonic CRDT; every
  touch of an HLLSet increments its set bits. Each commit snapshots it
  (`Commit.tf`).
- **Maintenance** — both objects are managed during ingestion and all
  HLLSet operations: commit touches its three channel blobs, merge touches
  the parents' blobs and merges the TF vectors, projection touches the
  queried channel blob.

## Milestones

- [x] **M1 — reference port.** `cortex-core`: tokens → hash →
  tokenLUT → HLLSet → materialize → output `gate_TF` → restored ids,
  with simulated `tid{n}` codec. LUTs are never gated; only the output is.
  Exit: round-trip on the conversation corpus, 0 leaks, coverage 1.0.
- [x] **E1 — evolution store.** `ewm-git` replaces the temporal pyramid:
  commit = timer, commit state = union HLLSet, `H(t) = (S(t), H(t-1),
  D, R, N)`, merge = lattice join, pruning = GC. Memory + loose-file
  stores. Exit: full lifecycle tests over both stores.
- [x] **E2 — archive-before-prune.** `Repository::gc_to(archive)`: doomed
  objects are written to a content-addressed archive (`h:<sha1>` keys)
  before being pruned from the working store — pruned branches stay
  addressable forever. Exit: pruned branch resolvable from the archive.
- [ ] **E3 — IPFS archive adapter.** Implement an IPFS-backed
  `ObjectStore` (modeled on `hllset-storage::IpfrsNativeStorage`:
  ipfrs-core + sled, no daemon) as the archive backend for `gc_to`; the
  `h:<sha1>` key convention is the bridge between the SHA1 object IDs and
  IPFS CIDs.
- [ ] **M2 — MoE/ETT integration.** `cortex-context`: wire the tested
  `hllset-attn::moe` into the pipeline — candidates from the `ewm-git`
  commit DAG, SHA1-shuffle convolution, ETT/EL, resolution A+B, hybrid
  gate on the restored IDs.
- [ ] **M3 — grounding/search.** Port token + structural hallucination
  diagnostics and page-granular search from the reference.
- [ ] **M4 — EWM alignment.** Document/verify the mapping to EWM
  principles (Noether DRN evolution, emergent ontology, holographic
  memory) and the `ewm-fpga-bridge` module DSL (`experts`, `merge`).
- [ ] **M5 — PyO3 bindings.** Expose the pipeline to Python for the real
  DeepSeek-OCR integration.

## The black-box contract (M1)

```rust
let codec = SimCodec::from_text(&text);        // simulated encoder vocab
let mut pipeline = CortexPipeline::new();
pipeline.set_gate(codec.vocab().iter().cloned());
let ids = codec.encode_text(&text);
let result = pipeline.process(&ids);           // black box
let restored = codec.decode_ids(&result.restored_ids);
assert!(result.ok());                          // 0 leaks
```

Semantics preserved from the reference:

- `gate_TF` — the **output TokenGate** (decoder-vocabulary limit);
  `Gate::exact_known` — authoritative membership (0 leak / 0 FN).
- TF accumulates **ungated** (monotonic); materialization resolves the
  **full** HLLSet by collection intersection per bit (TF only on collision
  ties); only the output is filtered.
- The LUT is append-only: departed IDs stay resolvable (IICA).
- The system stays **TokenGate-ready**: any future vocabulary limit in
  DeepSeek-OCR plugs into the same output gate, never into the LUTs.

## Verification (current)

```text
cortex-core CLI on corpus/conversation.txt:
  words 712 | vocab 340 | doc bits 324
  materialized 324 (one token per bit; collisions resolved by TF)
  restored 324 | leaks 0 ✓ | coverage 1.00
```

Workspace tests: cortex-core 11, hllset-attn 40, hllset-repro 25,
hllset-materialize 10, hllset-core 77, ewm-git 26 — all green.
