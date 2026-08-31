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

## Enhanced pipeline

```text
DeepSeek-OCR Encoder                     DeepSeek-OCR Decoder
      │ encoding IDs                          ▲ restored IDs
      ▼                                       │
╔════════════════════════════════════════════════════════════╗
║                    ewm-cortex (Rust workspace)             ║
║                                                            ║
║  cortex-core   (milestone 1 — reference port)              ║
║  encoding IDs → HLLSet → ∩ gate_TF → TF-LUT → materialize  ║  
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
║          hllset-next (canonical, path dep only)            ║
╚════════════════════════════════════════════════════════════╝
```

## Crate layout

```text
crates/
├── cortex-core/      # milestone 1: the black-box pipeline (this doc §2)
│     src/encoding.rs   SimCodec — simulated tid{n} encoder/decoder
│     src/gate.rs       gate_TF sketch gate + exact membership
│     src/lut.rs        TF-LUT — monotonic TF reverse index
│     src/pipeline.rs   CortexPipeline — process(ids) → PipelineResult
│     src/main.rs       cortex-cli demo
├── hllset-attn/      # K-storage, TokenMask, ContextVocabulary, MoE/ETT
└── hllset-repro/     # transformer POC + Phase 0–3 harnesses
```

## Milestones

- [x] **M1 — reference port.** `cortex-core`: ids → ingest → gate_TF →
  TF-LUT → materialize → restored ids, with simulated `tid{n}` codec.
  Exit: round-trip on the conversation corpus, 0 leaks, coverage 1.0.
- [ ] **M2 — MoE/ETT integration.** `cortex-context`: wire the tested
  `hllset-attn::moe` into the pipeline — context sub-lattice, SHA1-shuffle
  convolution, ETT/EL, resolution A+B, hybrid gate on the restored IDs.
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

- `gate_TF` — sketch gate (probabilistic intersection filter);
  `Gate::exact_known` — authoritative membership (0 leak / 0 FN).
- TF accumulates **pre-gate** (monotonic); materialization resolves the
  gated HLLSet **TF-ranked**.
- The LUT is append-only: departed IDs stay resolvable (IICA).

## Verification (current)

```text
cortex-core CLI on corpus/conversation.txt:
  words 712 | vocab 340 | doc bits 324 | gated bits 324
  restored 340 | leaks 0 ✓ | coverage 1.00
```

Workspace tests: cortex-core 10, hllset-attn 40, hllset-repro 24 — all green.
