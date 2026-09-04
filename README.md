# ewm-cortex

Enhanced reimplementation of **hllset-cortex** (the DeepSeek-OCR black-box
reference) in Rust, adding the MoE / Expert Think Tank context layer and
aligning with the EWM spec.

> **v0.2.0 — self-contained (intentional compatibility break).**
> This workspace no longer depends on `hllset-next`, `EWM`, or
> `ewm-fpga-bridge` by path. The HLLSet algebra (`hllset-core`) and the
> materialization layer (`hllset-materialize`) are now **first-party
> crates** and evolve independently. Old versions coexist in their own
> checkouts. See [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md).

See [`docs/CORTEX_ARCHITECTURE.md`](docs/CORTEX_ARCHITECTURE.md) for the
full architecture and milestone plan.

## Pipeline (reference port + enhancement)

```text
DeepSeek-OCR Encoder                     DeepSeek-OCR Decoder
      │ encoding IDs (tid{n})                 ▲ restored IDs
      ▼                                       │
╔══════════════════════════════════════════════════════════╗
║                    ewm-cortex (Rust workspace)           ║
║  cortex-core   : tokens → hash → tokenLUT → HLLSet →      ║
║                  : materialize → gate_TF → restored → decoder║
║  cortex-context: MoE/ETT → EL → Resolution A+B → F(t)    ║
╚══════════════════════════════════════════════════════════╝
```

## Layout

```text
ewm-cortex/
├── Cargo.toml                # workspace root (version 0.2.0)
├── corpus/
│   └── conversation.txt      # corpus for e2e training/testing
├── crates/
│   ├── hllset-core/          # vendored algebra: HLLSet, hashing, ops, TFVec
│   ├── hllset-materialize/   # TokenLUT / CatalogLUT / consensus + engine trait
│   ├── hllset-attn/          # K-storage + MoE/ETT: KStorage, TokenLutStorage,
│   │                         # BitKeyTable, KBridge, setkey, hybrid, moe,
│   │                         # ContextVocabulary, TokenMask
│   ├── ewm-git/              # 2005-style Git evolution store (replaces temporal pyramid)
│   ├── cortex-core/          # black-box pipeline (encoding, gate, TF-LUT, pipeline)
│   └── hllset-repro/         # token realm: hand-rolled autograd, char-level
│                             # transformer, Phase 0-3 harnesses
└── docs/
    ├── PROJECT_STRUCTURE.md                       # crate graph + ownership rules
    ├── CORTEX_ARCHITECTURE.md                     # enhanced hllset-cortex architecture
    ├── HLLSET_K_SPACE_MATH.md                     # theory (partition, adjunction, MoE/ETT §8)
    ├── HLLSET_LUT_TRANSFORMER_ARCHITECTURE.md     # implementation roadmap (Phases 0-4)
    └── notebooks/
        ├── 14_hllset_attention_demo.ipynb         # concepts demo (evcxr)
        └── 15_e2e_training_testing.ipynb          # train/test on unknown text (evcxr)
```

## Quick start

```bash
# M1 — cortex pipeline (tokens → hash → tokenLUT → HLLSet → materialize → gate_TF → restored)
cargo run -p cortex-core                     # uses corpus/conversation.txt
cargo run -p cortex-core -- path/to/text.txt

# G1 — content-addressed evolution store (commit DAG, H(t) view, merge, gc)
cargo run -p ewm-git

# POC harnesses
cargo run -p hllset-repro --bin phase1       # K-storage attach stats
cargo run -p hllset-repro --bin phase2       # address-key vs baseline
cargo run -p hllset-repro --bin phase3       # MoE/ETT + hybrid gate

# Tests
cargo test --workspace
```

## Status

### **Cortex milestones**

- [x] M1 — reference pipeline port (`cortex-core`: hash → tokenLUT → HLLSet → materialize → gate_TF)
- [x] E1 — evolution store (`ewm-git`: commit DAG, H(t) view, merge, GC)
- [x] E2 — archive-before-prune (`gc_to`: pruned branches stay addressable)
- [ ] E3 — IPFS archive adapter (`hllset-storage::IpfrsNativeStorage`)
- [ ] M2 — MoE/ETT integration (`cortex-context`, candidates from the commit DAG)
- [ ] M3 — grounding/search port
- [ ] M4 — EWM alignment + `ewm-fpga-bridge` module DSL
- [ ] M5 — PyO3 bindings for DeepSeek-OCR integration

### **Attention/K-space POC**

- [x] Phase 0 — reproduction baseline (hand-rolled f32 autograd + transformer)
- [x] Phase 1 — K-storage attach (coverage invariant, collision stats vs 1/3072)
- [x] Phase 2 — address-key attention (`k_t = E_bit[bit(t)]` via `KBridge`; gap +1.2%)
- [x] Phase 3 — MoE/ETT resolution + hybrid gate (`hllset-attn::moe`, `bin/phase3`)
- [ ] Phase 4 — multi-seed exactness, backends, persistence
