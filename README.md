# ewm-cortex

Upgrading hllset-cortex from DeepSeek-OCR fork to support ewm-fpga-bridge.

## Current work: HLLSet as the K-space of attention

Experimental workspace for the **reproduction transformer with HLLSet
LUT-based K-storage**. The canonical HLLSet framework lives in
[`../hllset-next`](../hllset-next) and is consumed here by path dependency —
it is **not modified** from this workspace.

### Layout

```text
ewm-cortex/
├── Cargo.toml                # workspace root
├── crates/
│   ├── hllset-attn/          # K-storage: KStorage trait, TokenLutStorage,
│   │                         # CatalogLutStorage, BitKeyTable, KBridge,
│   │                         # setkey, hybrid, ContextVocabulary, TokenMask
│   └── hllset-repro/         # token realm: hand-rolled autograd, char-level
│                             # transformer, Phase 0/1 harnesses
└── docs/
    ├── HLLSET_K_SPACE_MATH.md                 # theory (partition, adjunction, collisions)
    ├── HLLSET_LUT_TRANSFORMER_ARCHITECTURE.md # implementation roadmap (Phases 0-4)
    └── notebooks/
        └── 14_hllset_attention_demo.ipynb     # interactive demo (evcxr)
```

### Quick start

```bash
# Phase 1 — K-storage attach (coverage + collision statistics)
cargo run -p hllset-repro --bin phase1

# Phase 0 — reproduction transformer baseline (token realm)
cargo run -p hllset-repro --release

# Tests
cargo test -p hllset-attn
cargo test -p hllset-repro
```

### Status

- [x] Phase 0 — reproduction baseline (hand-rolled f32 autograd + transformer)
- [x] Phase 1 — K-storage attach (coverage invariant, collision stats vs 1/3072)
- [x] Phase 2 — address-key attention (`k_t = E_bit[bit(t)]` via `KBridge`; gap +1.2%)
- [ ] Phase 3 — set-key / hybrid attention over the context sub-lattice
- [ ] Phase 4 — multi-seed exactness, backends, persistence
