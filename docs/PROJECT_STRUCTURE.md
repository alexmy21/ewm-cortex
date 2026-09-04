# ewm-cortex — Project Structure

> **Version 0.2.0 — intentional compatibility break.**
> ewm-cortex is now **self-contained**: it vendors the HLLSet algebra
> (`hllset-core`) and the materialization layer (`hllset-materialize`) as
> first-party crates and no longer depends on `hllset-next`, `EWM`, or
> `ewm-fpga-bridge` by path. Those projects remain as historical upstreams;
> old and new versions coexist.

## Workspace crates (dependency order, bottom-up)

```text
hllset-core            vendored algebra: HLLSet, hashing, operations,
                       content-addressing, serialization, TFVec
        ▲
        ├── hllset-materialize   ingest.rs + materialize.rs: TokenLUT / CatalogLUT,
        │                        consensus, streaming Ingestor
        │                        + MaterializeEngine trait
        │
        ├── hllset-attn          K-storage (KStorage trait, TokenLutStorage,
        │                        CatalogLutStorage), TokenMask,
        │                        ContextVocabulary, MoE/ETT, setkey, hybrid
        │
        ├── ewm-git              2005-style Git: commit DAG, H(t) view,
        │                        merge, GC, archive-before-prune,
        │                        G1/G2/G3 commit-linked channels + time travel,
        │                        HllsetLut (<SHA1,TH>) + BitTf (32K TF vector)
        │
        └── cortex-core          black-box pipeline: tokens → hash →
                                 tokenLUT → HLLSet → materialize →
                                 gate_TF → restored → decoder
                 ▲
                 └── hllset-repro  POC transformer + Phase 0–3 harnesses
```

Dependency matrix (only these edges are allowed):

| Crate | Depends on |
|---|---|
| `hllset-core` | crates.io only (murmur3, sha1, hex, roaring, serde, serde_json) |
| `hllset-materialize` | `hllset-core`; `fpga-hostif`+`fpga-hllset` (optional, feature `fpga-sim`) |
| `hllset-attn` | `hllset-core`, `hllset-materialize` |
| `ewm-git` | `hllset-core`, `hllset-materialize` |
| `cortex-core` | `hllset-core`, `hllset-attn` |
| `hllset-repro` | `hllset-core`, `hllset-attn` |

Rules:

1. **No path dependencies outside this workspace.** Everything under
   `crates/` is first-party and may be changed freely.
2. **No cycles.** The graph is a DAG: algebra → storage/attention/evolution
   → pipeline → POC harnesses.
3. **`hllset-repro` is disposable.** It is the POC/benchmark surface; the
   production path is `cortex-core` (pipeline) + `ewm-git` (evolution) +
   `hllset-attn` (context/MoE).
4. **`hllset-core` is now ours.** It was vendored at v0.2.0 from
   `hllset-next`; from this point it evolves independently and may diverge.

## What moved (0.1.0 → 0.2.0)

| Before | After |
|---|---|
| `hllset-core` path-dep on `../hllset-next/crates/hllset-core` | `crates/hllset-core` (vendored, first-party) |
| `hllset-dsl` path-dep (only `materialize` was used) | `crates/hllset-materialize` (extracted module + trait) |
| docs pointing at hllset-next/EWM/ewm-fpga-bridge as authorities | those projects are historical upstreams only |

## Removed compatibility surface

- The former `hllset-dsl` dependency is gone; `hllset-attn` now imports
  `hllset_materialize::{…}`.
- The notebooks' `:dep` paths now point inside this workspace.
- Workspace version bumped to **0.2.0** to mark the break; 0.1.x and the
  old hllset-next-based versions remain available in their own checkouts.

## Future backends (trait sockets, not yet wired)

- `MaterializeEngine` — implemented by future DuckDB / FPGA-sim / physical
  engines; must be bit-exact against the in-memory reference.
- `ObjectStore` (ewm-git) — implemented by `MemoryStore`, `LooseStore`;
  the IPFS archive adapter (`IpfrsNativeStorage`) is the next backend
  (milestone G3).
- `KStorage` (hllset-attn) — implemented by `TokenLutStorage`,
  `CatalogLutStorage`; future `DenseLUT`/hardware backends plug here.
