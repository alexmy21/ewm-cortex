# EWM-CORTEX Architecture & Design Review

**Project:** `ewm-cortex` v0.2.0  
**Workspace:** `/home/alexmy/SGS/SGS_lib/fractal_manifold/ewm-cortex`  
**Review date:** 2026-09-01  
**Scope:** 6 crates, ~9.1 kLOC Rust

---

## Executive Summary

`ewm-cortex` is a well-architected, self-contained Rust workspace that reimplements the `hllset-cortex` black-box reference with a clean layered design. The codebase shows strong mathematical foundations (HLLSet lattice, BSS, content-addressing, MoE/ETT), good test coverage (~189 tests, all green), and a principled crate graph with no external path dependencies.

**Overall grade: B+** — solid architecture with clear separation of concerns, but several production-robustness issues should be addressed before the unfinished milestones (M2–M5, E3) land.

| Metric | Result |
|---|---|
| `cargo test --workspace` | ✅ all pass |
| `cargo clippy --workspace` | ✅ builds, 9 minor warnings |
| Crates | 6 (hllset-core, hllset-materialize, hllset-attn, ewm-git, cortex-core, hllset-repro) |
| Public trait sockets | MaterializeEngine, ObjectStore, KStorage, IngestSink |
| unwrap/expect occurrences | 124 (including tests and binaries) |

---

## 1. Strengths

### 1.1 Clean crate graph

The workspace dependency graph is a strict DAG:

```text
hllset-core
    ├── hllset-materialize
    ├── hllset-attn
    ├── ewm-git
    └── cortex-core ← hllset-attn
            ↑
        hllset-repro
```

This matches the documented layering in `docs/PROJECT_STRUCTURE.md` and makes reasoning about build order, test isolation, and public APIs straightforward.

### 1.2 Self-contained, first-party ownership

Version 0.2.0 intentionally broke compatibility with `hllset-next`, `EWM`, and `ewm-fpga-bridge` path dependencies. `hllset-core` and `hllset-materialize` are now vendored first-party crates, which removes upstream coupling risk and lets the project evolve independently.

### 1.3 Strong core abstractions

- `HLLSet` is a thin, deterministic wrapper over `RoaringBitmap` with explicit serialization.
- Content-addressing prefixes (`h:`, `c:`, `t:`, etc.) are explicit and validated.
- `TFVec`, `BitTf`, and `HllsetLut` implement monotonic CRDT semantics for frequency/touch counts.
- Trait sockets (`MaterializeEngine`, `ObjectStore`, `KStorage`, `IngestSink`) are present for future backends.

### 1.4 Solid evolution store

`ewm-git` is a particular strength: commit DAG, `H(t) = (S(t), H(t-1), D, R, N)` views, merge as lattice join, archive-before-prune GC, and time-travel projection are all implemented and integration-tested.

### 1.5 Good test density

Every crate has unit tests; `ewm-git` has full-lifecycle integration tests; `hllset-repro` has gradient checks for its hand-rolled autograd. The Phase 0–3 harnesses are exposed as both library functions and binaries, which is good for reproducibility.

---

## 2. Critical Issues (address first)

### 2.1 Production panic paths in `hllset-core`

The core algebra crate contains `unwrap`/`expect` calls that can panic in library usage:

| File | Line | Issue |
|---|---|---|
| `crates/hllset-core/src/core/hashing.rs` | 28 | `murmur3_hash_seeded` panics on in-memory `Cursor` failure |
| `crates/hllset-core/src/core/serialization.rs` | 16 | `HLLSet::to_bytes` panics on `serialize_into` failure |
| `crates/hllset-core/src/core/tfvec.rs` | 66 | `TFVec::increment` panics on out-of-range index |
| `crates/hllset-core/src/core/commit.rs` | 81 | `Commit::to_json` uses `expect` |

**Recommendation:** Introduce a `hllset_core::Result<T>` and propagate serialization/hashing errors. Replace `TFVec::increment` panic with `Result` or `checked` semantics, or at minimum document the panic contract on the public API.

### 2.2 O(n²) DAG traversal in `ewm-git`

`Repository::log` and `Repository::reachable_mark` use `Vec<ObjectId>` as a visited set:

- `crates/ewm-git/src/repo.rs:310-321`
- `crates/ewm-git/src/repo.rs:345-361`

`mark.contains(id)` and `seen.contains(&id)` are linear scans. For large commit DAGs this becomes a real bottleneck.

**Recommendation:** Replace `Vec<ObjectId>` visited sets with `HashSet<ObjectId>` or `BTreeSet<ObjectId>`.

### 2.3 `MaterializeEngine` trait signature does not match implementation

`MaterializeEngine::materialize` takes `positions: &[(u16, u8)]` (`crates/hllset-materialize/src/lib.rs:25`), but the only implementer (`Materializer` in `materialize.rs:749`) ignores the argument and recomputes positions internally. The trait also lives in `hllset-materialize`, so future backends must depend on the full materialization crate even if they only want to implement the algebra-side contract.

**Recommendation:**
- Remove the unused `positions` argument from the trait or split the interface into a query plan and an engine.
- Consider moving `MaterializeEngine` to `hllset-core` so backend crates depend only on the algebra crate.

---

## 3. Significant Design Issues

### 3.1 `ObjectId` lacks runtime validation

`ObjectId` is a `String` newtype with no validation at construction. Malformed or non-hex IDs can be created anywhere and will fail later in storage or serialization.

**Recommendation:** Add a validated constructor (e.g., `ObjectId::new_validated`) that rejects non-40-character hex strings, and use it at the boundaries of `ObjectStore` and serialization.

### 3.2 Error cause erasure in `LooseStore`

`LooseStore::get` maps any I/O error to `StoreError::NotFound` (`crates/ewm-git/src/store.rs:143`), which makes debugging permission or disk failures difficult.

**Recommendation:** Preserve the original `std::io::Error` inside `StoreError` variants.

### 3.3 `Repository::gc` silently ignores delete failures

`crates/ewm-git/src/repo.rs:377` discards the result of `self.store.delete(&id)`.

**Recommendation:** Either return delete failures in `GcReport` or propagate them as errors.

### 3.4 Content-addressing validation gap

`content_addr::parse_cid` checks SHA1 length but does not validate that characters are ASCII hex digits, despite the docstring claiming so (`crates/hllset-core/src/core/content_addr.rs:52`). `make_cid` validates; `parse_cid` should reuse that logic.

### 3.5 Wall-clock timestamps break commit determinism

`Commit` in `hllset-core` uses `SystemTime` microsecond timestamps (`crates/hllset-core/src/core/commit.rs:60-63`). This means two commits with identical parent/tree/message content get different IDs, weakening the content-addressing contract.

**Recommendation:** Either use a deterministic counter (e.g., logical clock / Lamport timestamp) or document that commit IDs are not purely content-addressed.

### 3.6 Hard-coded absolute paths in CLIs

`cortex-core/src/main.rs:16-17` and `ewm-git/src/main.rs:36` hard-code `/home/alexmy/SGS/SGS_lib/fractal_manifold/ewm-cortex/corpus/conversation.txt`. These break on any other machine or checkout location.

**Recommendation:** Default to `env::current_dir()` + `corpus/conversation.txt`, or take the corpus path as a CLI argument.

---

## 4. Performance & API Ergonomics

### 4.1 Clone-heavy hot paths

The codebase has ~201 `clone()` calls. Many are inherent to immutable HLLSet operations, but several are avoidable:

| File | Issue |
|---|---|
| `crates/ewm-git/src/hllset_lut.rs:69` | `HllsetLut::ranked` clones every entry; could return references or an iterator |
| `crates/hllset-materialize/src/materialize.rs:88-95, 580-587` | `collect_candidates` clones all candidate tokens per call |
| `crates/hllset-attn/src/bridge.rs:41, 48` | `KBridge::key_vector` clones the `KeyRef` twice |
| `crates/hllset-attn/src/context_vocab.rs` | `tokens` resolves one token at a time, each allocating a `Vec` |

**Recommendation:** Audit clones in materialization, K-storage, and LUT ranking paths; prefer borrowing and iterators where lifetimes permit.

### 4.2 `ObjectStore::put` API redundancy

`ObjectStore::put` takes `(&ObjectId, &Object)`, but `Object::id()` already returns the object's ID. The first argument is redundant and creates a risk of passing the wrong ID.

**Recommendation:** Drop the `ObjectId` argument and derive the storage key from `object.id()`.

### 4.3 `LooseStore::default()` root is surprising

`LooseStore::default()` uses `"."` as the storage root (`crates/ewm-git/src/store.rs:104`). This is not documented and can pollute the current working directory.

**Recommendation:** Require an explicit path in the constructor and remove `Default`, or document the behavior clearly.

### 4.4 Magic numbers

Several constants lack named definitions or documentation:

- `1.0 / 3072.0` collision probability constant appears in multiple places.
- `1000` iteration safety limit in De Bruijn path search (`materialize.rs:459`).
- `-1e9` causal-mask value (`hllset-repro/src/model.rs:337`).
- `0.15` tolerance in collision theory check (`hllset-repro/src/phase1.rs:107`).
- `recent_window`, `width`, `stride`, `tau_min` in Phase 3 binaries are hard-coded.

**Recommendation:** Extract these into named constants with comments explaining their derivation.

---

## 5. Testing & Quality Gaps

### 5.1 No benchmarks

There is no `criterion` or custom benchmark harness. HLLSet union/intersection and materialization are the obvious candidates.

**Recommendation:** Add `criterion` benchmarks for hot operations before optimizing.

### 5.2 No property-based or fuzz tests

Collision-finding code in `crates/cortex-core/src/lut.rs:108` and `crates/hllset-attn/src/context_vocab.rs:277` uses brute-force searches that can theoretically fail.

**Recommendation:** Add `proptest` or `quickcheck` tests for round-trip and collision properties.

### 5.3 `cortex-core` lacks integration tests

`cortex-core` has only module-level unit tests. The documented M1 exit criterion (round-trip on `corpus/conversation.txt`, 0 leaks, coverage 1.0) is tested manually via the CLI but not by `cargo test`.

**Recommendation:** Add an integration test in `crates/cortex-core/tests/` that runs the full pipeline on the corpus and asserts the M1 invariants.

### 5.4 Missing CI

No `.github/workflows` or equivalent CI configuration was found.

**Recommendation:** Add a GitHub Actions workflow running `cargo test --workspace`, `cargo clippy --workspace -- -D warnings`, and `cargo fmt --check`.

> **Maintainer decision (2026-09-01): NOT planned.** We are building a
> library, not an application. Composability is guaranteed by construction:
> the public surface is overwhelmingly HLLSet → HLLSet transformations, so
> correctness is enforced by the types and by `cargo test --workspace`,
> which remains the single required gate.

### 5.5 Clippy warnings

9 minor warnings remain:

- Type complexity in `materialize.rs:399, 449`
- Type complexity / needless range loop / manual `is_multiple_of` in `hllset-repro`
- Unused variable in `hllset-repro/src/bin/phase3.rs:52`

**Recommendation:** Fix these and enable `#![deny(warnings)]` or CI-level `-D warnings`.

---

## 6. Documentation & Process

### 6.1 Strengths

- Architecture docs exist and are referenced: `CORTEX_ARCHITECTURE.md`, `PROJECT_STRUCTURE.md`, `HLLSET_K_SPACE_MATH.md`, `HLLSET_LUT_TRANSFORMER_ARCHITECTURE.md`.
- Module-level and struct-level rustdoc is generally good, especially in `hllset-core`, `hllset-attn`, and `ewm-git`.
- README has a clear milestone checklist and quick-start commands.

### 6.2 Gaps

- No rustdoc examples for `MaterializeEngine` consumers or `cortex-core` public items.
- No `CONTRIBUTING.md` or architecture decision records.
- Missing changelog / migration notes for the 0.1 → 0.2 compatibility break beyond the README note.
- Public functions in `hllset_core::content_addr` and `TFVec` lack `# Panics` sections.

---

## 7. Prioritized Recommendations

| Priority | Item | Rationale | Effort |
|---|---|---|---|
| **P0** | Replace core library panics with `Result` | Foundation everything builds on | Medium |
| **P0** | Fix O(n²) DAG traversal in `ewm-git` | Real performance bug for large histories | Small |
| **P0** | Reconcile `MaterializeEngine` trait signature | API dishonesty blocks backend work | Small |
| **P1** | Validate `ObjectId` at construction | Prevents data corruption | Small |
| **P1** | Preserve I/O errors in `LooseStore` and surface GC delete failures | Better observability | Small |
| **P1** | Add `cortex-core` integration test for M1 invariants | Catches regressions | Small |
| **P2** | Reduce clones in LUT/K-storage hot paths | Measurable throughput gains | Medium |
| **P2** | Add CI + benchmarks | Quality guardrails | Medium |
| **P2** | Fix absolute paths in CLIs | Portability | Small |
| **P3** | Name magic constants | Maintainability | Small |
| **P3** | Fix clippy warnings and enable `-D warnings` | Code quality | Small |

---

## 8. Architecture Roadmap Observations

The documented milestones are:

- ✅ M1 — `cortex-core` reference pipeline
- ✅ E1 — `ewm-git` evolution store
- ✅ E2 — archive-before-prune
- ⬜ E3 — IPFS archive adapter
- ⬜ M2 — MoE/ETT integration into pipeline
- ⬜ M3 — grounding/search
- ⬜ M4 — EWM alignment + FPGA module DSL
- ⬜ M5 — PyO3 bindings

Before tackling M2/M5, it is worth stabilizing the items in P0/P1 above. The current trait/socket design is a good foundation, but the `MaterializeEngine` signature mismatch and core panics will make integration work harder.

A specific note on M2: `cortex-context` is referenced in docs but does not yet exist as a crate. Consider whether it should be a new crate or a module inside `cortex-core`, given the current dependency rules.

---

## 9. Conclusion

`ewm-cortex` is a strong, principled implementation with a clear architecture and good test coverage. The main risks are production-robustness issues (panics in core APIs, O(n²) traversals, weak validation) rather than fundamental design flaws. Addressing the P0 and P1 recommendations will put the project on a solid footing for the remaining milestones.

---

*Generated by automated codebase review: `cargo test --workspace` and `cargo clippy --workspace` run on 2026-09-01.*

---

## Maintainer Responses (2026-09-01)

### 2.2 O(n²) DAG traversal — NOT a system-path concern (and fixed anyway)

The review correctly observes that `Repository::log` and
`Repository::reachable_mark` used a `Vec` as a visited set. However, the
claim that this is a "real performance bug for large histories" is not
correct **architecturally**:

- The system **never traverses the whole DAG** for navigation. HLLSet
  content addressing selects the correct next HLLSet from the top down:
  - `state(commit)` / `state_channel(commit, GX)` — one SHA1 lookup into
    the object store, O(1);
  - `project(H, t, GX) = H ∩ GX(t)` — reads exactly one commit's channel
    blob, O(1);
  - MoE candidate selection — reads the HEAD commit's channel states and
    intersects, it does not walk ancestors;
  - `merge` — two `state` lookups + one union, O(1).
- The only whole-DAG walks are `log` (a **diagnostic** convenience) and
  `reachable_mark` (GC, run explicitly and off the hot path).

That said, the visited sets were replaced with `HashSet<ObjectId>` in both
places (2026-09-01), so the critique is moot even for diagnostics/GC.

### Rest of the review

Accepted as fair; hardening items completed 2026-09-01:

- ✅ 2.2 — visited sets now `HashSet<ObjectId>` (see above)
- ✅ 2.3 — `MaterializeEngine::materialize` now takes only `&HLLSet`
  (unused `positions` argument removed)
- ✅ 3.1 — `ObjectId::validated` added; used at the deserialize / `read_head`
  boundaries
- ✅ 3.2 — `StoreError::Io` now preserves the underlying `std::io::Error`;
  `LooseStore::get` distinguishes `NotFound` from real I/O failures
- ✅ 3.3 — `gc`/`gc_to` report `delete_failed` in `GcReport`; `gc` returns
  `Result<GcReport>`
- ✅ 5.3 — `crates/cortex-core/tests/pipeline_m1.rs` asserts the M1
  invariants (0 leaks, coverage 1.0, one token per resolved bit) on
  `corpus/conversation.txt`
- ✅ 5.4 — CI not planned (library, not application; see 5.4)

Still open (medium effort, not blocking): 2.1 core-panic `Result`
propagation, 3.4 `parse_cid` hex validation, 3.5 deterministic commit
timestamps, 3.6 CLI absolute paths, 4.x clone audit, 5.1/5.2
benchmarks/property tests, 5.5 clippy warnings. These are scheduled as
hardening, not gate items.

