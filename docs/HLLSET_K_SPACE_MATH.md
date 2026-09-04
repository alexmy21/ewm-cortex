# HLLSet as the K-Space of Attention — Mathematical Summary

> **Status:** Design notes from the 2026-08-28 session (ewm-cortex workspace).
> **Companions:** [`HLLSET_LUT_TRANSFORMER_ARCHITECTURE.md`](HLLSET_LUT_TRANSFORMER_ARCHITECTURE.md)
> (implementation architecture, same directory). Canonical framework docs:
> [`GENERAL_HLLSET_ALGEBRA.md`](../../hllset-next/_DOCS/dev/GENERAL_HLLSET_ALGEBRA.md) and
> [`CONVERSATION_CONTEXT_LLM.md`](../../hllset-next/_DOCS/arch/CONVERSATION_CONTEXT_LLM.md).

---

## 1. What K is, mechanically

In attention, the only computed quantity is the dot product:

```math
\text {score (i, j)} = \text{softmax}_j( \frac {q_i · k_j}  {√d} )
```

with `q = W_Q x`, `k = W_K x`, `v = W_V x`. Two immediate consequences:

### **Invariance.** For any invertible matrix `A`, the substitution

```math
q → A q ,\text{  }   k → A⁻ᵀ k
```

leaves every score unchanged. The labels or semantics attached to K are
therefore **not invariants of the mechanism** — the only invariant object is
the matrix of pairwise similarities `q_i · k_j`.

### **The only genuine constraint on K.**

The key space must support the routing
geometry the task needs. Nothing in the mechanism references an ontology.
Ontology is an interpretability gloss projected onto the geometry after
training; it is not a requirement of the computation.

---

## 2. The partition principle

**Definition.** A *K-representation* of a vocabulary `V` is any function
`k : V → S` into a key space `S`.

**Claim.** Any partition of `V` into mutually exclusive subsets
`V = ⊔_{b ∈ S} V_b` yields a K-representation `k(t) = b(t)`. An ontology is
one such partition; the HLLSet bit assignment is another. Mechanically they
are equivalent up to the resolution of `S`. The human convenience of an
ontology is an interpretation property, not a computational one.

---

## 3. The HLLSet partition

From `hllset-core` (`M = 1024`, `P = 10`, `BITS_PER_REG = 32`,
`TOTAL_BITS = 32768`):

```text
hash    = murmur3_64(token, seed 0)
reg     = hash & 1023                      (low 10 bits)
rem     = hash >> 10                       (54 bits)
tz      = trailing_zeros(rem), clamped to 31   (rem = 0 ⇒ tz = 31)
bit     = reg * 32 + tz                    (0 .. 32767)
```

Each token maps to exactly one `<reg, tz>` cell: the assignment is a
**mutually exclusive partition** of the vocabulary into at most 32,768
cells.

**Distribution.** `tz` is geometric:

```text
P(tz = z) = 2^-(z+1)   for z = 0..30
P(tz = 31) = 2^-31     (tail absorbed by the clamp)
```

so the partition is **multiresolution** *in expectation over the hash
space*: a `(reg, tz = z)` cell holds ≈ `N · 2^-(z+11)` tokens
(`z = 0`: `N/2048`, `z = 4`: `N/32768`, `z ≥ 6`: mostly empty or singleton
for `N ≈ 50K`). The register axis is uniform; the `tz` axis is a
coarse-to-fine rarity gradient.

**Hash prior vs token reality.** This is a statement about hash values,
not about tokens. For an actual vocabulary, the count in a given cell is
binomial with the expectation above and standard deviation ≈ `√λ` — the
deviations dominate for small `N`. And for a *corpus*, token occurrences
are Zipfian, not uniform: the multiplicity-free bitmap records which cells
are occupied, but not how often. The LUT's **TF field** (`TFVec`, 32768
entries indexed by bit, monotonic CRDT) records the empirical,
occurrence-weighted per-bit counts — the data-dependent multiresolution
that corrects the hash prior exactly where it can be wrong for tokens.

---

## 4. Resolution and collision statistics

**Pairwise collision probability.** Two random tokens share a cell iff they
share `reg` and `tz`:

```text
p = (1/M) · Σ_z P(tz = z)² = (1/1024) · (1/3) = 1/3072 ≈ 3.26 × 10⁻⁴
```

Expected collision partners for a vocabulary of `N` token types:

| N        | partners/token | occupied cells (of 32,768) |
|----------|----------------|----------------------------|
| 10,000   | ~3.3           | ~3,707                     |
| 50,000   | ~16.3          | ~6,085                     |
| 100,000  | ~32.6          | ~7,109                     |
| 200,000  | ~65.1          | ~8,133                     |

**Sub-vocabulary precision.** For an active sub-vocabulary of `n` tokens,
the expected number of colliding pairs is `C(n,2) · p`:

```text
n = 100  →  ~1.6 pairs      (extraction essentially exact)
n = 300  →  ~14.6 pairs     (mostly singletons)
n = 1000 →  ~162.6 pairs    (needs structural layers or multi-seed)
```

Managing LLM context as a **sub-lattice** keeps the active vocabulary in
the regime where the single-seed K-space is near-exact.

**Multi-seed.** With `s` independent seeds, the collision probability is
`p^s`:

| s | collision      | partners at N = 200K |
|---|----------------|----------------------|
| 1 | 3.26 × 10⁻⁴    | ~65                  |
| 2 | 1.06 × 10⁻⁷    | ~0.02                |
| 3 | 3.45 × 10⁻¹¹   | ~7 × 10⁻⁶            |

`CatalogLUT` uses 3 seeds with a ≥2-of-3 quorum: false acceptance per value
≈ `p² ≈ 10⁻⁷`, independent of vocabulary size.

---

## 5. The IICA chain is an adjunction

Define two posets:

- **Token realm** `(T, ⊆)`: sets of tokens ordered by inclusion.
- **HLLSet realm** `(L, ⊆)`: bit-vectors over 32,768 positions ordered by
  bitwise inclusion — a bounded distributive lattice with join `∪` (OR) and
  meet `∩` (AND).

The IICA morphism chain

```text
{tokens} --ingest I--> HLLSet --materialize M--> {tokens}
```

defines a **Galois connection**:

```text
I(S) ⊆ H   ⟺   S ⊆ M(H)
```

Proof: `I(S) ⊆ H` says every bit set by a token in `S` is in `H`;
`S ⊆ M(H)` says every token in `S` lies in the collision group of some bit
in `H`. These are the same statement.

Composites:

```text
M ∘ I  = closure operator on the token realm
         (expands S to the union of its collision groups;
          extensive, monotone, idempotent)

I ∘ M  = identity on the materializable sub-lattice
         (the HLLSet realm is a retract of the token realm)
```

**Exactness asymmetry.** `I` is an exact join-homomorphism — `I(A ∪ B) =
I(A) ∪ I(B)` — but only a conservative meet-map:

```text
I(A ∩ B) ⊆ I(A) ∩ I(B)
```

equality can fail when two different tokens collide on one bit. The
2-/3-gram structural layers exist to recover the discrimination that `∩`
conservatively loses.

---

## 6. Q, K, V in the HLLSet realm

The adjunction assigns the transformer's three roles:

```text
Q   = I(query)                  fresh ingestion of the query into the lattice
K   = stored HLLSets             the context sub-lattice, content-addressed by CID
V   = M(K ∩ Q)                   materialized tokens of the matched subset
attention = a lattice morphism   BSS inclusion τ(Q ⊆ K), Jaccard, or overlap popcount
```

This is CAAL-LLM restated: *learning = accumulating HLLSets; inference =
BSS retrieval; output = materialization.*

**K-storage is the LUT.** The LUTs are the concrete witnesses of the
adjunction:

- `TokenLUT.forward : token → (reg, tz)` — the ingestion direction `I`.
- `TokenLUT.index   : (reg, tz) → [tokens]` — the materialization
  direction `M` (the collision group behind each key).
- `CatalogLUT.forward : value → 3 positions`, `CatalogLUT.reverse :
  position → [values]` — the same adjunction with a multi-seed quorum.

A key is a **bit address**; the label of that key is its collision group.
This is content-addressed memory in the most literal sense: the address is
computed from the content, and the stored value is the set of contents that
share the address.

---

## 7. Sub-lattice consistency and precise extraction

**Theorem (restriction is a homomorphism).** For any sub-lattice `L' ⊆ L`
(any bit-closed subset), the projection `π : L → L'` commutes with `∪` and
`∩`. Local joins equal the global join restricted:

```text
π(A ∪ B) = π(A) ∪ π(B) ,   π(A ∩ B) = π(A) ∩ π(B)
```

**Naturality.** Materialization against a sub-vocabulary LUT is the
restriction of the full materialization: partial views are consistent
projections, never approximations.

**Sub-lattice extraction theorem.** Let `C` be a context sub-lattice and
`LUT` a reverse index with **coverage** of `C`'s active bits
(`LUT.bits ⊇ bits(C)`). Then `M(Q ∩ C)`:

1. is deterministic and touches no bit outside `C`;
2. single-seed: exact up to collision groups (tier 1);
3. multi-seed quorum: exact membership (tier 2);
4. structural (2-/3-gram cross-validation, De Bruijn): disambiguates
   tier-1 groups through order (tier 3).

The coverage invariant is the single condition under which "precise
extraction" holds literally; `confidence = resolved_bits / total_bits` is
the gauge — `confidence = 1.0` iff coverage holds.

---

## 8. Context, MoE, and the Expert Think Tank

Formal definitions of the LLM-context objects, per the 2026-08-28 session.

**History and observation.** The history is an ordered sequence of original
observations `H = S₁ … Sₜ`, each `Sᵢ ∈ L` an HLLSet. The recent observation
window is `R(t) = ∪ { Sᵢ : t − m < i ≤ t }`.

**1. LLM context — sub-lattice.** `C(t) = ∪ { Sᵢ : i ∈ I(t) }` for the
most relevant observations `I(t)`. The generator is `G(t) = {Sᵢ}`; the
context sub-lattice is its span `⟨G(t)⟩`; the stored object is the top
`C(t)`. Connectedness means the overlap graph of `G(t)` (edges where
`Sᵢ ∩ Sⱼ ≠ ∅`, or R-links `S_prev ∩ S_curr`) is connected.

**2. MoE — convolution of the context sub-lattice.** Candidates are the
original observations of recent history, retrieved through the
**content-addressed evolution store** (`ewm-git`): recent history is the
set of commits reachable from `HEAD`; the commit hash is the timer and the
commit state is the union HLLSet. The temporal pyramid is replaced by this
store and is no longer used for grouping. Candidates are shuffled by
content address and convolved:

```text
π    = sort_by_sha1( { Sᵢ : reachable from HEAD } )   // IICA shuffle
F_k  = { S_π(i) : i ∈ [k·s, k·s + w) }       // convolution frame (width w, stride s)
E_k  = ∪ F_k                                  // Expert
MoE(t) = { E_k }
```

Multi-shuffle: `π_c = sort_by_sha1(Sᵢ ‖ salt_c)` gives independent MoE
partitions for ensembling.

**3. Expert.** `E_k = ∪ F_k` — the union of the HLLSets of one frame.
Every object is used as a single HLLSet = union of its members.

**4. Expert Think Tank.** Ordered by relevance to the recent observation:

```text
ρ(E) = BSSτ(E, R(t)) = |E ∩ R(t)| / |R(t)|     // pure BSS, no recency decay
ETT(t) = { (E_k, ρ_k) }  sorted by ρ descending, ties broken by recency
```

Recency acts **only** at candidate extraction (the commit DAG's reachable
window from `HEAD`), not inside `ρ` and not in the frame grouping.

**5. Expert Leader.** `EL(t) = E_{k*}` where `k* = argmax_k ρ_k`.

**6. Resolution (A+B).** The final response sub-lattice is the thresholded
union — the leader plus every expert above `τ_min`:

```text
F(t) = EL ∪ ( ∪ { E_k : ρ_k ≥ τ_min } )        // bitmap level (monotone, IICA)
```

Materialization resolves `F(t)` by collection intersection per bit (TF
only breaks collision ties); the OUTPUT is then ranked by the TF field so
the leader's tokens dominate generation — the bitmap decides *what can be
said*, the TF ranking decides *what is said first*.

**Properties.** `F(t) ⊆ C(t)`, so `|F(t)|` is bounded by the context bits.
The pipeline is monotone and IICA throughout; the coverage invariant
(`confidence = 1.0`) is the precondition for materializing `F(t)`.

---

## 9. One-page summary

```text
K is a routing address. The only invariant is the similarity geometry q_i·k_j.
Any mutually exclusive partition of the vocabulary is a valid K-representation.

HLLSet partition:  bit = 32·reg + tz,  reg uniform over 1024,
                   tz geometric (P(tz=z) = 2^-(z+1)),  multiresolution
                   (hash prior; LUT TF field carries the empirical token counts).

Resolution:        pairwise collision p = 1/3072;
                   multi-seed s: p^s (s=2 ⇒ 10⁻⁷, s=3 ⇒ 10⁻¹¹);
                   sub-vocabulary n: C(n,2)/3072 expected collisions.

IICA adjunction:   I ⊣ M,   I(S) ⊆ H ⟺ S ⊆ M(H);
                   M∘I closure,  I∘M = id on the materializable sub-lattice;
                   join exact, meet conservative.

Q,K,V:             Q = I(query),  K = context sub-lattice (stored in LUTs),
                   V = M(K∩Q),  attention = BSS/Jaccard lattice morphism.

Sub-lattice:       restriction is a lattice homomorphism; extraction is
                   deterministic, local, and precise under LUT coverage.

Context/MoE:       C(t) = ∪ relevant Sᵢ (sub-lattice top); candidates from
                   recent history via the ewm-git commit DAG (HEAD reachable);
                   π = sort_by_sha1(Sᵢ) — IICA shuffle; E_k = ∪ frame_k;
                   ETT ranked by ρ = BSSτ(E_k, R(t)); EL = argmax;
                   F(t) = EL ∪ { E_k : ρ_k ≥ τ_min }  (A+B resolution).

Evolution:         ewm-git replaces the temporal pyramid; commit = timer,
                   commit state = union HLLSet, merge = lattice join,
                   H(t) = (S(t), H(t-1), D, R, N), pruning = GC.
```
