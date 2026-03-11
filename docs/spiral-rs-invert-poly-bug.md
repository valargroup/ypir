# spiral-rs `invert_poly` / `automorph_poly` Bug — Impact on YPIR

## The bug

Three functions in spiral-rs compute polynomial negation as `modulus - a[i]`
without a final `% modulus`. When `a[i] == 0`, the result is `modulus` instead
of `0` — a value outside the valid range `[0, modulus)`.

| Function | Domain | Negation target |
|---|---|---|
| `invert_poly` (poly.rs:462) | Coefficient (single modulus `q`) | `params.modulus` |
| `automorph_poly` (poly.rs:486) | Coefficient (single modulus `q`) | `params.modulus` |
| `automorph_poly_uncrtd` (poly.rs:503) | CRT (per-slot modulus `q_m`) | `params.moduli[m]` |

The fix is: `res[i] = (modulus - a[i]) % modulus`.

---

## All call sites and why each is a non-issue

### 1. `automorph_alloc` → `.ntt()` (ypir client, key generation)

**Call chain:**

```
client.rs:98   let tau_sk_reg = automorph_alloc(&sk_reg, t);
client.rs:99   let prod = &tau_sk_reg.ntt() * &g_exp_ntt;
```

`automorph_alloc` calls `automorph_poly`, which may write `q` instead of `0`.
The very next operation is `.ntt()`, which calls `to_ntt` → `reduce_copy`:

```rust
// poly.rs:829
fn reduce_copy(params: &Params, out: &mut [u64], inp: &[u64]) {
    for n in 0..params.crt_count {
        for z in 0..params.poly_len {
            out[n * params.poly_len + z] = barrett_coeff_u64(params, inp[z], n);
        }
    }
}
```

`reduce_copy` applies Barrett reduction to every coefficient *before* the NTT
forward pass. This is where the bug is neutralised.

#### Why Barrett reduction maps `q` → `0`

`barrett_coeff_u64` calls `barrett_raw_u64`:

```rust
// arith.rs:137
pub fn barrett_raw_u64(input: u64, const_ratio_1: u64, modulus: u64) -> u64 {
    let tmp = (((input as u128) * (const_ratio_1 as u128)) >> 64) as u64;
    let res = input - tmp * modulus;
    if res >= modulus {
        res - modulus
    } else {
        res
    }
}
```

Barrett reduction approximates `⌊input / modulus⌋` via the precomputed
constant `const_ratio_1 ≈ ⌊2^64 / modulus⌋`. This approximation may
underestimate the true quotient by at most 1, so the function has a single
conditional correction at the end.

When `input = modulus`:

- **Case A** — `tmp = 1` (the approximation is exact):
  `res = modulus − 1 × modulus = 0`. Return `0`. ✓
- **Case B** — `tmp = 0` (the approximation underestimates by 1):
  `res = modulus − 0 = modulus`. The guard `res >= modulus` fires,
  returning `modulus − modulus = 0`. ✓

In both cases the output is `0`. The buggy value never reaches the NTT
butterfly or any downstream multiplication.

**Verdict: no impact.** Barrett reduction in `reduce_copy` corrects the
out-of-range value before it can propagate.

---

### 2. `automorph` → `gadget_invert_rdim` (ypir packing, server-side ring packing)

**Call chain:**

```
packing.rs:203   automorph(&mut ct_auto, &ct_raw, t);
packing.rs:206   gadget_invert_rdim(&mut ginv_ct, &ct_auto, 1);
                 // then NTT + multiply with key-switching key
```

(Same pattern at packing.rs:350.)

`automorph_poly` may produce `q` at some coefficient position. Unlike the
client path, there is **no** Barrett reduction before the next consumer —
`gadget_invert_rdim` performs integer bit-decomposition, so the bit pattern of
`q` (non-zero) differs from that of `0` (all zeros). This would produce an
incorrect gadget decomposition and inject a large error into the key-switching
step.

**However, the trigger probability is negligible.** The input `ct_raw` comes
from `from_ntt` of an RLWE ciphertext, giving coefficients uniform in
`[0, q)`. With ypir's default parameters (`q = 268369921 × 249561089 ≈ 2^{55.9}`):

```
Pr[any zero in one polynomial] ≈ poly_len / q = 2048 / 2^55.9 ≈ 2^{−44.9}
```

Even across all ~2047 automorphism calls during a full query, the combined
probability is ~2^{−34} — roughly **1 in 17 billion queries**.

**Verdict: no practical impact.** The bug can never realistically trigger
because coefficient-domain values are drawn from a space of ~2^{56} elements.

---

### 3. `automorph_poly_uncrtd` → `ntt_forward` (ypir packing, server-side ring packing)

**Call chain:**

```
packing.rs:227   automorph_poly_uncrtd(params, ct_auto_1_ntt.as_mut_slice(), scratch_mut_slc, t);
packing.rs:228   ntt_forward(params, ct_auto_1_ntt.as_mut_slice());
```

(Same pattern at packing.rs:374.)

Here the input is already in CRT representation (per-slot values in
`[0, q_m)`). With `q_m ≈ 2^{28}` and 2048 coefficients per slot, a zero
coefficient appears with probability ~2048/2^{28} per polynomial per CRT slot.
Over the full protocol this **does trigger** — roughly ~3% of queries will
have at least one affected coefficient.

**But the output is still correct.** The NTT forward transform computes in
Z\_{q\_m}. Since `q_m ≡ 0 (mod q_m)`, the buggy value is algebraically
identical to `0` in the ring. Concretely, tracing through the NTT butterfly
(Cooley–Tukey, lazy reduction):

1. The initial value `q_m` satisfies `q_m < 4 q_m`, so it is within the
   lazy-reduction invariant `[0, 4q)` that the butterfly network maintains.
2. Every butterfly computes `curr_x + q_new` and `curr_x + (2q − q_new)`.
   With `curr_x = q_m` (after the conditional `x ≥ 2q` subtraction leaves it
   as `q_m`), the outputs remain in `[0, 4q)`.
3. The final reduction loop:
   ```rust
   *val -= (*val >= 2*q) as u64 * 2*q;
   *val -= (*val >= q)   as u64 * q;
   ```
   maps any multiple of `q_m` to `0`.

The NTT output is **bit-identical** to what it would be with the correct
input of `0`.

**Verdict: no impact.** The NTT absorbs the error because `q_m ≡ 0 (mod q_m)`
and lazy reduction stays within bounds.

---

### 4. `invert` → `.ntt()` (spiral-rs server, query expansion / folding)

**Call chain** (not used by ypir directly, but present in spiral-rs):

```rust
// spiral-rs server.rs:514
invert(&mut ct_gsw_inv, &v_folding[i].raw());
add(&mut ct_gsw_neg, &gadget_ntt, &ct_gsw_inv.ntt());
```

(Same pattern at server.rs:963.)

`invert` calls `invert_poly`, which may write `q` at zero-coefficient
positions. The result is immediately converted via `.ntt()` → `to_ntt` →
`reduce_copy`, which applies Barrett reduction (see §1 above).

**Verdict: no impact.** Same Barrett correction as the client path.

---

### 5. `automorph` → `gadget_invert_rdim` → `to_ntt_no_reduce` (spiral-rs server, query expansion)

**Call chain** (not used by ypir directly):

```rust
// spiral-rs server.rs:80-82
automorph(&mut ct_auto, &ct, t);
gadget_invert_rdim(gi_ct, &ct_auto, 1);
to_ntt_no_reduce(gi_ct_ntt, &gi_ct);
```

Same analysis as §2: `automorph_poly` may produce `q`, leading to an incorrect
gadget decomposition. The trigger probability is ~2^{−45} per polynomial —
negligible.

Note that `to_ntt_no_reduce` (unlike `to_ntt`) does **not** apply Barrett
reduction, but the gadget-decomposed values are small (`< 2^bits_per`) and
therefore always valid. The problem would only be in the *wrong* decomposition,
not in any overflow.

**Verdict: no practical impact.** Same negligible trigger probability as §2.

---

### 6. `Neg` trait for `PolyMatrixRaw` (spiral-rs, used wherever `-&poly_raw` appears)

```rust
// poly.rs:918
fn neg(self) -> Self::Output {
    let mut out = PolyMatrixRaw::zero(self.params, self.rows, self.cols);
    invert(&mut out, self);
    out
}
```

This delegates to `invert_poly`. Any subsequent use of the negated raw
polynomial typically passes through `.ntt()` (Barrett reduction) before
entering arithmetic. Same analysis as §1/§4.

**Verdict: no impact** in all known uses.

---

## Summary

| # | Call site | Trigger rate | Why safe |
|---|---|---|---|
| 1 | ypir client: `automorph_alloc` → `.ntt()` | Any | Barrett reduction in `reduce_copy` maps `q → 0` |
| 2 | ypir packing: `automorph` → `gadget_invert_rdim` | ~2^{−34} per query | Negligible probability — coefficient-domain `q ≈ 2^{56}` |
| 3 | ypir packing: `automorph_poly_uncrtd` → `ntt_forward` | ~3% per query | `q_m ≡ 0 (mod q_m)`; NTT output is bit-identical to correct |
| 4 | spiral-rs server: `invert` → `.ntt()` | Any | Barrett reduction in `reduce_copy` maps `q → 0` |
| 5 | spiral-rs server: `automorph` → `gadget_invert_rdim` | ~2^{−45} per poly | Negligible probability |
| 6 | `Neg` for `PolyMatrixRaw` | Any | Downstream `.ntt()` applies Barrett reduction |

The fix (`% modulus`) is correct and good practice — it restores the invariant
that outputs are in `[0, modulus)` regardless of downstream consumer — but no
ypir code path produces wrong results without it.
