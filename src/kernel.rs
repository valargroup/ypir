// kernel.rs — Batched matrix-vector dot product, the hot loop of YPIR.
//
// The core operation is  c += A · B^T  where:
//   - A is K×a_elems  (query vectors, K=1 for SimplePIR)
//   - B^T is b_rows×b_cols stored in column-major (transposed) layout
//   - c is K×b_cols  output accumulator
//
// Three implementations coexist behind feature flags:
//   1. `fast_batched_dot_product_explicit_avx512` — hand-written AVX-512
//      intrinsics (8 u64 lanes per cycle).  Requires `explicit_avx512`.
//   2. `fast_batched_dot_product_implicit` — scalar fallback using plain
//      u64 arithmetic.  Always available.
//   3. Rayon parallel wrappers (feature `rayon`) — partition the output
//      columns across threads, then call the same inner kernel per thread.
//
// Both (1) and (2) decompose each u64 element into its low and high 32-bit
// halves before multiplying.  This is necessary because spiral-rs stores
// values in a CRT representation where the modulus q = q0 * q1, with q0
// and q1 each fitting in 32 bits.  Accumulating lo and hi products
// separately avoids u64 overflow, and the final result is recovered via
// Barrett reduction on each half followed by CRT recomposition.

use spiral_rs::{arith::*, params::*, poly::*};

use crate::server::ToU64;

use super::server::ToM512;

#[cfg(feature = "explicit_avx512")]
use std::arch::x86_64::*;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

// ── Send/Sync raw-pointer newtype for rayon closures ───────────────────────
//
// Rayon's `par_chunks_mut().for_each(closure)` requires the closure to be
// `Send + Sync`, which means any captured raw pointer must also be `Send +
// Sync`.  Raw pointers are `!Send + !Sync` by default.  A common idiom is
// to cast `ptr as usize` to smuggle the address across the boundary and
// cast back inside the closure, but that pattern drops pointer provenance
// and will fail under strict-provenance rules (tracked by Rust's strict-
// provenance lint and already flagged by Miri under
// `-Zmiri-strict-provenance`).
//
// Wrap the raw pointer in a `Copy` newtype with explicit `unsafe impl
// Send + Sync`.  The safety obligation is the same as with the `as usize`
// dance: (1) the pointee outlives the parallel region, (2) no aliasing
// `&mut` exists for the duration, (3) writes through the pointer are
// disjoint across threads.  Moving the `unsafe` here makes the contract
// an attribute of the type rather than folklore spread across two
// call sites.
#[cfg(feature = "rayon")]
#[derive(Copy, Clone)]
struct SendPtr<T>(*const T);

#[cfg(all(feature = "rayon", feature = "explicit_avx512"))]
#[derive(Copy, Clone)]
struct SendMutPtr<T>(*mut T);

#[cfg(feature = "rayon")]
unsafe impl<T> Send for SendPtr<T> {}
#[cfg(feature = "rayon")]
unsafe impl<T> Sync for SendPtr<T> {}

#[cfg(all(feature = "rayon", feature = "explicit_avx512"))]
unsafe impl<T> Send for SendMutPtr<T> {}
#[cfg(all(feature = "rayon", feature = "explicit_avx512"))]
unsafe impl<T> Sync for SendMutPtr<T> {}

#[cfg(feature = "rayon")]
impl<T> SendPtr<T> {
    /// Extract the inner raw pointer.  Takes `self` by value so that RFC
    /// 2229 disjoint captures treats uses of `send_ptr.get()` as a whole-
    /// struct access (method call on `Self`) rather than field access on
    /// `.0` — the latter would cause the closure to capture `&*const T`,
    /// which is `!Sync`, defeating the whole point of the newtype.
    #[inline(always)]
    fn get(self) -> *const T {
        self.0
    }
}

#[cfg(all(feature = "rayon", feature = "explicit_avx512"))]
impl<T> SendMutPtr<T> {
    /// Extract the inner raw pointer. See `SendPtr::get` for why this takes
    /// `self` by value.
    #[inline(always)]
    fn get(self) -> *mut T {
        self.0
    }
}

// ── Shared helpers ─────────────────────────────────────────────────────────
//
// Every kernel variant (AVX-512 / scalar × rayon / sequential) goes through
// these two steps:
//
//   1. Split the flat `a` buffer of length K*a_elems into K per-batch query
//      vectors of length a_elems.
//   2. For each output cell, finalize `(sum_lo, sum_hi)` into `c[j]` by
//      applying Barrett reduction to each CRT component, CRT-recomposing,
//      and accumulating into the (possibly pre-existing) value in `c[j]`.
//
// Extracted so the four code paths don't each carry their own copy and risk
// drifting (e.g. a bugfix applied to three of four call sites).

/// Split `a` (laid out as K batch-rows concatenated) into K equal-length
/// sub-slices, each of length `a.len() / K`.
///
/// # Panics
///
/// Panics if `K == 0` (a const-generic misuse that should be caught at
/// the call site anyway) or if `a.len()` is not a multiple of K.
///
/// The length check is a **runtime** `assert!`, not a `debug_assert!`,
/// because violating it would silently drop the trailing `a.len() % K`
/// elements: `chunks_exact` discards the remainder rather than panicking.
/// For a PIR kernel where the output goes back to a client as a
/// ciphertext, a silently-truncated dot product produces a response
/// that decrypts to the wrong value — a correctness failure that no
/// downstream code can recover from and that release builds must not
/// be able to hit.  The check is one cmp+jne per kernel invocation
/// (not per inner-loop iteration) so it does not affect hot-path
/// performance.
#[inline(always)]
fn split_a<const K: usize>(a: &[u64]) -> [&[u64]; K] {
    assert!(K > 0, "split_a requires K > 0");
    assert_eq!(
        a.len() % K,
        0,
        "split_a: a.len() ({}) must be a multiple of K ({})",
        a.len(),
        K
    );
    let mut out: [&[u64]; K] = [&[]; K];
    for (slot, chunk) in out.iter_mut().zip(a.chunks_exact(a.len() / K)) {
        *slot = chunk;
    }
    out
}

/// Finalize a single output cell from its two CRT half-sums.
///
/// Applies Barrett reduction to each half-sum (mod q0 and mod q1, the two
/// CRT primes with q = q0 * q1), CRT-recomposes into a single element
/// mod q, adds it to the pre-existing `*c_cell`, and applies a final
/// Barrett reduction so the stored value remains in `[0, q)`.
///
/// This is the common tail of both the scalar and AVX-512 kernels.
///
/// # Preconditions
///
/// - `*c_cell` is already in `[0, q)` before the call (i.e. the buffer
///   was either zero-initialized by the caller, or previously written
///   through this same function).  The final Barrett step only folds a
///   single q-worth of carry, so violating this would leave `*c_cell`
///   outside `[0, q)` and corrupt later accumulations.
/// - `sum_lo`, `sum_hi` fit in `u64` (the inner loops tile the
///   accumulation so this holds; see the `chunk_size` tuning in each
///   kernel).
#[inline(always)]
fn writeback(params: &Params, c_cell: &mut u64, sum_lo: u64, sum_hi: u64) {
    let lo = barrett_coeff_u64(params, sum_lo, 0);
    let hi = barrett_coeff_u64(params, sum_hi, 1);
    let res = params.crt_compose_2(lo, hi);
    *c_cell = barrett_u64(params, *c_cell + res);
}

/// AVX-512 tail: horizontally reduce the two 512-bit accumulators to two
/// `u64`s, then delegate to `writeback`.
///
/// Stores each `__m512i` into an 8-lane `u64` scratch array on the stack
/// and sums the lanes.  Accepts the same preconditions as `writeback` for
/// the final scalar step.
///
/// # Safety
///
/// - Caller must be running on a CPU with AVX-512F.  The `__m512i`
///   arguments are produced by `_mm512_setzero_si512` and subsequent
///   `_mm512_add_epi64` / `_mm512_mul_epu32` calls; this helper only
///   reads them back via an unaligned store.
///
/// We use `_mm512_storeu_si512` rather than `_mm512_store_si512` because
/// a stack-allocated `[u64; 8]` is only 8-byte aligned by Rust's
/// type-alignment rules, and the aligned store formally requires the
/// destination to be 64-byte aligned (Intel SDM: #GP on misalignment).
/// LLVM currently lowers the aligned intrinsic to `vmovdqu64` in most
/// cases so it has "worked in practice", but that's a codegen accident;
/// the unaligned variant is spec-correct on any alignment and the two
/// generate identical code on Skylake-X and later (same `vmovdqu64`).
#[cfg(feature = "explicit_avx512")]
#[inline(always)]
unsafe fn writeback_avx512(
    params: &Params,
    c_cell: &mut u64,
    sum_lo: __m512i,
    sum_hi: __m512i,
) {
    let mut vl = [0u64; 8];
    let mut vh = [0u64; 8];
    _mm512_storeu_si512(vl.as_mut_ptr() as *mut _, sum_lo);
    _mm512_storeu_si512(vh.as_mut_ptr() as *mut _, sum_hi);
    writeback(params, c_cell, vl.iter().sum(), vh.iter().sum());
}

/// Dispatches to the AVX-512 implementation when `explicit_avx512` is
/// enabled, falling back to the scalar implementation otherwise.
pub fn fast_batched_dot_product<const K: usize, T: Copy>(
    params: &Params,
    c: &mut [u64],
    a: &[u64],
    a_elems: usize,
    b_t: &[T], // transposed
    b_rows: usize,
    b_cols: usize,
) where
    *const T: ToM512 + ToU64,
{
    #[cfg(feature = "explicit_avx512")]
    fast_batched_dot_product_explicit_avx512::<K, T>(params, c, a, a_elems, b_t, b_rows, b_cols);
    #[cfg(not(feature = "explicit_avx512"))]
    fast_batched_dot_product_implicit::<K, T>(params, c, a, a_elems, b_t, b_rows, b_cols);
}

#[cfg(not(feature = "explicit_avx512"))]
pub fn fast_batched_dot_product_explicit_avx512<const K: usize, T: Copy>(
    _params: &Params,
    _c: &mut [u64],
    _a: &[u64],
    _a_elems: usize,
    _b_t: &[T], // transposed
    _b_rows: usize,
    _b_cols: usize,
) where
    *const T: ToM512 + ToU64,
{
    panic!("explicit_avx512 not enabled");
}

/// AVX-512 batched dot product: c += A · B^T
///
/// Uses `_mm512_mul_epu32` to multiply 8 pairs of 32-bit values at once,
/// accumulating into 64-bit lanes.  The inner loop is tiled into
/// `chunk_size` groups of 8 elements to keep intermediate sums in
/// registers without overflowing.
///
/// Layout:
///   - `a`:   K contiguous row vectors, each of length `a_elems`
///   - `b_t`: column-major (transposed), element [row, col] at `col * b_rows + row`
///   - `c`:   K output rows, each of length `b_cols`
#[cfg(feature = "explicit_avx512")]
pub fn fast_batched_dot_product_explicit_avx512<const K: usize, T: Copy>(
    params: &Params,
    c: &mut [u64],
    a: &[u64],
    a_elems: usize,
    b_t: &[T], // transposed
    b_rows: usize,
    b_cols: usize,
) where
    *const T: ToM512 + ToU64,
{
    assert_eq!(a_elems, b_rows);

    let simd_width = 8; // 512 bits / 64 bits per lane

    // Tile the accumulation dimension into chunks to balance register
    // pressure against loop overhead.  8192/K keeps the working set in
    // L1; `.min(a_elems / simd_width)` handles small inputs.
    let chunk_size = (8192 / K.next_power_of_two()).min(a_elems / simd_width);
    let num_chunks = (a_elems / simd_width) / chunk_size;

    // ── Rayon parallel path (AVX-512) ──────────────────────────────────
    //
    // Column-partitioned / batch-broadcast: each worker owns a disjoint
    // output-column range and computes all K batch rows for those columns.
    // This preserves the batching win by streaming each `b_t` column once
    // and reusing it across K query rows in registers.
    //
    // Safety of the pointer captures:
    //   - `a` and `b_t` are read-only for the entire duration of the
    //     parallel region; the caller holds &[u64] / &[T] borrows.
    //   - Each thread writes only to `batch * b_cols + j` for its disjoint
    //     `j` range, so output writes are disjoint across threads.
    //   - Raw pointers are wrapped in `SendPtr` / `SendMutPtr` (Copy + Send
    //     + Sync) so the rayon closure can capture them while preserving
    //     provenance.
    #[cfg(feature = "rayon")]
    {
        let b_send = SendPtr(b_t.as_ptr());
        let a_send = SendPtr(a.as_ptr());
        let c_send = SendMutPtr(c.as_mut_ptr());
        let a_len = a.len();
        let num_threads = rayon::current_num_threads();
        // `.max(1)` keeps empty-column cases from producing a zero chunk
        // size; empty ranges below simply do no work.
        let cols_per_chunk = ((b_cols + num_threads - 1) / num_threads).max(1);

        (0..num_threads).into_par_iter().for_each(move |chunk_idx| {
            let j_start = chunk_idx * cols_per_chunk;
            let j_end = (j_start + cols_per_chunk).min(b_cols);

            unsafe {
                let b_ptr = b_send.get();
                let c_ptr = c_send.get();
                let a_slice = std::slice::from_raw_parts(a_send.get(), a_len);
                let a_slcs = split_a::<K>(a_slice);

                // Outer loop over row chunks (accumulation dimension).
                // Each k_outer processes `chunk_size` groups of 8 (simd_width)
                // elements, accumulating partial sums in AVX-512 registers.
                for k_outer in 0..num_chunks {
                    for j in j_start..j_end {
                        let mut total_sum_lo = [_mm512_setzero_si512(); K];
                        let mut total_sum_hi = [_mm512_setzero_si512(); K];
                        let mut tmp = [_mm512_setzero_si512(); K];

                        // Inner dot-product: multiply 8 a[] values by 8 b[]
                        // values at a time using _mm512_mul_epu32, accumulating
                        // low and high 32-bit halves separately to avoid overflow.
                        for k_inner in 0..chunk_size {
                            let k = simd_width * (k_outer * chunk_size + k_inner);
                            let b_val_simd = b_ptr.add(j * b_rows + k).to_m512();

                            for batch in 0..K {
                                tmp[batch] = _mm512_load_si512(
                                    a_slcs[batch].as_ptr().add(k) as *const _,
                                );
                            }

                            for batch in 0..K {
                                let a_val_lo = tmp[batch];
                                let a_val_hi = _mm512_srli_epi64(tmp[batch], 32);
                                total_sum_lo[batch] = _mm512_add_epi64(
                                    total_sum_lo[batch],
                                    _mm512_mul_epu32(a_val_lo, b_val_simd),
                                );
                                total_sum_hi[batch] = _mm512_add_epi64(
                                    total_sum_hi[batch],
                                    _mm512_mul_epu32(a_val_hi, b_val_simd),
                                );
                            }
                        }

                        for batch in 0..K {
                            writeback_avx512(
                                params,
                                &mut *c_ptr.add(batch * b_cols + j),
                                total_sum_lo[batch],
                                total_sum_hi[batch],
                            );
                        }
                    }
                }
            }
        });
        return;
    }

    #[cfg(not(feature = "rayon"))]
    unsafe {
        let a_slcs = split_a::<K>(a);
        let b_ptr = b_t.as_ptr();

        for k_outer in 0..num_chunks {
            for j in 0..b_cols {
                let mut total_sum_lo = [_mm512_setzero_si512(); K];
                let mut total_sum_hi = [_mm512_setzero_si512(); K];
                let mut tmp = [_mm512_setzero_si512(); K];

                for k_inner in 0..chunk_size {
                    let k = simd_width * (k_outer * chunk_size + k_inner);
                    let b_val_simd = b_ptr.add(j * b_rows + k).to_m512();

                    for batch in 0..K {
                        tmp[batch] = _mm512_load_si512(a_slcs[batch].as_ptr().add(k) as *const _);
                    }

                    for batch in 0..K {
                        let a_val_lo = tmp[batch];
                        let a_val_hi = _mm512_srli_epi64(tmp[batch], 32);

                        total_sum_lo[batch] = _mm512_add_epi64(
                            total_sum_lo[batch],
                            _mm512_mul_epu32(a_val_lo, b_val_simd),
                        );
                        total_sum_hi[batch] = _mm512_add_epi64(
                            total_sum_hi[batch],
                            _mm512_mul_epu32(a_val_hi, b_val_simd),
                        );
                    }
                }

                for (batch, c_row) in c.chunks_exact_mut(c.len() / K).enumerate() {
                    writeback_avx512(params, &mut c_row[j], total_sum_lo[batch], total_sum_hi[batch]);
                }
            }
        }
    }
}

/// Scalar fallback for batched dot product: c += A · B^T
///
/// Same mathematical operation as the AVX-512 variant but using plain
/// u64 arithmetic (one element per iteration instead of eight).
pub fn fast_batched_dot_product_implicit<const K: usize, T: Copy>(
    params: &Params,
    c: &mut [u64],
    a: &[u64],
    a_elems: usize,
    b_t: &[T], // transposed
    b_rows: usize,
    b_cols: usize,
) where
    *const T: ToM512 + ToU64,
{
    assert_eq!(a_elems, b_rows);
    #[cfg(feature = "rayon")]
    assert_eq!(K, 1);

    let simd_width = 1; // scalar: one element per iteration

    // Larger chunk_size than AVX-512 since there are no vector registers
    // to spill; 65536 elements per chunk keeps partial sums in cache.
    let chunk_size = (65536 / K.next_power_of_two()).min(a_elems / simd_width);
    let num_chunks = (a_elems / simd_width) / chunk_size;

    // ── Rayon parallel path (scalar / implicit) ────────────────────────
    //
    // Identical column-parallel strategy as the AVX-512 rayon path above,
    // but using scalar arithmetic instead of 512-bit SIMD intrinsics.
    //
    // Each element is split into its low and high 32-bit halves so that
    // the u64 multiplication  a[k] * b[k]  is decomposed as:
    //   a_lo * b  (low  32 bits of a)
    //   a_hi * b  (high 32 bits of a)
    // This mirrors the CRT representation used by spiral-rs, where the
    // modulus is a product of two ~32-bit primes.  After accumulation,
    // Barrett reduction + CRT recomposition recovers the result mod q.
    //
    // Also K=1-only, and for the same two reasons as the AVX-512 block:
    // (1) `par_chunks_mut` on the flat `c` buffer only produces valid
    // single-batch column ranges when K=1 (for K>1, chunks straddle the
    // batch-row boundaries of the K*b_cols layout), and (2) the writeback
    // only reads `total_sum_*[0]`, silently discarding batches ≥ 1.
    // Hence the unconditional `assert_eq!(K, 1)` at the top of this fn.
    // Generalizing is tracked in ZCA-256
    // (https://linear.app/zcale/issue/ZCA-256), sub-issue of ZCA-239.
    //
    // Safety / threading notes are the same as the AVX-512 rayon block.
    #[cfg(feature = "rayon")]
    {
        let b_send = SendPtr(b_t.as_ptr());
        let a_send = SendPtr(a.as_ptr());
        let a_len = a.len();
        let num_threads = rayon::current_num_threads();
        // See the matching comment in the AVX-512 path above for why
        // `.max(1)` is needed.
        let cols_per_chunk = ((b_cols + num_threads - 1) / num_threads).max(1);

        // `move` + `SendPtr::get()`: see matching comment in the AVX-512
        // path for why we can't access `.0` directly here.
        c.par_chunks_mut(cols_per_chunk)
            .enumerate()
            .for_each(move |(chunk_idx, c_chunk)| {
                let j_start = chunk_idx * cols_per_chunk;
                unsafe {
                    let b_ptr = b_send.get();
                    let a_slice = std::slice::from_raw_parts(a_send.get(), a_len);
                    let a_slcs = split_a::<K>(a_slice);

                    for k_outer in 0..num_chunks {
                        for (j_local, c_cell) in c_chunk.iter_mut().enumerate() {
                            let j = j_start + j_local;
                            debug_assert!(
                                j < b_cols,
                                "par_chunks_mut partition invariant broken: j={j} >= b_cols={b_cols}"
                            );

                            let mut total_sum_lo = [0u64; K];
                            let mut total_sum_hi = [0u64; K];
                            let mut tmp = [0u64; K];

                            for k_inner in 0..chunk_size {
                                let k = simd_width * (k_outer * chunk_size + k_inner);
                                let b_val_simd = (b_ptr.add(j * b_rows + k)).to_u64();

                                for batch in 0..K {
                                    tmp[batch] = *(a_slcs[batch].as_ptr().add(k));
                                }

                                for batch in 0..K {
                                    let a_val_lo = (tmp[batch] as u32) as u64;
                                    let a_val_hi = ((tmp[batch] >> 32) as u32) as u64;
                                    total_sum_lo[batch] += a_val_lo * b_val_simd;
                                    total_sum_hi[batch] += a_val_hi * b_val_simd;
                                }
                            }

                            // K=1-only path: reduce and write back batch 0.
                            // See the K=1 discussion comment above.
                            writeback(params, c_cell, total_sum_lo[0], total_sum_hi[0]);
                        }
                    }
                }
            });
        return;
    }

    #[cfg(not(feature = "rayon"))]
    unsafe {
        let a_slcs = split_a::<K>(a);
        let b_ptr = b_t.as_ptr();

        for k_outer in 0..num_chunks {
            for j in 0..b_cols {
                let mut total_sum_lo = [0u64; K];
                let mut total_sum_hi = [0u64; K];
                let mut tmp = [0u64; K];

                for k_inner in 0..chunk_size {
                    let k = simd_width * (k_outer * chunk_size + k_inner);
                    let b_val_simd = (b_ptr.add(j * b_rows + k) as *const T).to_u64();

                    for batch in 0..K {
                        tmp[batch] = *(a_slcs[batch].as_ptr().add(k));
                    }

                    for batch in 0..K {
                        let a_val_lo = (tmp[batch] as u32) as u64;
                        let a_val_hi = ((tmp[batch] >> 32) as u32) as u64;
                        total_sum_lo[batch] += a_val_lo * b_val_simd;
                        total_sum_hi[batch] += a_val_hi * b_val_simd;
                    }
                }

                for (batch, c_row) in c.chunks_exact_mut(c.len() / K).enumerate() {
                    writeback(params, &mut c_row[j], total_sum_lo[batch], total_sum_hi[batch]);
                }
            }
        }
    }
}

pub fn scalar_multiply_avx(res: &mut PolyMatrixNTT, a: &PolyMatrixNTT, b: &PolyMatrixNTT) {
    assert_eq!(a.rows, 1);
    assert_eq!(a.cols, 1);

    let params = res.params;
    let pol2 = a.get_poly(0, 0);
    for i in 0..b.rows {
        for j in 0..b.cols {
            let res_poly = res.get_poly_mut(i, j);
            let pol1 = b.get_poly(i, j);
            crate::packing::multiply_poly_avx(params, res_poly, pol1, pol2);
        }
    }
}

pub fn multiply_matrices_raw_not_transposed<T>(
    params: &Params,
    a: &[u64],
    a_rows: usize,
    a_cols: usize,
    b: &[T], // NOT transposed
    b_rows: usize,
    b_cols: usize,
) -> Vec<u64>
where
    T: ToU64 + Copy,
{
    assert_eq!(a_cols, b_rows);

    let mut result = vec![0u128; a_rows * b_cols];

    for i in 0..a_rows {
        for k in 0..a_cols {
            for j in 0..b_cols {
                let a_idx = i * a_cols + k;
                let b_idx = k * b_cols + j;
                let res_idx = i * b_cols + j;

                unsafe {
                    let a_val = *a.get_unchecked(a_idx);
                    let b_val = (*b.get_unchecked(b_idx)).to_u64();

                    let prod = a_val as u128 * b_val as u128;
                    result[res_idx] += prod;
                }
            }
        }
    }

    let mut result_u64 = vec![0u64; a_rows * b_cols];
    for i in 0..result.len() {
        result_u64[i] = barrett_reduction_u128(params, result[i]);
    }

    result_u64
}

#[cfg(test)]
mod test {
    use std::time::Instant;

    use log::debug;
    use spiral_rs::aligned_memory::AlignedMemory64;
    use spiral_rs::poly::*;

    use super::super::util::test_params;
    use super::*;
    use crate::{transpose::*, util::*};
    use test_log::test;

    fn test_fast_batched_dot_product(use_explicit: bool) {
        let params = test_params();

        const A_ROWS: usize = 1;
        let a_cols = 65536;
        let b_rows = a_cols;
        let b_cols = 32768;

        let a = PolyMatrixRaw::random(&params, A_ROWS, a_cols);
        let mut b = AlignedMemory64::new(b_rows * b_cols);
        let mut c = AlignedMemory64::new(A_ROWS * b_cols);
        let trials = 10;
        let mut sum = 0u64;
        let mut sum_time = 0;
        for _ in 0..trials {
            for i in 0..b.len() {
                b[i] = fastrand::u64(..);
            }
            let b_u16_slc =
                unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u16, b.len() * 4) };

            let now = Instant::now();
            // fast_dot_product_avx512(&params, a.as_slice(), rows, b_as_t_slice, rows, cols);
            // fast_dot_product_avx512_fastest(&params, a.as_slice(), rows, b_as_t_slice, rows, cols);
            if use_explicit {
                fast_batched_dot_product_explicit_avx512::<A_ROWS, _>(
                    &params,
                    c.as_mut_slice(),
                    a.as_slice(),
                    a_cols,
                    b_u16_slc,
                    b_rows,
                    b_cols,
                );
            } else {
                fast_batched_dot_product_implicit::<A_ROWS, _>(
                    &params,
                    c.as_mut_slice(),
                    a.as_slice(),
                    a_cols,
                    b_u16_slc,
                    b_rows,
                    b_cols,
                );
            }

            sum_time += now.elapsed().as_micros();
            sum += c.as_slice()[fastrand::usize(..c.len())];
        }
        debug!(
            "fast_matmul_avx512 in {} us ({}: {} x {})",
            sum_time, trials, a_cols, b_cols
        );
        debug!("");
        debug!("{}", sum);
    }

    #[cfg(feature = "explicit_avx512")]
    #[test]
    #[ignore]
    fn test_fast_batched_dot_product_explicit() {
        test_fast_batched_dot_product(true);
    }

    #[test]
    #[ignore]
    fn test_fast_batched_dot_product_implicit() {
        test_fast_batched_dot_product(false);
    }

    #[test]
    fn test_negacyclic_mul_db_col() {
        let params = test_params();
        let pol_a = PolyMatrixRaw::random(&params, 1, 1);
        let pol_b: PolyMatrixRaw<'_> = PolyMatrixRaw::random(&params, 1, 1);
        let a = pol_a.get_poly(0, 0);
        let b = pol_b.get_poly(0, 0);
        let negacylic_a = negacyclic_matrix(&a, params.modulus);
        let negacyclic_a_t = transpose_generic(&negacylic_a, params.poly_len, params.poly_len);

        // the twist is that b is a 'column of the db'
        // we want to compute (b as a row vector) * (transpose(negacylic_a)))

        assert_eq!(negacylic_a[0], a[0]);
        assert_eq!(
            negacylic_a[params.poly_len],
            (params.modulus - a[params.poly_len - 1]) % params.modulus
        );
        let prod = multiply_matrices_raw_not_transposed(
            &params,
            b,
            1,
            params.poly_len,
            &negacyclic_a_t,
            params.poly_len,
            params.poly_len,
        );

        // we think this is equivalent to:
        // poly_b * poly([a_0 -a_d-1 -a_d-2 ... -a_1])
        // = poly_b * poly(negacyclic_perm(a, 0))

        let transformed_a = negacyclic_perm(a, 0, params.modulus);
        let mut pol_a_transformed = PolyMatrixRaw::zero(&params, 1, 1);
        pol_a_transformed
            .data
            .as_mut_slice()
            .copy_from_slice(&transformed_a);
        let pol_c = (&pol_a_transformed.ntt() * &pol_b.ntt()).raw();
        let c = pol_c.get_poly(0, 0);

        for i in 0..params.poly_len {
            assert_eq!(prod[i] % params.modulus, c[i] % params.modulus, "i = {}", i);
        }
    }

    #[test]
    fn test_negacyclic_mul() {
        let params = test_params();
        let pol_a = PolyMatrixRaw::random(&params, 1, 1);
        let pol_b = PolyMatrixRaw::random(&params, 1, 1);
        let a = pol_a.get_poly(0, 0);
        let b = pol_b.get_poly(0, 0);
        let negacylic_a = negacyclic_matrix(&a, params.modulus);
        // let negacylic_a_t = transpose_generic(&negacylic_a, params.poly_len, params.poly_len);
        assert_eq!(negacylic_a[0], a[0]);
        assert_eq!(
            negacylic_a[params.poly_len],
            (params.modulus - a[params.poly_len - 1]) % params.modulus
        );
        let prod = multiply_matrices_raw_not_transposed(
            &params,
            b,
            1,
            params.poly_len,
            &negacylic_a,
            params.poly_len,
            params.poly_len,
        );

        let pol_c = (&pol_a.ntt() * &pol_b.ntt()).raw();
        let c = pol_c.get_poly(0, 0);

        for i in 0..params.poly_len {
            assert_eq!(prod[i] % params.modulus, c[i] % params.modulus, "i = {}", i);
        }
    }

    // ── Reference implementation for correctness testing ─────────────

    /// Reference implementation of the CRT-split dot product.
    ///
    /// Each `a[k]` is CRT-packed: low 32 bits hold the mod-q0 component,
    /// high 32 bits hold the mod-q1 component.  We accumulate the lo and
    /// hi products separately, apply Barrett reduction on each, and
    /// recompose via `crt_compose_2` — exactly mirroring the optimized
    /// kernel logic but using plain u128 arithmetic for clarity.
    fn reference_dot_product_transposed_u16(
        params: &Params,
        c: &mut [u64],
        a: &[u64],
        a_elems: usize,
        b_t: &[u16],
        b_rows: usize,
        b_cols: usize,
    ) {
        assert_eq!(a_elems, b_rows);
        for j in 0..b_cols {
            let mut sum_lo = 0u64;
            let mut sum_hi = 0u64;
            for k in 0..a_elems {
                let a_val = a[k];
                let a_lo = (a_val as u32) as u64;
                let a_hi = ((a_val >> 32) as u32) as u64;
                let b_val = b_t[j * b_rows + k] as u64;
                sum_lo += a_lo * b_val;
                sum_hi += a_hi * b_val;
            }
            let (lo, hi) = (
                barrett_coeff_u64(params, sum_lo, 0),
                barrett_coeff_u64(params, sum_hi, 1),
            );
            let res = params.crt_compose_2(lo, hi);
            c[j] = barrett_u64(params, c[j] + res);
        }
    }

    fn reference_batched_dot_product_u16<const K: usize>(
        params: &Params,
        c: &mut [u64],
        a: &[u64],
        a_elems: usize,
        b_t: &[u16],
        b_rows: usize,
        b_cols: usize,
    ) {
        assert_eq!(a.len(), K * a_elems);
        assert_eq!(c.len(), K * b_cols);
        let a_rows = split_a::<K>(a);

        for (batch, c_row) in c.chunks_exact_mut(b_cols).enumerate() {
            reference_dot_product_transposed_u16(
                params,
                c_row,
                a_rows[batch],
                a_elems,
                b_t,
                b_rows,
                b_cols,
            );
        }
    }

    /// Helper: builds aligned u64 data (needed by AVX-512 which requires
    /// 64-byte alignment for _mm512_load_si512).
    fn random_bounded_aligned(len: usize, bound: u64) -> AlignedMemory64 {
        let mut mem = AlignedMemory64::new(len);
        for i in 0..len {
            mem[i] = fastrand::u64(..) % bound;
        }
        mem
    }

    fn random_u16_vec(len: usize) -> Vec<u16> {
        (0..len).map(|_| fastrand::u16(..)).collect()
    }

    // ── Correctness: implicit matches reference ─────────────────────

    #[test]
    fn test_implicit_matches_reference() {
        let params = test_params();
        let a_elems = 65536;
        let b_cols = 1024;

        let a = random_bounded_aligned(a_elems, params.modulus);
        let b_t_u16 = random_u16_vec(a_elems * b_cols);

        let mut c_ref = vec![0u64; b_cols];
        reference_dot_product_transposed_u16(
            &params,
            &mut c_ref,
            a.as_slice(),
            a_elems,
            &b_t_u16,
            a_elems,
            b_cols,
        );

        let mut c_impl = AlignedMemory64::new(b_cols);
        fast_batched_dot_product_implicit::<1, _>(
            &params,
            c_impl.as_mut_slice(),
            a.as_slice(),
            a_elems,
            &b_t_u16,
            a_elems,
            b_cols,
        );

        for j in 0..b_cols {
            assert_eq!(
                c_impl.as_slice()[j], c_ref[j],
                "mismatch at column {j}: implicit={} ref={}",
                c_impl.as_slice()[j], c_ref[j]
            );
        }
    }

    #[cfg(not(feature = "rayon"))]
    fn assert_implicit_k5_matches_reference(a_elems: usize, b_cols: usize) {
        let params = test_params();
        const K: usize = 5;

        let a = random_bounded_aligned(K * a_elems, params.modulus);
        let b_t_u16 = random_u16_vec(a_elems * b_cols);

        let mut c_ref = vec![0u64; K * b_cols];
        reference_batched_dot_product_u16::<K>(
            &params,
            &mut c_ref,
            a.as_slice(),
            a_elems,
            &b_t_u16,
            a_elems,
            b_cols,
        );

        let mut c_impl = AlignedMemory64::new(K * b_cols);
        fast_batched_dot_product_implicit::<K, _>(
            &params,
            c_impl.as_mut_slice(),
            a.as_slice(),
            a_elems,
            &b_t_u16,
            a_elems,
            b_cols,
        );

        assert_eq!(
            c_impl.as_slice(),
            c_ref.as_slice(),
            "implicit K=5 mismatch (a_elems={a_elems}, b_cols={b_cols})"
        );
    }

    #[cfg(not(feature = "rayon"))]
    #[test]
    fn test_implicit_k5_matches_reference() {
        assert_implicit_k5_matches_reference(65536, 1024);
    }

    #[cfg(not(feature = "rayon"))]
    #[test]
    fn test_implicit_k5_modulus_boundary() {
        let params = test_params();
        let a_elems = 65536;
        let b_cols = 32;
        const K: usize = 5;

        let mut a = AlignedMemory64::new(K * a_elems);
        for i in 0..K * a_elems {
            a[i] = params.modulus - 1;
        }
        let b_t_u16: Vec<u16> = vec![u16::MAX; a_elems * b_cols];

        let mut c_ref = vec![0u64; K * b_cols];
        reference_batched_dot_product_u16::<K>(
            &params,
            &mut c_ref,
            a.as_slice(),
            a_elems,
            &b_t_u16,
            a_elems,
            b_cols,
        );

        let mut c_impl = AlignedMemory64::new(K * b_cols);
        fast_batched_dot_product_implicit::<K, _>(
            &params,
            c_impl.as_mut_slice(),
            a.as_slice(),
            a_elems,
            &b_t_u16,
            a_elems,
            b_cols,
        );

        assert_eq!(c_impl.as_slice(), c_ref.as_slice());
    }

    #[cfg(not(feature = "rayon"))]
    #[test]
    fn test_implicit_k5_single_column() {
        assert_implicit_k5_matches_reference(65536, 1);
    }

    // ── Correctness: accumulation (calling twice doubles) ───────────

    #[test]
    fn test_implicit_accumulates_into_c() {
        let params = test_params();
        let a_elems = 65536;
        let b_cols = 128;

        let a = random_bounded_aligned(a_elems, params.modulus);
        let b_t_u16 = random_u16_vec(a_elems * b_cols);

        let mut c_once = AlignedMemory64::new(b_cols);
        fast_batched_dot_product_implicit::<1, _>(
            &params,
            c_once.as_mut_slice(),
            a.as_slice(),
            a_elems,
            &b_t_u16,
            a_elems,
            b_cols,
        );

        let mut c_twice = AlignedMemory64::new(b_cols);
        fast_batched_dot_product_implicit::<1, _>(
            &params,
            c_twice.as_mut_slice(),
            a.as_slice(),
            a_elems,
            &b_t_u16,
            a_elems,
            b_cols,
        );
        fast_batched_dot_product_implicit::<1, _>(
            &params,
            c_twice.as_mut_slice(),
            a.as_slice(),
            a_elems,
            &b_t_u16,
            a_elems,
            b_cols,
        );

        for j in 0..b_cols {
            let expected = barrett_u64(
                &params,
                c_once.as_slice()[j] + c_once.as_slice()[j],
            );
            assert_eq!(
                c_twice.as_slice()[j], expected,
                "accumulation mismatch at column {j}"
            );
        }
    }

    // ── Edge case: single column ────────────────────────────────────

    #[test]
    fn test_implicit_single_column() {
        let params = test_params();
        let a_elems = 65536;
        let b_cols = 1;

        let a = random_bounded_aligned(a_elems, params.modulus);
        let b_t_u16 = random_u16_vec(a_elems);

        let mut c_ref = vec![0u64; b_cols];
        reference_dot_product_transposed_u16(
            &params,
            &mut c_ref,
            a.as_slice(),
            a_elems,
            &b_t_u16,
            a_elems,
            b_cols,
        );

        let mut c_impl = AlignedMemory64::new(b_cols);
        fast_batched_dot_product_implicit::<1, _>(
            &params,
            c_impl.as_mut_slice(),
            a.as_slice(),
            a_elems,
            &b_t_u16,
            a_elems,
            b_cols,
        );

        assert_eq!(c_impl.as_slice()[0], c_ref[0], "single column mismatch");
    }

    // ── Edge case: all-zero input ───────────────────────────────────

    #[test]
    fn test_implicit_zero_input() {
        let params = test_params();
        let a_elems = 65536;
        let b_cols = 64;

        let a = AlignedMemory64::new(a_elems); // zeros
        let b_t_u16 = random_u16_vec(a_elems * b_cols);

        let mut c = AlignedMemory64::new(b_cols);
        fast_batched_dot_product_implicit::<1, _>(
            &params,
            c.as_mut_slice(),
            a.as_slice(),
            a_elems,
            &b_t_u16,
            a_elems,
            b_cols,
        );

        for j in 0..b_cols {
            assert_eq!(c.as_slice()[j], 0, "expected zero at column {j}");
        }
    }

    // ── Edge case: modulus-boundary values ───────────────────────────

    #[test]
    fn test_implicit_modulus_boundary_values() {
        let params = test_params();
        let a_elems = 65536;
        let b_cols = 32;

        let mut a = AlignedMemory64::new(a_elems);
        for i in 0..a_elems {
            a[i] = params.modulus - 1;
        }
        let b_t_u16: Vec<u16> = vec![u16::MAX; a_elems * b_cols];

        let mut c_ref = vec![0u64; b_cols];
        reference_dot_product_transposed_u16(
            &params,
            &mut c_ref,
            a.as_slice(),
            a_elems,
            &b_t_u16,
            a_elems,
            b_cols,
        );

        let mut c_impl = AlignedMemory64::new(b_cols);
        fast_batched_dot_product_implicit::<1, _>(
            &params,
            c_impl.as_mut_slice(),
            a.as_slice(),
            a_elems,
            &b_t_u16,
            a_elems,
            b_cols,
        );

        for j in 0..b_cols {
            assert_eq!(
                c_impl.as_slice()[j], c_ref[j],
                "modulus-boundary mismatch at column {j}"
            );
        }
    }

    // ── Edge case: small dimension (minimum viable size) ────────────

    #[test]
    fn test_implicit_small_dimension() {
        let params = test_params();
        let a_elems = 65536;
        let b_cols = 2;

        let a = random_bounded_aligned(a_elems, params.modulus);
        let b_t_u16 = random_u16_vec(a_elems * b_cols);

        let mut c_ref = vec![0u64; b_cols];
        reference_dot_product_transposed_u16(
            &params,
            &mut c_ref,
            a.as_slice(),
            a_elems,
            &b_t_u16,
            a_elems,
            b_cols,
        );

        let mut c_impl = AlignedMemory64::new(b_cols);
        fast_batched_dot_product_implicit::<1, _>(
            &params,
            c_impl.as_mut_slice(),
            a.as_slice(),
            a_elems,
            &b_t_u16,
            a_elems,
            b_cols,
        );

        for j in 0..b_cols {
            assert_eq!(
                c_impl.as_slice()[j], c_ref[j],
                "small dimension mismatch at column {j}"
            );
        }
    }

    // ── Rayon tests: parallel must match sequential ─────────────────

    #[cfg(feature = "rayon")]
    mod rayon_tests {
        use super::*;
        use test_log::test;

        /// Runs the same dot product with rayon (parallel) against the
        /// u128 reference implementation and asserts they match.
        fn assert_rayon_matches_reference(a_elems: usize, b_cols: usize) {
            let params = test_params();

            let a = random_bounded_aligned(a_elems, params.modulus);
            let b_t_u16 = random_u16_vec(a_elems * b_cols);

            let mut c_ref = vec![0u64; b_cols];
            reference_dot_product_transposed_u16(
                &params,
                &mut c_ref,
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );

            let mut c_rayon = AlignedMemory64::new(b_cols);
            fast_batched_dot_product_implicit::<1, _>(
                &params,
                c_rayon.as_mut_slice(),
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );

            for j in 0..b_cols {
                assert_eq!(
                    c_rayon.as_slice()[j], c_ref[j],
                    "rayon mismatch at col {j} (a_elems={a_elems}, b_cols={b_cols})"
                );
            }
        }

        #[test]
        fn test_rayon_implicit_standard() {
            assert_rayon_matches_reference(65536, 1024);
        }

        #[test]
        fn test_rayon_implicit_single_column() {
            assert_rayon_matches_reference(65536, 1);
        }

        #[test]
        fn test_rayon_implicit_two_columns() {
            assert_rayon_matches_reference(65536, 2);
        }

        #[test]
        fn test_rayon_implicit_cols_less_than_threads() {
            let num_threads = rayon::current_num_threads();
            if num_threads > 3 {
                assert_rayon_matches_reference(65536, 3);
            }
        }

        #[test]
        fn test_rayon_implicit_non_power_of_two_cols() {
            assert_rayon_matches_reference(65536, 100);
        }

        #[test]
        fn test_rayon_implicit_large() {
            assert_rayon_matches_reference(65536, 32768);
        }

        /// Edge case: `b_cols = 0`.  The rayon path must not panic and
        /// must not write anywhere.  This exercises `par_chunks_mut` on
        /// an empty output, which yields zero chunks — the inner closure
        /// should simply not run.
        #[test]
        fn test_rayon_implicit_b_cols_zero() {
            let params = test_params();
            let a_elems = 65536;
            let b_cols = 0;

            let a = random_bounded_aligned(a_elems, params.modulus);
            let b_t_u16: Vec<u16> = Vec::new();

            // Zero-length output buffer — the kernel must accept this
            // and simply return without touching anything.
            let mut c = AlignedMemory64::new(b_cols.max(1));
            let c_before = c.as_slice()[0];
            fast_batched_dot_product_implicit::<1, _>(
                &params,
                &mut c.as_mut_slice()[..b_cols],
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );
            // Sanity: the one byte past the empty slice we allocated is
            // untouched, so the kernel did not write out of bounds.
            assert_eq!(c.as_slice()[0], c_before);
        }

        /// Edge case: `b_cols` exactly equals `num_threads`.
        ///
        /// With `cols_per_chunk = ceil(b_cols / num_threads) = 1`, every
        /// rayon thread gets exactly one column.  This is the boundary
        /// between "some threads get 2 cols" and "some threads get 0
        /// cols" in the chunk-size arithmetic.
        #[test]
        fn test_rayon_implicit_b_cols_equal_num_threads() {
            let num_threads = rayon::current_num_threads();
            assert_rayon_matches_reference(65536, num_threads);
        }

        #[test]
        fn test_rayon_implicit_accumulates() {
            let params = test_params();
            let a_elems = 65536;
            let b_cols = 256;

            let a = random_bounded_aligned(a_elems, params.modulus);
            let b_t_u16 = random_u16_vec(a_elems * b_cols);

            let mut c_once = AlignedMemory64::new(b_cols);
            fast_batched_dot_product_implicit::<1, _>(
                &params,
                c_once.as_mut_slice(),
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );

            let mut c_twice = AlignedMemory64::new(b_cols);
            fast_batched_dot_product_implicit::<1, _>(
                &params,
                c_twice.as_mut_slice(),
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );
            fast_batched_dot_product_implicit::<1, _>(
                &params,
                c_twice.as_mut_slice(),
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );

            for j in 0..b_cols {
                let expected = barrett_u64(
                    &params,
                    c_once.as_slice()[j] + c_once.as_slice()[j],
                );
                assert_eq!(
                    c_twice.as_slice()[j], expected,
                    "rayon accumulation mismatch at column {j}"
                );
            }
        }

        #[test]
        fn test_rayon_implicit_zero_a() {
            let params = test_params();
            let a_elems = 65536;
            let b_cols = 64;

            let a = AlignedMemory64::new(a_elems);
            let b_t_u16 = random_u16_vec(a_elems * b_cols);

            let mut c = AlignedMemory64::new(b_cols);
            fast_batched_dot_product_implicit::<1, _>(
                &params,
                c.as_mut_slice(),
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );

            for j in 0..b_cols {
                assert_eq!(c.as_slice()[j], 0, "expected zero at column {j}");
            }
        }

        #[test]
        fn test_rayon_implicit_modulus_boundary() {
            let params = test_params();
            let a_elems = 65536;
            let b_cols = 32;

            let mut a = AlignedMemory64::new(a_elems);
            for i in 0..a_elems {
                a[i] = params.modulus - 1;
            }
            let b_t_u16: Vec<u16> = vec![u16::MAX; a_elems * b_cols];

            let mut c_ref = vec![0u64; b_cols];
            reference_dot_product_transposed_u16(
                &params,
                &mut c_ref,
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );

            let mut c_rayon = AlignedMemory64::new(b_cols);
            fast_batched_dot_product_implicit::<1, _>(
                &params,
                c_rayon.as_mut_slice(),
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );

            for j in 0..b_cols {
                assert_eq!(
                    c_rayon.as_slice()[j], c_ref[j],
                    "rayon modulus-boundary mismatch at column {j}"
                );
            }
        }

        /// Verify determinism: running the same computation twice must
        /// produce identical results despite non-deterministic thread
        /// scheduling.
        #[test]
        fn test_rayon_deterministic() {
            let params = test_params();
            let a_elems = 65536;
            let b_cols = 512;

            let a = random_bounded_aligned(a_elems, params.modulus);
            let b_t_u16 = random_u16_vec(a_elems * b_cols);

            let mut c1 = AlignedMemory64::new(b_cols);
            let mut c2 = AlignedMemory64::new(b_cols);

            fast_batched_dot_product_implicit::<1, _>(
                &params,
                c1.as_mut_slice(),
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );
            fast_batched_dot_product_implicit::<1, _>(
                &params,
                c2.as_mut_slice(),
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );

            assert_eq!(c1.as_slice(), c2.as_slice(), "rayon results not deterministic");
        }

        /// Concurrent invocations must not corrupt each other.
        ///
        /// A PIR server pins one kernel call per inflight request and
        /// can be servicing many queries at once, so the rayon kernel
        /// is re-entered from multiple OS threads concurrently.  Rayon
        /// handles this by running nested `par_iter` work on a shared
        /// global pool, but any subtle shared-state bug (e.g. a helper
        /// accidentally capturing a `static mut`, or a future rayon
        /// upgrade interacting badly with nested pools) would show up
        /// as cross-request data corruption — a catastrophic failure
        /// mode for PIR where the client cannot detect it.
        ///
        /// We spawn N std::threads, each driving the kernel with its
        /// own independent `a` / `b_t` / `c` buffers, and assert every
        /// thread's result matches a pre-computed reference.  No kernel
        /// invocation sees another invocation's inputs, so any
        /// mismatch is unambiguously a concurrency bug.
        #[test]
        fn test_rayon_concurrent_invocations() {
            use std::sync::Arc;
            use std::thread;

            let params = Arc::new(test_params());
            let a_elems = 65536;
            let b_cols = 1024;
            // Enough threads to exceed rayon's default pool fan-out and
            // force contention for worker threads between invocations.
            let num_drivers = 8;

            // Pre-compute each driver's reference on the main thread so
            // concurrent test logic can stay focused on the kernel.
            let inputs: Vec<_> = (0..num_drivers)
                .map(|seed| {
                    fastrand::seed(seed as u64);
                    let a = random_bounded_aligned(a_elems, params.modulus);
                    let b_t_u16 = random_u16_vec(a_elems * b_cols);
                    let mut c_ref = vec![0u64; b_cols];
                    reference_dot_product_transposed_u16(
                        &params,
                        &mut c_ref,
                        a.as_slice(),
                        a_elems,
                        &b_t_u16,
                        a_elems,
                        b_cols,
                    );
                    // AlignedMemory64 isn't Send, but we only need the
                    // underlying bytes in the worker threads, so copy
                    // into a plain Vec.  Each worker allocates a fresh
                    // AlignedMemory64 for the kernel's alignment needs.
                    (a.as_slice().to_vec(), b_t_u16, c_ref)
                })
                .collect();

            let handles: Vec<_> = inputs
                .into_iter()
                .enumerate()
                .map(|(idx, (a_vec, b_t_u16, c_ref))| {
                    let params = Arc::clone(&params);
                    thread::spawn(move || {
                        // Copy `a` into aligned memory inside the worker.
                        let mut a = AlignedMemory64::new(a_vec.len());
                        a.as_mut_slice().copy_from_slice(&a_vec);

                        let mut c = AlignedMemory64::new(b_cols);
                        fast_batched_dot_product_implicit::<1, _>(
                            &params,
                            c.as_mut_slice(),
                            a.as_slice(),
                            a_elems,
                            &b_t_u16,
                            a_elems,
                            b_cols,
                        );
                        for j in 0..b_cols {
                            assert_eq!(
                                c.as_slice()[j], c_ref[j],
                                "driver {idx}: concurrent mismatch at col {j}"
                            );
                        }
                    })
                })
                .collect();

            for h in handles {
                h.join().expect("concurrent kernel driver panicked");
            }
        }
    }

    // ── AVX-512 tests: hand-written intrinsics must match reference ──
    //
    // These tests require the `explicit_avx512` Cargo feature and an
    // AVX-512F-capable CPU at runtime.  They cover the hand-written
    // `fast_batched_dot_product_explicit_avx512` kernel, which on CPUs
    // with AVX-512 is the production hot path.
    //
    // Two code paths exist inside that function and both are covered:
    //   - K == 1, `rayon` feature enabled: the parallel `par_chunks_mut`
    //     path that splits `b_cols` across threads.
    //   - K >= 1 without rayon *or* K > 1 with rayon: the sequential
    //     column loop.  The K=2 test forces this path even when `rayon`
    //     is enabled, because the K=1 rayon gate only fires for K=1.
    //
    // Oracles:
    //   - `reference_dot_product_transposed_u16` is the u128-precise
    //     scalar reference — the authoritative oracle for numerical
    //     correctness.
    //   - `fast_batched_dot_product_implicit` is the scalar kernel used
    //     on non-AVX-512 targets.  AVX-512 must bit-exactly agree with
    //     it, which is a stronger invariant than matching the reference
    //     (both must agree *and* agree with the reference).
    //
    // Running locally:
    //   cargo test --release --features explicit_avx512,rayon \
    //       kernel::test::avx512_tests
    //
    // These tests panic with "illegal instruction" rather than failing
    // gracefully if run on a CPU without AVX-512F.  The assumption is
    // that anyone enabling `explicit_avx512` knows their target.
    #[cfg(feature = "explicit_avx512")]
    mod avx512_tests {
        use super::*;
        use test_log::test;

        /// Runs the AVX-512 kernel and asserts bit-exact agreement with
        /// the u128 reference.  This is the primary correctness oracle.
        fn assert_avx512_matches_reference(a_elems: usize, b_cols: usize) {
            let params = test_params();

            let a = random_bounded_aligned(a_elems, params.modulus);
            let b_t_u16 = random_u16_vec(a_elems * b_cols);

            let mut c_ref = vec![0u64; b_cols];
            reference_dot_product_transposed_u16(
                &params,
                &mut c_ref,
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );

            let mut c_avx = AlignedMemory64::new(b_cols);
            fast_batched_dot_product_explicit_avx512::<1, _>(
                &params,
                c_avx.as_mut_slice(),
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );

            for j in 0..b_cols {
                assert_eq!(
                    c_avx.as_slice()[j], c_ref[j],
                    "AVX-512 vs reference mismatch at col {j} \
                     (a_elems={a_elems}, b_cols={b_cols})"
                );
            }
        }

        /// Runs both AVX-512 and scalar kernels on identical inputs and
        /// asserts bit-exact agreement.  Guards against regressions where
        /// one path gets updated but the other does not.
        fn assert_avx512_matches_implicit(a_elems: usize, b_cols: usize) {
            let params = test_params();

            let a = random_bounded_aligned(a_elems, params.modulus);
            let b_t_u16 = random_u16_vec(a_elems * b_cols);

            let mut c_avx = AlignedMemory64::new(b_cols);
            fast_batched_dot_product_explicit_avx512::<1, _>(
                &params,
                c_avx.as_mut_slice(),
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );

            let mut c_scalar = AlignedMemory64::new(b_cols);
            fast_batched_dot_product_implicit::<1, _>(
                &params,
                c_scalar.as_mut_slice(),
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );

            assert_eq!(
                c_avx.as_slice(),
                c_scalar.as_slice(),
                "AVX-512 and scalar kernels disagree (a_elems={a_elems}, b_cols={b_cols})"
            );
        }

        #[cfg(not(feature = "rayon"))]
        fn assert_avx512_k5_matches_reference(a_elems: usize, b_cols: usize) {
            let params = test_params();
            const K: usize = 5;

            let a = random_bounded_aligned(K * a_elems, params.modulus);
            let b_t_u16 = random_u16_vec(a_elems * b_cols);

            let mut c_ref = vec![0u64; K * b_cols];
            reference_batched_dot_product_u16::<K>(
                &params,
                &mut c_ref,
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );

            let mut c_avx = AlignedMemory64::new(K * b_cols);
            fast_batched_dot_product_explicit_avx512::<K, _>(
                &params,
                c_avx.as_mut_slice(),
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );

            assert_eq!(
                c_avx.as_slice(),
                c_ref.as_slice(),
                "AVX-512 K=5 mismatch (a_elems={a_elems}, b_cols={b_cols})"
            );
        }

        #[cfg(feature = "rayon")]
        fn assert_avx512_rayon_k5_matches_reference(a_elems: usize, b_cols: usize) {
            let params = test_params();
            const K: usize = 5;

            let a = random_bounded_aligned(K * a_elems, params.modulus);
            let b_t_u16 = random_u16_vec(a_elems * b_cols);

            let mut c_ref = vec![0u64; K * b_cols];
            reference_batched_dot_product_u16::<K>(
                &params,
                &mut c_ref,
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );

            let mut c_avx = AlignedMemory64::new(K * b_cols);
            fast_batched_dot_product_explicit_avx512::<K, _>(
                &params,
                c_avx.as_mut_slice(),
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );

            assert_eq!(
                c_avx.as_slice(),
                c_ref.as_slice(),
                "rayon AVX-512 K=5 mismatch (a_elems={a_elems}, b_cols={b_cols})"
            );
        }

        #[test]
        fn test_avx512_standard() {
            assert_avx512_matches_reference(65536, 1024);
        }

        #[cfg(feature = "rayon")]
        #[test]
        fn test_avx512_rayon_k5_matches_reference() {
            assert_avx512_rayon_k5_matches_reference(65536, 1024);
        }

        #[cfg(feature = "rayon")]
        #[test]
        fn test_avx512_rayon_k5_non_power_of_two_cols() {
            assert_avx512_rayon_k5_matches_reference(65536, 100);
        }

        #[cfg(feature = "rayon")]
        #[test]
        fn test_avx512_rayon_k5_single_column() {
            assert_avx512_rayon_k5_matches_reference(65536, 1);
        }

        #[cfg(feature = "rayon")]
        #[test]
        fn test_avx512_rayon_k5_cols_less_than_threads() {
            let num_threads = rayon::current_num_threads();
            if num_threads > 3 {
                assert_avx512_rayon_k5_matches_reference(65536, 3);
            }
        }

        #[cfg(feature = "rayon")]
        #[test]
        fn test_avx512_rayon_k5_accumulates() {
            let params = test_params();
            let a_elems = 65536;
            let b_cols = 100;
            const K: usize = 5;

            let a = random_bounded_aligned(K * a_elems, params.modulus);
            let b_t_u16 = random_u16_vec(a_elems * b_cols);

            let mut c_initial = vec![0u64; K * b_cols];
            for i in 0..K * b_cols {
                c_initial[i] = (i as u64 * 17) % params.modulus;
            }
            let mut c_ref = c_initial.clone();
            reference_batched_dot_product_u16::<K>(
                &params,
                &mut c_ref,
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );

            let mut c_rayon = AlignedMemory64::new(K * b_cols);
            c_rayon.as_mut_slice().copy_from_slice(&c_initial);
            fast_batched_dot_product_explicit_avx512::<K, _>(
                &params,
                c_rayon.as_mut_slice(),
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );

            assert_eq!(c_rayon.as_slice(), c_ref.as_slice());
        }

        #[cfg(not(feature = "rayon"))]
        #[test]
        fn test_avx512_k5_matches_reference() {
            assert_avx512_k5_matches_reference(65536, 1024);
        }

        #[cfg(not(feature = "rayon"))]
        #[test]
        fn test_avx512_k5_modulus_boundary() {
            let params = test_params();
            let a_elems = 65536;
            let b_cols = 32;
            const K: usize = 5;

            let mut a = AlignedMemory64::new(K * a_elems);
            for i in 0..K * a_elems {
                a[i] = params.modulus - 1;
            }
            let b_t_u16: Vec<u16> = vec![u16::MAX; a_elems * b_cols];

            let mut c_ref = vec![0u64; K * b_cols];
            reference_batched_dot_product_u16::<K>(
                &params,
                &mut c_ref,
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );

            let mut c_avx = AlignedMemory64::new(K * b_cols);
            fast_batched_dot_product_explicit_avx512::<K, _>(
                &params,
                c_avx.as_mut_slice(),
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );

            assert_eq!(c_avx.as_slice(), c_ref.as_slice());
        }

        #[cfg(not(feature = "rayon"))]
        #[test]
        fn test_avx512_k5_single_column() {
            assert_avx512_k5_matches_reference(65536, 1);
        }

        /// Exercises the inner loop with a single output column.  On the
        /// rayon path this also means only one chunk is produced.
        #[test]
        fn test_avx512_single_column() {
            assert_avx512_matches_reference(65536, 1);
        }

        /// Small `b_cols` — every thread on the rayon path gets at most
        /// one column, and the sequential path iterates twice.
        #[test]
        fn test_avx512_two_columns() {
            assert_avx512_matches_reference(65536, 2);
        }

        /// Non-power-of-two `b_cols` catches off-by-one errors in chunk
        /// sizing and tail handling.
        #[test]
        fn test_avx512_non_power_of_two_cols() {
            assert_avx512_matches_reference(65536, 100);
        }

        #[test]
        fn test_avx512_large() {
            assert_avx512_matches_reference(65536, 32768);
        }

        /// Edge case: `b_cols = 0` must not panic or write anywhere.
        #[test]
        fn test_avx512_b_cols_zero() {
            let params = test_params();
            let a_elems = 65536;
            let b_cols = 0;

            let a = random_bounded_aligned(a_elems, params.modulus);
            let b_t_u16: Vec<u16> = Vec::new();

            let mut c = AlignedMemory64::new(b_cols.max(1));
            let c_before = c.as_slice()[0];
            fast_batched_dot_product_explicit_avx512::<1, _>(
                &params,
                &mut c.as_mut_slice()[..b_cols],
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );
            assert_eq!(c.as_slice()[0], c_before);
        }

        /// Verifies that the kernel accumulates (`c += A · B^T`) rather
        /// than overwriting.  The CRT-reduced double result must equal
        /// what you get from running the kernel twice into the same
        /// buffer.
        #[test]
        fn test_avx512_accumulates() {
            let params = test_params();
            let a_elems = 65536;
            let b_cols = 256;

            let a = random_bounded_aligned(a_elems, params.modulus);
            let b_t_u16 = random_u16_vec(a_elems * b_cols);

            let mut c_once = AlignedMemory64::new(b_cols);
            fast_batched_dot_product_explicit_avx512::<1, _>(
                &params,
                c_once.as_mut_slice(),
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );

            let mut c_twice = AlignedMemory64::new(b_cols);
            for _ in 0..2 {
                fast_batched_dot_product_explicit_avx512::<1, _>(
                    &params,
                    c_twice.as_mut_slice(),
                    a.as_slice(),
                    a_elems,
                    &b_t_u16,
                    a_elems,
                    b_cols,
                );
            }

            for j in 0..b_cols {
                let expected = barrett_u64(
                    &params,
                    c_once.as_slice()[j] + c_once.as_slice()[j],
                );
                assert_eq!(
                    c_twice.as_slice()[j], expected,
                    "AVX-512 accumulation mismatch at column {j}"
                );
            }
        }

        /// All-zero `a` — every column must be exactly 0.  Trivially
        /// catches any spurious non-zero writes from the reduction /
        /// writeback path.
        #[test]
        fn test_avx512_zero_a() {
            let params = test_params();
            let a_elems = 65536;
            let b_cols = 64;

            let a = AlignedMemory64::new(a_elems);
            let b_t_u16 = random_u16_vec(a_elems * b_cols);

            let mut c = AlignedMemory64::new(b_cols);
            fast_batched_dot_product_explicit_avx512::<1, _>(
                &params,
                c.as_mut_slice(),
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );

            for j in 0..b_cols {
                assert_eq!(c.as_slice()[j], 0, "expected zero at column {j}");
            }
        }

        /// Maximum-magnitude inputs.  Exercises the overflow boundaries
        /// of the 32x32 → 64-bit multiplies and the subsequent Barrett
        /// reductions.  Any mis-ordered addition or missed reduction
        /// will show up as a mismatch with the u128 reference.
        #[test]
        fn test_avx512_modulus_boundary() {
            let params = test_params();
            let a_elems = 65536;
            let b_cols = 32;

            let mut a = AlignedMemory64::new(a_elems);
            for i in 0..a_elems {
                a[i] = params.modulus - 1;
            }
            let b_t_u16: Vec<u16> = vec![u16::MAX; a_elems * b_cols];

            let mut c_ref = vec![0u64; b_cols];
            reference_dot_product_transposed_u16(
                &params,
                &mut c_ref,
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );

            let mut c_avx = AlignedMemory64::new(b_cols);
            fast_batched_dot_product_explicit_avx512::<1, _>(
                &params,
                c_avx.as_mut_slice(),
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );

            for j in 0..b_cols {
                assert_eq!(
                    c_avx.as_slice()[j], c_ref[j],
                    "AVX-512 modulus-boundary mismatch at column {j}"
                );
            }
        }

        /// Determinism: repeated invocations on identical inputs must
        /// produce bit-identical outputs.  The rayon path's thread
        /// schedule is non-deterministic, but since each thread writes
        /// to a disjoint output slice and accumulation within a column
        /// is ordered, the result must be stable across runs.
        #[test]
        fn test_avx512_deterministic() {
            let params = test_params();
            let a_elems = 65536;
            let b_cols = 512;

            let a = random_bounded_aligned(a_elems, params.modulus);
            let b_t_u16 = random_u16_vec(a_elems * b_cols);

            let mut c1 = AlignedMemory64::new(b_cols);
            let mut c2 = AlignedMemory64::new(b_cols);

            fast_batched_dot_product_explicit_avx512::<1, _>(
                &params,
                c1.as_mut_slice(),
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );
            fast_batched_dot_product_explicit_avx512::<1, _>(
                &params,
                c2.as_mut_slice(),
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );

            assert_eq!(
                c1.as_slice(),
                c2.as_slice(),
                "AVX-512 results not deterministic"
            );
        }

        /// Bit-exact cross-check against the scalar kernel.  Stronger
        /// than matching the reference alone — any divergence between
        /// the two production paths will show up here.
        #[test]
        fn test_avx512_matches_scalar_standard() {
            assert_avx512_matches_implicit(65536, 1024);
        }

        #[test]
        fn test_avx512_matches_scalar_single_column() {
            assert_avx512_matches_implicit(65536, 1);
        }

        #[test]
        fn test_avx512_matches_scalar_non_power_of_two() {
            assert_avx512_matches_implicit(65536, 100);
        }

        /// K > 1 sequential path.  Even when the `rayon` feature is
        /// enabled, the AVX-512 kernel falls through to the sequential
        /// column loop for K > 1 (the rayon path is currently K=1 only,
        /// tracked in ZCA-256).  This test exercises that path.
        #[test]
        fn test_avx512_k2_sequential() {
            let params = test_params();
            let a_elems = 65536;
            let b_cols = 256;
            const K: usize = 2;

            // Two query rows, concatenated: `a = [query0, query1]`.
            let a0 = random_bounded_aligned(a_elems, params.modulus);
            let a1 = random_bounded_aligned(a_elems, params.modulus);
            let mut a = AlignedMemory64::new(K * a_elems);
            a.as_mut_slice()[..a_elems].copy_from_slice(a0.as_slice());
            a.as_mut_slice()[a_elems..].copy_from_slice(a1.as_slice());

            let b_t_u16 = random_u16_vec(a_elems * b_cols);

            // Reference: run K=1 kernel twice, one per batch row.
            let mut c_ref0 = vec![0u64; b_cols];
            let mut c_ref1 = vec![0u64; b_cols];
            reference_dot_product_transposed_u16(
                &params,
                &mut c_ref0,
                a0.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );
            reference_dot_product_transposed_u16(
                &params,
                &mut c_ref1,
                a1.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );

            // AVX-512 K=2: output is K*b_cols = 2*b_cols, laid out as
            // [batch0_col0..batch0_col_{b_cols-1}, batch1_col0..].
            let mut c_avx = AlignedMemory64::new(K * b_cols);
            fast_batched_dot_product_explicit_avx512::<K, _>(
                &params,
                c_avx.as_mut_slice(),
                a.as_slice(),
                a_elems,
                &b_t_u16,
                a_elems,
                b_cols,
            );

            for j in 0..b_cols {
                assert_eq!(
                    c_avx.as_slice()[j], c_ref0[j],
                    "AVX-512 K=2 batch 0 mismatch at column {j}"
                );
                assert_eq!(
                    c_avx.as_slice()[b_cols + j], c_ref1[j],
                    "AVX-512 K=2 batch 1 mismatch at column {j}"
                );
            }
        }
    }
}
