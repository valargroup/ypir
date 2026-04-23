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
/// slices.  Caller must ensure `a.len() % K == 0`; assertion is implicit in
/// `chunks_exact`.
#[inline(always)]
fn split_a<const K: usize>(a: &[u64]) -> [&[u64]; K] {
    let mut out: [&[u64]; K] = [&[]; K];
    for (slot, chunk) in out.iter_mut().zip(a.chunks_exact(a.len() / K)) {
        *slot = chunk;
    }
    out
}

/// Finalize a single output cell from its two CRT half-sums.
///
/// Applies Barrett reduction to each half (mod q0, mod q1), recomposes via
/// CRT into a single element mod q, and accumulates into `*c_cell` with a
/// final Barrett reduction so the stored value stays in `[0, q)`.
///
/// This is the common tail of both the scalar and AVX-512 kernels.
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
/// # Safety
/// Caller must be running on a CPU with AVX-512F.  The `__m512i` arguments
/// are produced by `_mm512_setzero_si512` and subsequent
/// `_mm512_add_epi64` / `_mm512_mul_epu32` calls; this helper only reads
/// them via `_mm512_store_si512`.
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
    _mm512_store_si512(vl.as_mut_ptr() as *mut _, sum_lo);
    _mm512_store_si512(vh.as_mut_ptr() as *mut _, sum_hi);
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
    // The matrix-vector product  c = a · B^T  iterates over b_cols output
    // columns.  Each column j produces an independent dot product:
    //   c[j] = Σ_k  a[k] * b_t[j * b_rows + k]
    //
    // Because columns are independent, we can split them across threads
    // with zero synchronization.  `par_chunks_mut` gives each rayon thread
    // an exclusive &mut slice of the output buffer `c`, so no locks or
    // atomics are needed.
    //
    // Restricted to K=1 (the only value used in SimplePIR's online phase).
    // The limitation is in how `c` is partitioned across threads, not in
    // the arithmetic — the non-rayon path below is already K-generic.
    // Two concrete reasons today's code only works for K=1:
    //
    //   1. Output layout vs. `par_chunks_mut`.  `c` is a flat buffer of
    //      length K*b_cols, laid out as K batch-rows concatenated:
    //        [batch0_col0 .. batch0_col_{b_cols-1},
    //         batch1_col0 .. batch1_col_{b_cols-1}, …]
    //      For K=1, `c.len() == b_cols`, so `par_chunks_mut(cols_per_chunk)`
    //      produces chunks that are exactly column ranges of the one
    //      output row, and the inner `j = j_start + j_local` indexing
    //      into `b_t[j * b_rows + k]` is correct.  For K>1, the flat
    //      chunks can straddle the batch-row boundary (a chunk may end
    //      inside batch 0 and continue into batch 1), so `j_local` no
    //      longer maps 1-to-1 onto a single database column for a single
    //      batch row.  A K-generic rayon path has to partition differently
    //      (e.g. split `c` into K disjoint sub-slices per thread via raw
    //      pointer math, mirroring the `a`/`b_t` address-by-usize trick
    //      already used here).
    //
    //   2. Reduction hard-codes batch 0.  The inner loop computes
    //      `total_sum_lo[K]` / `total_sum_hi[K]` correctly for every
    //      batch, but the horizontal reduction and writeback below only
    //      reads `total_sum_*[0]` and stores into `c_chunk[j_local]`.
    //      Batches ≥ 1 are computed and silently discarded.  K-generic
    //      support requires looping `for batch in 0..K` around the
    //      reduction and writing to the per-batch output sub-slice.
    //
    // Design note: the right fix is column-partitioned / batch-broadcast
    // (thread owns a column range, computes all K batches within it).
    // This preserves the whole point of batching — `b_t` is streamed from
    // DRAM once and reused K times in registers, so arithmetic intensity
    // scales with K.  The naive alternative (partition threads by batch)
    // would K× the DRAM traffic and cap parallelism at K.  Tracked in
    // ZCA-256 (https://linear.app/zcale/issue/ZCA-256), sub-issue of
    // ZCA-239 (PIR batch query).
    //
    // Safety of the pointer casts:
    //   - `a` and `b_t` are read-only for the entire duration of the
    //     parallel region; the caller holds &[u64] / &[T] borrows.
    //   - Each thread writes only to its own `c_chunk` (disjoint slices).
    //   - Raw pointers are cast via usize to satisfy Send/Sync bounds on
    //     the rayon closure; the underlying data outlives the closure.
    //
    // Edge cases:
    //   - b_cols not divisible by num_threads: the last chunk produced by
    //     `par_chunks_mut` is shorter; the `j >= b_cols` guard is
    //     redundant but kept as a safety net.
    //   - b_cols < num_threads: some chunks are empty (par_chunks_mut
    //     simply produces fewer chunks).
    #[cfg(feature = "rayon")]
    if K == 1 {
        let b_addr: usize = b_t.as_ptr() as usize;
        let a_addr: usize = a.as_ptr() as usize;
        let a_len = a.len();
        let num_threads = rayon::current_num_threads();
        // `.max(1)` so `par_chunks_mut` never gets a zero chunk size — it
        // panics on 0.  When b_cols == 0 the slice is empty, so any
        // positive chunk size yields zero chunks and the closure is a
        // no-op, which is the correct result.
        let cols_per_chunk = ((b_cols + num_threads - 1) / num_threads).max(1);

        c.par_chunks_mut(cols_per_chunk)
            .enumerate()
            .for_each(|(chunk_idx, c_chunk)| {
                let j_start = chunk_idx * cols_per_chunk;
                unsafe {
                    let b_ptr = b_addr as *const T;
                    let a_slice = std::slice::from_raw_parts(a_addr as *const u64, a_len);
                    let a_slcs = split_a::<K>(a_slice);

                    // Outer loop over row chunks (accumulation dimension).
                    // Each k_outer processes `chunk_size` groups of 8 (simd_width)
                    // elements, accumulating partial sums in AVX-512 registers.
                    for k_outer in 0..num_chunks {
                        for (j_local, c_cell) in c_chunk.iter_mut().enumerate() {
                            let j = j_start + j_local;
                            if j >= b_cols {
                                break;
                            }

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

                            // K=1-only path: reduce and write back batch 0.
                            // See the K=1 discussion comment above.
                            writeback_avx512(params, c_cell, total_sum_lo[0], total_sum_hi[0]);
                        }
                    }
                }
            });
        return;
    }

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
/// Only supports K=1.
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
        let b_addr: usize = b_t.as_ptr() as usize;
        let a_addr: usize = a.as_ptr() as usize;
        let a_len = a.len();
        let num_threads = rayon::current_num_threads();
        // See the matching comment in the AVX-512 path above for why
        // `.max(1)` is needed.
        let cols_per_chunk = ((b_cols + num_threads - 1) / num_threads).max(1);

        c.par_chunks_mut(cols_per_chunk)
            .enumerate()
            .for_each(|(chunk_idx, c_chunk)| {
                let j_start = chunk_idx * cols_per_chunk;
                unsafe {
                    let b_ptr = b_addr as *const T;
                    let a_slice = std::slice::from_raw_parts(a_addr as *const u64, a_len);
                    let a_slcs = split_a::<K>(a_slice);

                    for k_outer in 0..num_chunks {
                        for (j_local, c_cell) in c_chunk.iter_mut().enumerate() {
                            let j = j_start + j_local;
                            if j >= b_cols {
                                break;
                            }

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
    }
}
