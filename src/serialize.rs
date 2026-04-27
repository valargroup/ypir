use std::{
    fs::File,
    io::{self, BufReader, Read, Seek, Write},
};

use spiral_rs::{aligned_memory::AlignedMemory64, params::*, poly::*, util::read_arbitrary_bits};

use crate::client::{YPIRQuery, YPIRSimpleQuery};

pub type Precomp<'a> = Vec<(PolyMatrixNTT<'a>, Vec<PolyMatrixNTT<'a>>, Vec<Vec<usize>>)>;

// ── Cache I/O (warm-restart precompute cache) ────────────────────────────────
//
// Checked binary dump/load for `OfflinePrecomputedValues` and (in `server.rs`)
// `YServer<u16>`. Unlike the existing `ToBytes`/`FromBytesParams` helpers
// above (which use unchecked slicing safe only for trusted in-process bytes),
// this API is contractually safe for disk-loaded input that may be truncated,
// partially overwritten, or corrupted. The reader bounds-checks every access
// and returns a typed `CacheError` instead of panicking.
//
// Format conventions:
//   - all integers LE
//   - all variable-length sections preceded by `u64 LE` length
//   - `usize` is never serialized; converted to/from `u64 LE`
//   - binary-stable within a major version; bumping `valar-ypir`'s major
//     version may change the layout
//
// This API is for warm-restart caching only. It is not intended for cross-
// process or network transfer; the cache is tied to specific build flags
// (CPU features) and YPIR `Params`. The consumer is expected to wrap the
// payload in its own header containing those identity fields.

/// Errors returned by `OfflinePrecomputedValues::load_from` /
/// `YServer::load_from`. `Io` wraps unexpected I/O errors; `Truncated` and
/// `Malformed` indicate the cache file itself is unusable.
#[cfg(feature = "server")]
#[derive(Debug)]
pub enum CacheError {
    /// Reader returned EOF before the expected number of bytes had been read,
    /// or a length prefix would require more bytes than the source can supply.
    Truncated {
        what: &'static str,
        needed: usize,
        got: usize,
    },
    /// A field violated an internal invariant (e.g. dimensions inconsistent
    /// with `Params`, `usize` value out of range, unknown payload version).
    Malformed { what: &'static str, detail: String },
    /// Underlying reader error (not EOF).
    Io(io::Error),
}

#[cfg(feature = "server")]
impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CacheError::Truncated { what, needed, got } => {
                write!(f, "cache truncated: {what} needed {needed} bytes, got {got}")
            }
            CacheError::Malformed { what, detail } => {
                write!(f, "cache malformed: {what}: {detail}")
            }
            CacheError::Io(e) => write!(f, "cache I/O error: {e}"),
        }
    }
}

#[cfg(feature = "server")]
impl std::error::Error for CacheError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CacheError::Io(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(feature = "server")]
impl From<io::Error> for CacheError {
    fn from(e: io::Error) -> Self {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            CacheError::Truncated { what: "reader", needed: 0, got: 0 }
        } else {
            CacheError::Io(e)
        }
    }
}

/// Payload format version embedded in the dump. Bump on any change that
/// breaks cache validity, including:
///   - Wire-format changes (added/removed/reordered fields, new encoding)
///   - Algorithm changes that produce different bytes for the same input
///     (e.g., RNG seeding, polynomial precomputation order) without changing
///     the wire format
///
/// The loader rejects unknown values, so consumers don't need to track the
/// crate version separately. If the dump bytes for the same input change in
/// any way, bump this. The loader rejects unknown values.
#[cfg(feature = "server")]
const PAYLOAD_FORMAT_V1: u8 = 1;

// The cache I/O bulk readers and writers reinterpret `&[u64]` as `&[u8]` via
// raw pointer cast (see `write_u64_slice` / `read_u64_slice` below) for the
// multi-GB `db_buf_aligned` and `hint_0` transfers. The on-wire format is
// documented as little-endian; on a big-endian target the in-memory bytes
// would not match, so the dump bytes would silently disagree with the docs.
// Hard-fail at compile time rather than ship a silently-broken format. If
// big-endian support is ever needed, replace the bulk transmute with an
// explicit byteswap loop and remove this guard.
#[cfg(all(feature = "server", not(target_endian = "little")))]
compile_error!(
    "valar-ypir cache I/O assumes a little-endian target. Build on x86_64, \
     aarch64, or another LE target."
);

#[cfg(feature = "server")]
pub(crate) mod cache_io {
    use super::*;

    pub(crate) fn write_u8<W: Write>(w: &mut W, v: u8) -> io::Result<()> {
        w.write_all(&[v])
    }

    pub(crate) fn write_u32_le<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
        w.write_all(&v.to_le_bytes())
    }

    pub(crate) fn write_u64_le<W: Write>(w: &mut W, v: u64) -> io::Result<()> {
        w.write_all(&v.to_le_bytes())
    }

    pub(crate) fn read_u8<R: Read>(r: &mut R, what: &'static str) -> Result<u8, CacheError> {
        let mut buf = [0u8; 1];
        read_exact(r, &mut buf, what)?;
        Ok(buf[0])
    }

    pub(crate) fn read_u32_le<R: Read>(r: &mut R, what: &'static str) -> Result<u32, CacheError> {
        let mut buf = [0u8; 4];
        read_exact(r, &mut buf, what)?;
        Ok(u32::from_le_bytes(buf))
    }

    pub(crate) fn read_u64_le<R: Read>(r: &mut R, what: &'static str) -> Result<u64, CacheError> {
        let mut buf = [0u8; 8];
        read_exact(r, &mut buf, what)?;
        Ok(u64::from_le_bytes(buf))
    }

    pub(super) fn read_exact<R: Read>(
        r: &mut R,
        buf: &mut [u8],
        what: &'static str,
    ) -> Result<(), CacheError> {
        match r.read_exact(buf) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Err(CacheError::Truncated {
                what,
                needed: buf.len(),
                got: 0,
            }),
            Err(e) => Err(CacheError::Io(e)),
        }
    }

    /// Sanity ceiling on any length prefix. ~2 G elements; well above the
    /// largest legitimate `db_buf_aligned` (a few hundred million `u64`s) and
    /// any `Vec<PolyMatrixNTT>` count, but small enough that a corrupted
    /// length byte can't trigger an unbounded allocation that OOMs the
    /// process before we get a chance to return `CacheError`.
    pub(crate) const MAX_LEN_PREFIX: u64 = 1 << 31;

    /// Convert `u64 LE` length to `usize`, rejecting values that exceed
    /// [`MAX_LEN_PREFIX`] or `usize::MAX`. Catches corrupted length bytes
    /// that would otherwise OOM the process.
    pub(crate) fn u64_to_usize(v: u64, what: &'static str) -> Result<usize, CacheError> {
        if v > MAX_LEN_PREFIX {
            return Err(CacheError::Malformed {
                what,
                detail: format!(
                    "length {v} exceeds sanity ceiling {MAX_LEN_PREFIX} (likely corrupted)"
                ),
            });
        }
        usize::try_from(v).map_err(|_| CacheError::Malformed {
            what,
            detail: format!("length {v} exceeds usize::MAX"),
        })
    }

    /// Bulk write a `&[u64]` as raw LE bytes (one `write_all` call instead of
    /// `len` separate calls). On a little-endian target the in-memory layout
    /// is already the on-wire layout, so this is a single memcpy. Per-u64
    /// streaming would be ~5-10x slower for the multi-GB AlignedMemory64
    /// dumps. The consumer's `target_hash` rejects caches loaded on a
    /// big-endian host.
    fn write_u64_slice<W: Write>(w: &mut W, s: &[u64]) -> io::Result<()> {
        // SAFETY: u64 is a plain integer type with no padding; reinterpreting
        // its byte representation is well-defined. Length is exact.
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 8) };
        w.write_all(bytes)
    }

    /// Bulk read u64 LE bytes into the destination slice (one `read_exact`
    /// call instead of `len` separate calls). Same LE-target rationale as
    /// [`write_u64_slice`].
    fn read_u64_slice<R: Read>(
        r: &mut R,
        dst: &mut [u64],
        what: &'static str,
    ) -> Result<(), CacheError> {
        // SAFETY: as in write_u64_slice.
        let dst_bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, dst.len() * 8)
        };
        match r.read_exact(dst_bytes) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Err(CacheError::Truncated {
                what,
                needed: dst_bytes.len(),
                got: 0,
            }),
            Err(e) => Err(CacheError::Io(e)),
        }
    }

    pub(super) fn dump_vec_u64<W: Write>(w: &mut W, v: &[u64]) -> io::Result<()> {
        write_u64_le(w, v.len() as u64)?;
        write_u64_slice(w, v)
    }

    /// Read a `Vec<u64>` whose length must match `expected_len`. Validates
    /// the on-disk length prefix BEFORE allocating, so a corrupted length
    /// like `1 << 30` can't trigger a multi-GB allocation before being
    /// rejected. Use this whenever the expected length is derivable from
    /// `params` (e.g. `hint_0` size); use [`load_vec_u64_capped`] only for
    /// fields whose length truly varies.
    pub(super) fn load_vec_u64_exact<R: Read>(
        r: &mut R,
        expected_len: usize,
        what: &'static str,
    ) -> Result<Vec<u64>, CacheError> {
        let on_disk = u64_to_usize(read_u64_le(r, what)?, what)?;
        if on_disk != expected_len {
            return Err(CacheError::Malformed {
                what,
                detail: format!("length {on_disk} != expected {expected_len}"),
            });
        }
        let mut out = vec![0u64; expected_len];
        read_u64_slice(r, &mut out, what)?;
        Ok(out)
    }

    pub(crate) fn dump_aligned_memory64<W: Write>(
        w: &mut W,
        m: &AlignedMemory64,
    ) -> io::Result<()> {
        let slice: &[u64] = m.as_slice();
        write_u64_le(w, slice.len() as u64)?;
        write_u64_slice(w, slice)
    }

    /// Read an `AlignedMemory64` whose length must match `expected_len_u64`.
    /// Validates the on-disk length prefix BEFORE allocating, so a corrupted
    /// length can't trigger a multi-GB allocation before being rejected.
    pub(crate) fn load_aligned_memory64_exact<R: Read>(
        r: &mut R,
        expected_len_u64: usize,
        what: &'static str,
    ) -> Result<AlignedMemory64, CacheError> {
        let on_disk = u64_to_usize(read_u64_le(r, what)?, what)?;
        if on_disk != expected_len_u64 {
            return Err(CacheError::Malformed {
                what,
                detail: format!(
                    "length {on_disk} u64s != expected {expected_len_u64} u64s"
                ),
            });
        }
        let mut out = AlignedMemory64::new(expected_len_u64);
        read_u64_slice(r, out.as_mut_slice(), what)?;
        Ok(out)
    }

    pub(super) fn dump_poly_matrix_ntt<W: Write>(
        w: &mut W,
        m: &PolyMatrixNTT,
    ) -> io::Result<()> {
        write_u32_le(w, m.rows as u32)?;
        write_u32_le(w, m.cols as u32)?;
        write_u64_slice(w, m.as_slice())
    }

    /// Read a `PolyMatrixNTT` whose dimensions must match
    /// `(expected_rows, expected_cols)`. Validates the on-disk dim prefix
    /// BEFORE allocating, so a corrupted rows or cols can't multiply
    /// through to a multi-GB allocation before being rejected. All current
    /// SimplePIR consumers use this; the `MAX_LEN_PREFIX` cap is no longer
    /// the load-bearing protection for any matrix in the cache format.
    pub(super) fn load_poly_matrix_ntt_exact<'a, R: Read>(
        r: &mut R,
        params: &'a Params,
        expected_rows: usize,
        expected_cols: usize,
        what: &'static str,
    ) -> Result<PolyMatrixNTT<'a>, CacheError> {
        let rows = u64_to_usize(read_u32_le(r, what)? as u64, what)?;
        let cols = u64_to_usize(read_u32_le(r, what)? as u64, what)?;
        if rows != expected_rows || cols != expected_cols {
            return Err(CacheError::Malformed {
                what,
                detail: format!(
                    "matrix dims {rows}x{cols} != expected {expected_rows}x{expected_cols}"
                ),
            });
        }
        let mut out = PolyMatrixNTT::zero(params, expected_rows, expected_cols);
        read_u64_slice(r, out.as_mut_slice(), what)?;
        Ok(out)
    }

    pub(super) fn dump_vec_pmntt<W: Write>(
        w: &mut W,
        v: &[PolyMatrixNTT],
    ) -> io::Result<()> {
        write_u64_le(w, v.len() as u64)?;
        for m in v {
            dump_poly_matrix_ntt(w, m)?;
        }
        Ok(())
    }

    /// Read a `Vec<PolyMatrixNTT>` whose outer length and per-matrix dims
    /// must all match the expected values. Validates outer count BEFORE
    /// `Vec::with_capacity`, then validates each matrix's dims BEFORE
    /// `PolyMatrixNTT::zero`. Use this whenever both outer count and matrix
    /// shape are derivable from `params`.
    pub(super) fn load_vec_pmntt_exact<'a, R: Read>(
        r: &mut R,
        params: &'a Params,
        expected_outer: usize,
        expected_rows: usize,
        expected_cols: usize,
        what: &'static str,
    ) -> Result<Vec<PolyMatrixNTT<'a>>, CacheError> {
        let on_disk = u64_to_usize(read_u64_le(r, what)?, what)?;
        if on_disk != expected_outer {
            return Err(CacheError::Malformed {
                what,
                detail: format!("outer length {on_disk} != expected {expected_outer}"),
            });
        }
        let mut out = Vec::with_capacity(expected_outer);
        for _ in 0..expected_outer {
            out.push(load_poly_matrix_ntt_exact(
                r,
                params,
                expected_rows,
                expected_cols,
                what,
            )?);
        }
        Ok(out)
    }

    pub(super) fn dump_vec_vec_pmntt<W: Write>(
        w: &mut W,
        v: &[Vec<PolyMatrixNTT>],
    ) -> io::Result<()> {
        write_u64_le(w, v.len() as u64)?;
        for inner in v {
            dump_vec_pmntt(w, inner)?;
        }
        Ok(())
    }

    /// Read a `Vec<Vec<PolyMatrixNTT>>` where outer count, inner count, and
    /// per-matrix dims are all known up front and validated BEFORE any
    /// allocation. Used by `prepacked_lwe`, whose shape (per-tracing of
    /// `prep_pack_many_lwes` / `prep_pack_lwes` for SimplePIR) is fully
    /// deterministic from `params`.
    pub(super) fn load_vec_vec_pmntt_exact<'a, R: Read>(
        r: &mut R,
        params: &'a Params,
        expected_outer: usize,
        expected_inner: usize,
        expected_rows: usize,
        expected_cols: usize,
        what: &'static str,
    ) -> Result<Vec<Vec<PolyMatrixNTT<'a>>>, CacheError> {
        let on_disk_outer = u64_to_usize(read_u64_le(r, what)?, what)?;
        if on_disk_outer != expected_outer {
            return Err(CacheError::Malformed {
                what,
                detail: format!(
                    "outer length {on_disk_outer} != expected {expected_outer}"
                ),
            });
        }
        let mut out = Vec::with_capacity(expected_outer);
        for _ in 0..expected_outer {
            let on_disk_inner = u64_to_usize(read_u64_le(r, what)?, what)?;
            if on_disk_inner != expected_inner {
                return Err(CacheError::Malformed {
                    what,
                    detail: format!(
                        "inner length {on_disk_inner} != expected {expected_inner}"
                    ),
                });
            }
            let mut inner = Vec::with_capacity(expected_inner);
            for _ in 0..expected_inner {
                inner.push(load_poly_matrix_ntt_exact(
                    r,
                    params,
                    expected_rows,
                    expected_cols,
                    what,
                )?);
            }
            out.push(inner);
        }
        Ok(out)
    }

    pub(super) fn dump_vec_usize<W: Write>(w: &mut W, v: &[usize]) -> io::Result<()> {
        write_u64_le(w, v.len() as u64)?;
        for &x in v {
            write_u64_le(w, x as u64)?;
        }
        Ok(())
    }

    /// Read a `Vec<usize>` of exact length, validated BEFORE allocation.
    pub(super) fn load_vec_usize_exact<R: Read>(
        r: &mut R,
        expected_len: usize,
        what: &'static str,
    ) -> Result<Vec<usize>, CacheError> {
        let on_disk = u64_to_usize(read_u64_le(r, what)?, what)?;
        if on_disk != expected_len {
            return Err(CacheError::Malformed {
                what,
                detail: format!("length {on_disk} != expected {expected_len}"),
            });
        }
        let mut out = Vec::with_capacity(expected_len);
        for _ in 0..expected_len {
            out.push(u64_to_usize(read_u64_le(r, what)?, what)?);
        }
        Ok(out)
    }

    pub(super) fn dump_vec_vec_usize<W: Write>(
        w: &mut W,
        v: &[Vec<usize>],
    ) -> io::Result<()> {
        write_u64_le(w, v.len() as u64)?;
        for inner in v {
            dump_vec_usize(w, inner)?;
        }
        Ok(())
    }

    /// Read a `Vec<Vec<usize>>` with exact outer and inner lengths, both
    /// validated BEFORE allocation.
    pub(super) fn load_vec_vec_usize_exact<R: Read>(
        r: &mut R,
        expected_outer: usize,
        expected_inner: usize,
        what: &'static str,
    ) -> Result<Vec<Vec<usize>>, CacheError> {
        let on_disk_outer = u64_to_usize(read_u64_le(r, what)?, what)?;
        if on_disk_outer != expected_outer {
            return Err(CacheError::Malformed {
                what,
                detail: format!(
                    "outer length {on_disk_outer} != expected {expected_outer}"
                ),
            });
        }
        let mut out = Vec::with_capacity(expected_outer);
        for _ in 0..expected_outer {
            out.push(load_vec_usize_exact(r, expected_inner, what)?);
        }
        Ok(out)
    }

    pub(super) fn dump_precomp<W: Write>(w: &mut W, p: &Precomp) -> io::Result<()> {
        write_u64_le(w, p.len() as u64)?;
        for (m, vm, vvu) in p {
            dump_poly_matrix_ntt(w, m)?;
            dump_vec_pmntt(w, vm)?;
            dump_vec_vec_usize(w, vvu)?;
        }
        Ok(())
    }

    /// Expected per-tuple shape inside `Precomp` for SimplePIR. Per
    /// tracing of `precompute_pack`:
    /// - `m`: `working_set[0].clone()`, shape `2x1`
    /// - `vm`: built up by `res.push(condense_matrix(.., t_exp_left × 1))`,
    ///   total length = `poly_len - 1` (sum of `1 << (ell - cur_ell)` over
    ///   `cur_ell ∈ 1..=ell` where `ell = poly_len_log2`)
    /// - `vvu`: `generate_automorph_tables_brute_force(params)`, outer
    ///   length `poly_len_log2`, inner length `poly_len`
    pub(super) struct PrecompShape {
        pub outer: usize,
        pub m_rows: usize,
        pub m_cols: usize,
        pub vm_len: usize,
        pub vm_rows: usize,
        pub vm_cols: usize,
        pub vvu_outer: usize,
        pub vvu_inner: usize,
    }

    /// Read a `Precomp` whose every shape is known up front and validated
    /// BEFORE any allocation. Replaces the previous `_outer_exact` variant
    /// that accepted `MAX_LEN_PREFIX`-bounded inner allocations.
    pub(super) fn load_precomp_exact<'a, R: Read>(
        r: &mut R,
        params: &'a Params,
        shape: &PrecompShape,
        what: &'static str,
    ) -> Result<Precomp<'a>, CacheError> {
        let on_disk = u64_to_usize(read_u64_le(r, what)?, what)?;
        if on_disk != shape.outer {
            return Err(CacheError::Malformed {
                what,
                detail: format!("outer length {on_disk} != expected {}", shape.outer),
            });
        }
        let mut out = Vec::with_capacity(shape.outer);
        for _ in 0..shape.outer {
            let m =
                load_poly_matrix_ntt_exact(r, params, shape.m_rows, shape.m_cols, what)?;
            let on_disk_vm = u64_to_usize(read_u64_le(r, what)?, what)?;
            if on_disk_vm != shape.vm_len {
                return Err(CacheError::Malformed {
                    what,
                    detail: format!(
                        "vm length {on_disk_vm} != expected {}",
                        shape.vm_len
                    ),
                });
            }
            let mut vm = Vec::with_capacity(shape.vm_len);
            for _ in 0..shape.vm_len {
                vm.push(load_poly_matrix_ntt_exact(
                    r,
                    params,
                    shape.vm_rows,
                    shape.vm_cols,
                    what,
                )?);
            }
            let vvu = load_vec_vec_usize_exact(r, shape.vvu_outer, shape.vvu_inner, what)?;
            out.push((m, vm, vvu));
        }
        Ok(out)
    }
}

#[cfg(feature = "server")]
impl<'a> OfflinePrecomputedValues<'a> {
    /// Dump the precomputed values to a writer in a checked binary format.
    /// Format is binary-stable within a major version of `valar-ypir`.
    /// Intended for warm-restart caching only; see module docs for caveats.
    ///
    /// **Endianness:** the on-wire format is little-endian. The bulk
    /// `&[u64]` writes inside the cache I/O helpers reinterpret memory as
    /// raw bytes for performance, so this code only compiles on
    /// little-endian targets (enforced by a `compile_error!` at the top of
    /// the cache I/O module).
    ///
    /// # Errors
    ///
    /// Returns the underlying I/O error if the writer fails. Panics if
    /// `smaller_server` is `Some(_)` (this dump is for SimplePIR servers,
    /// where `smaller_server` is always `None`).
    pub fn dump_into<W: Write>(&self, w: &mut W) -> io::Result<()> {
        assert!(
            self.smaller_server.is_none(),
            "dump_into requires smaller_server to be None (SimplePIR)"
        );

        cache_io::write_u8(w, PAYLOAD_FORMAT_V1)?;
        // flags reserved for future use; bit 0 = smaller_server present
        cache_io::write_u8(w, 0)?;

        cache_io::dump_vec_u64(w, &self.hint_0)?;
        cache_io::dump_vec_u64(w, &self.hint_1)?;
        cache_io::dump_vec_pmntt(w, &self.pseudorandom_query_1)?;
        cache_io::dump_vec_pmntt(w, &self.y_constants.0)?;
        cache_io::dump_vec_pmntt(w, &self.y_constants.1)?;
        cache_io::dump_vec_vec_pmntt(w, &self.prepacked_lwe)?;
        cache_io::dump_vec_pmntt(w, &self.fake_pack_pub_params)?;
        cache_io::dump_precomp(w, &self.precomp)?;
        Ok(())
    }

    /// Load precomputed values from a reader. Bounds-checks every access;
    /// returns `CacheError` on truncation, malformed data, or version
    /// mismatch. Never panics on disk-derived input.
    ///
    /// `params` must match the params the cache was originally produced with;
    /// the consumer is responsible for verifying that via its own header
    /// (typically a hash of the relevant `Params` fields).
    ///
    /// **Trailing-byte contract:** this method consumes exactly the bytes
    /// produced by one call to [`Self::dump_into`] and stops; it does NOT
    /// check whether the reader has more bytes after that. Callers that want
    /// "this file contains exactly one dump" semantics must perform their
    /// own EOF check on the reader after this returns. (Within the consumer
    /// repo this lets us chain `YServer::load_from` then
    /// `OfflinePrecomputedValues::load_from` against a single cache file.)
    ///
    /// **Validation order:** length prefixes for every well-known field are
    /// checked against the expected size derived from `params` BEFORE any
    /// allocation. A corrupted length prefix can't trigger a multi-GB
    /// allocation before being rejected.
    pub fn load_from<R: Read>(
        r: &mut R,
        params: &'a Params,
    ) -> Result<Self, CacheError> {
        let v = cache_io::read_u8(r, "payload version")?;
        if v != PAYLOAD_FORMAT_V1 {
            return Err(CacheError::Malformed {
                what: "payload version",
                detail: format!("unknown version {v}, expected {PAYLOAD_FORMAT_V1}"),
            });
        }

        // Known flag bits for PAYLOAD_FORMAT_V1:
        //   bit 0: smaller_server-present (must be 0 — SimplePIR-only loader)
        // Bits 1..=7 are reserved; reject if set, so a future format that
        // assigns them isn't silently mis-loaded by older code.
        const KNOWN_FLAGS: u8 = 0b0000_0001;
        let flags = cache_io::read_u8(r, "flags")?;
        if flags & !KNOWN_FLAGS != 0 {
            return Err(CacheError::Malformed {
                what: "flags",
                detail: format!(
                    "unknown flag bits set: 0x{flags:02x} (known mask 0x{KNOWN_FLAGS:02x})"
                ),
            });
        }
        if flags & 0x01 != 0 {
            return Err(CacheError::Malformed {
                what: "flags",
                detail: "smaller_server-present flag set; this loader expects None".to_string(),
            });
        }

        // Expected SimplePIR shapes from
        // `YServer::perform_offline_precomputation_simplepir`. If the upstream
        // algorithm changes any of these (e.g. different num_rlwe_outputs
        // calculation), bump PAYLOAD_FORMAT_V1 and update this map together.
        let db_cols = params
            .instances
            .checked_mul(params.poly_len)
            .ok_or(CacheError::Malformed {
                what: "params",
                detail: "instances * poly_len overflowed".to_string(),
            })?;
        let expected_hint_0_len = params
            .poly_len
            .checked_mul(db_cols)
            .ok_or(CacheError::Malformed {
                what: "params",
                detail: "poly_len * db_cols overflowed".to_string(),
            })?;
        let num_rlwe_outputs = params.instances;

        // Each `_exact` loader checks the on-disk length prefix against the
        // expected value BEFORE allocating, so a corrupted length cannot
        // trigger a multi-GB allocation.
        let hint_0 = cache_io::load_vec_u64_exact(r, expected_hint_0_len, "hint_0")?;
        let hint_1 = cache_io::load_vec_u64_exact(r, 0, "hint_1 (SimplePIR expects empty)")?;
        let pseudorandom_query_1 = cache_io::load_vec_pmntt_exact(
            r,
            params,
            0,
            0,
            0,
            "pseudorandom_query_1 (SimplePIR expects empty)",
        )?;
        let y_constants_0 = cache_io::load_vec_pmntt_exact(
            r,
            params,
            params.poly_len_log2,
            1,
            1,
            "y_constants.0",
        )?;
        let y_constants_1 = cache_io::load_vec_pmntt_exact(
            r,
            params,
            params.poly_len_log2,
            1,
            1,
            "y_constants.1",
        )?;
        // prepacked_lwe shape (per-tracing of `prep_pack_many_lwes` /
        // `prep_pack_lwes` in packing.rs for SimplePIR):
        //   - outer length: num_rlwe_outputs (= params.instances)
        //   - inner length: params.poly_len
        //   - per matrix: 2 x 1 (PolyMatrixRaw::zero(params, 2, 1).ntt())
        let prepacked_lwe = cache_io::load_vec_vec_pmntt_exact(
            r,
            params,
            num_rlwe_outputs,
            params.poly_len,
            2,
            1,
            "prepacked_lwe",
        )?;
        let fake_pack_pub_params = cache_io::load_vec_pmntt_exact(
            r,
            params,
            params.poly_len_log2,
            2,
            params.t_exp_left,
            "fake_pack_pub_params",
        )?;
        // precomp shape (per-tracing of `precompute_pack` in packing.rs):
        //   - outer length: num_rlwe_outputs
        //   - per tuple:
        //     - first PolyMatrixNTT: 2 x 1 (working_set[0].clone())
        //     - second Vec<PolyMatrixNTT>:
        //         length poly_len - 1 (sum of `1 << (ell - cur_ell)` over
        //         cur_ell in 1..=poly_len_log2)
        //         per matrix: t_exp_left x 1 (condense_matrix preserves
        //         shape from PolyMatrixNTT::zero(params, t_exp_left, 1))
        //     - third Vec<Vec<usize>> (automorph tables):
        //         outer poly_len_log2, inner poly_len
        let precomp_shape = cache_io::PrecompShape {
            outer: num_rlwe_outputs,
            m_rows: 2,
            m_cols: 1,
            vm_len: params.poly_len.checked_sub(1).ok_or(CacheError::Malformed {
                what: "params",
                detail: "poly_len < 1".to_string(),
            })?,
            vm_rows: params.t_exp_left,
            vm_cols: 1,
            vvu_outer: params.poly_len_log2,
            vvu_inner: params.poly_len,
        };
        let precomp = cache_io::load_precomp_exact(r, params, &precomp_shape, "precomp")?;

        Ok(OfflinePrecomputedValues {
            hint_0,
            hint_1,
            pseudorandom_query_1,
            y_constants: (y_constants_0, y_constants_1),
            smaller_server: None,
            prepacked_lwe,
            fake_pack_pub_params,
            precomp,
        })
    }
}

#[cfg(feature = "server")]
use crate::server::YServer;

#[cfg(not(feature = "server"))]
type YServer<'a, T> = T;

#[derive(Clone)]
pub struct OfflinePrecomputedValues<'a> {
    pub hint_0: Vec<u64>,
    pub hint_1: Vec<u64>,
    pub pseudorandom_query_1: Vec<PolyMatrixNTT<'a>>,
    pub y_constants: (Vec<PolyMatrixNTT<'a>>, Vec<PolyMatrixNTT<'a>>),
    pub smaller_server: Option<YServer<'a, u16>>,
    pub prepacked_lwe: Vec<Vec<PolyMatrixNTT<'a>>>,
    pub fake_pack_pub_params: Vec<PolyMatrixNTT<'a>>,
    pub precomp: Precomp<'a>,
}

pub trait ToBytes {
    fn to_bytes(&self) -> Vec<u8>;
}

pub trait AsBytes {
    fn as_bytes(&self) -> &[u8];
}

pub trait AsBytesMut {
    fn as_bytes_mut(&mut self) -> &mut [u8];
}

pub trait FromBytes {
    fn from_bytes(data: &[u8]) -> Self;
}

pub trait FromBytesParams<'a> {
    fn from_bytes(data: &[u8], params: &'a Params) -> Self;
}

impl ToBytes for &[u32] {
    fn to_bytes(&self) -> Vec<u8> {
        // fast
        unsafe {
            let ptr = self.as_ptr() as *const u8;
            std::slice::from_raw_parts(ptr, self.len() * 4).to_vec()
        }
    }
}

impl AsBytes for &[u32] {
    fn as_bytes(&self) -> &[u8] {
        // fast
        unsafe {
            let ptr = self.as_ptr() as *const u8;
            std::slice::from_raw_parts(ptr, self.len() * 4)
        }
    }
}

impl FromBytes for Vec<u32> {
    fn from_bytes(data: &[u8]) -> Self {
        // fast
        unsafe {
            let mut out = Vec::with_capacity(data.len() / 4);
            let u8_mut = std::slice::from_raw_parts(data.as_ptr(), data.len());
            out.set_len(data.len() / 4);
            let ptr = out.as_mut_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(u8_mut.as_ptr(), ptr, data.len());
            out
        }
    }
}

impl ToBytes for &[u64] {
    fn to_bytes(&self) -> Vec<u8> {
        // fast
        unsafe {
            let ptr = self.as_ptr() as *const u8;
            std::slice::from_raw_parts(ptr, self.len() * 8).to_vec()
        }
    }
}

impl AsBytes for &[u64] {
    fn as_bytes(&self) -> &[u8] {
        // fast
        unsafe {
            let ptr = self.as_ptr() as *const u8;
            std::slice::from_raw_parts(ptr, self.len() * 8)
        }
    }
}

impl AsBytesMut for &mut [u64] {
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        // fast
        unsafe {
            let ptr = self.as_mut_ptr() as *mut u8;
            std::slice::from_raw_parts_mut(ptr, self.len() * 8)
        }
    }
}

impl FromBytes for AlignedMemory64 {
    fn from_bytes(data: &[u8]) -> Self {
        // fast
        unsafe {
            let mut out = AlignedMemory64::new(data.len() / 8);
            let u8_mut = std::slice::from_raw_parts(data.as_ptr(), data.len());
            let ptr = out.as_mut_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(u8_mut.as_ptr(), ptr, data.len());
            out
        }
    }
}

impl FromBytes for Vec<u64> {
    fn from_bytes(data: &[u8]) -> Self {
        // fast
        unsafe {
            let mut out = Vec::with_capacity(data.len() / 8);
            let u8_mut = std::slice::from_raw_parts(data.as_ptr(), data.len());
            out.set_len(data.len() / 8);
            let ptr = out.as_mut_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(u8_mut.as_ptr(), ptr, data.len());
            out
        }
    }
}

impl ToBytes for YPIRQuery {
    fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(self.0.as_slice().as_bytes());
        out.extend_from_slice(self.1.as_slice().as_bytes());
        out.extend_from_slice(self.2.as_slice().as_bytes());
        out
    }
}

impl ToBytes for YPIRSimpleQuery {
    fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(self.0.as_slice().as_bytes());
        out.extend_from_slice(self.1.as_slice().as_bytes());
        out
    }
}

impl ToBytes for Vec<usize> {
    fn to_bytes(&self) -> Vec<u8> {
        // fast
        unsafe {
            let ptr = self.as_ptr() as *const u8;
            std::slice::from_raw_parts(ptr, self.len() * 8).to_vec()
        }
    }
}

impl FromBytesParams<'_> for Vec<usize> {
    fn from_bytes(data: &[u8], _params: &Params) -> Vec<usize> {
        // fast
        unsafe {
            let mut out = Vec::with_capacity(data.len() / std::mem::size_of::<usize>());
            let u8_mut = std::slice::from_raw_parts(data.as_ptr(), data.len());
            out.set_len(data.len() / std::mem::size_of::<usize>());
            let ptr = out.as_mut_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(u8_mut.as_ptr(), ptr, data.len());
            out
        }
    }
}

impl<'a> ToBytes for PolyMatrixNTT<'a> {
    fn to_bytes(&self) -> Vec<u8> {
        // write rows, cols, and data
        let mut out = Vec::new();
        out.extend_from_slice(&self.rows.to_be_bytes());
        out.extend_from_slice(&self.cols.to_be_bytes());
        out.extend_from_slice(self.as_slice().as_bytes());
        out
    }
}

impl<'a> FromBytesParams<'a> for PolyMatrixNTT<'a> {
    fn from_bytes(data: &[u8], params: &'a Params) -> PolyMatrixNTT<'a> {
        let rows = u64::from_be_bytes(data[0..8].try_into().unwrap()) as usize;
        let cols = u64::from_be_bytes(data[8..16].try_into().unwrap()) as usize;
        let data = &data[16..];
        let mut out = PolyMatrixNTT::zero(params, rows, cols);
        out.as_mut_slice().as_bytes_mut().copy_from_slice(data);
        out
    }
}

impl<T: ToBytes> ToBytes for Vec<T> {
    fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.len() as u64).to_be_bytes());
        for item in self {
            let item_bytes = item.to_bytes();
            out.extend_from_slice(&(item_bytes.len() as u64).to_be_bytes());
            out.extend_from_slice(&item_bytes);
        }
        out
    }
}

impl<'a, T: FromBytesParams<'a>> FromBytesParams<'a> for Vec<T> {
    fn from_bytes(data: &[u8], params: &'a Params) -> Vec<T> {
        let mut out = Vec::new();
        let mut data = data;
        let len = u64::from_be_bytes(data[0..8].try_into().unwrap()) as usize;
        data = &data[8..];
        for _ in 0..len {
            let item_len = u64::from_be_bytes(data[0..8].try_into().unwrap()) as usize;
            let item = T::from_bytes(&data[8..8 + item_len], params);
            out.push(item);
            data = &data[8 + item_len..];
        }
        out
    }
}

// length 2 tuple
impl<T1: ToBytes, T2: ToBytes> ToBytes for (T1, T2) {
    fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let item1 = self.0.to_bytes();
        let item2 = self.1.to_bytes();
        out.extend_from_slice(&(item1.len() as u64).to_be_bytes());
        out.extend_from_slice(&item1);
        out.extend_from_slice(&item2);
        out
    }
}

impl<'a, T1: FromBytesParams<'a>, T2: FromBytesParams<'a>> FromBytesParams<'a> for (T1, T2) {
    fn from_bytes(data: &[u8], params: &'a Params) -> (T1, T2) {
        let len1 = u64::from_be_bytes(data[0..8].try_into().unwrap()) as usize;
        let item1 = T1::from_bytes(&data[8..8 + len1], params);
        let item2 = T2::from_bytes(&data[8 + len1..], params);
        (item1, item2)
    }
}

// length 3 tuple
impl<T1: ToBytes, T2: ToBytes, T3: ToBytes> ToBytes for (T1, T2, T3) {
    fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let item1 = self.0.to_bytes();
        let item2 = self.1.to_bytes();
        let item3 = self.2.to_bytes();
        out.extend_from_slice(&(item1.len() as u64).to_be_bytes());
        out.extend_from_slice(&item1);
        out.extend_from_slice(&(item2.len() as u64).to_be_bytes());
        out.extend_from_slice(&item2);
        out.extend_from_slice(&item3);
        out
    }
}

impl<'a, T1: FromBytesParams<'a>, T2: FromBytesParams<'a>, T3: FromBytesParams<'a>>
    FromBytesParams<'a> for (T1, T2, T3)
{
    fn from_bytes(data: &[u8], params: &'a Params) -> (T1, T2, T3) {
        let len1 = u64::from_be_bytes(data[0..8].try_into().unwrap()) as usize;
        let item1 = T1::from_bytes(&data[8..8 + len1], params);
        let len2 = u64::from_be_bytes(data[8 + len1..16 + len1].try_into().unwrap()) as usize;
        let item2 = T2::from_bytes(&data[16 + len1..16 + len1 + len2], params);
        let item3 = T3::from_bytes(&data[16 + len1 + len2..], params);
        (item1, item2, item3)
    }
}

// pub hint_0: Vec<u64>,
// pub hint_1: Vec<u64>,
// pub pseudorandom_query_1: Vec<PolyMatrixNTT<'a>>,
// pub y_constants: (Vec<PolyMatrixNTT<'a>>, Vec<PolyMatrixNTT<'a>>),
// pub smaller_server: Option<YServer<'a, u16>>,
// pub prepacked_lwe: Vec<Vec<PolyMatrixNTT<'a>>>,
// pub fake_pack_pub_params: Vec<PolyMatrixNTT<'a>>,
// pub precomp: Precomp<'a>,

impl<'a> ToBytes for OfflinePrecomputedValues<'a> {
    fn to_bytes(&self) -> Vec<u8> {
        assert!(self.smaller_server.is_none());

        let mut out = Vec::new();
        out.extend_from_slice(&self.hint_0.as_slice().to_bytes());
        out.extend_from_slice(&self.hint_1.as_slice().to_bytes());
        out.extend_from_slice(&self.pseudorandom_query_1.to_bytes());
        out.extend_from_slice(&self.y_constants.0.to_bytes());
        out.extend_from_slice(&self.y_constants.1.to_bytes());
        out.extend_from_slice(&self.prepacked_lwe.to_bytes());
        out.extend_from_slice(&self.fake_pack_pub_params.to_bytes());
        out.extend_from_slice(&self.precomp.to_bytes());
        out
    }
}

impl<'a> FromBytesParams<'a> for OfflinePrecomputedValues<'a> {
    fn from_bytes(data: &[u8], params: &'a Params) -> OfflinePrecomputedValues<'a> {
        let mut data = data;
        let hint_0 = Vec::<u64>::from_bytes(data);
        data = &data[hint_0.len() * 8..];
        let hint_1 = Vec::<u64>::from_bytes(data);
        data = &data[hint_1.len() * 8..];
        let pseudorandom_query_1 = Vec::<PolyMatrixNTT>::from_bytes(data, params);
        data = &data[pseudorandom_query_1.to_bytes().len()..];
        let y_constants = (
            Vec::<PolyMatrixNTT>::from_bytes(data, params),
            Vec::<PolyMatrixNTT>::from_bytes(data, params),
        );
        data = &data[y_constants.0.to_bytes().len()..];
        let prepacked_lwe = Vec::<Vec<PolyMatrixNTT>>::from_bytes(data, params);
        data = &data[prepacked_lwe.to_bytes().len()..];
        let fake_pack_pub_params = Vec::<PolyMatrixNTT>::from_bytes(data, params);
        data = &data[fake_pack_pub_params.to_bytes().len()..];
        let precomp = Precomp::from_bytes(data, params);
        OfflinePrecomputedValues {
            hint_0,
            hint_1,
            pseudorandom_query_1,
            y_constants,
            smaller_server: None,
            prepacked_lwe,
            fake_pack_pub_params,
            precomp,
        }
    }
}

pub fn read_file_to_vec_u64(filename: &str) -> Vec<u64> {
    let mut file = File::open(filename).unwrap();
    let mut data = Vec::new();
    file.read_to_end(&mut data).unwrap();
    let mut out = Vec::with_capacity(data.len() / 8);
    let mut iter = data.chunks_exact(8);
    for _ in 0..iter.len() {
        out.push(u64::from_le_bytes(iter.next().unwrap().try_into().unwrap()));
    }
    out
}

pub fn write_vec_u64_to_file(filename: &str, data: &[u64]) {
    let mut file = File::create(filename).unwrap();
    for &x in data {
        file.write_all(&x.to_le_bytes()).unwrap();
    }
}

pub fn pack_vec_pm(
    params: &Params,
    rows: usize,
    cols: usize,
    v_cts: &[PolyMatrixNTT],
) -> AlignedMemory64 {
    assert_eq!(v_cts[0].rows, rows);
    assert_eq!(v_cts[0].cols, cols);
    assert_eq!(params.crt_count, 2);
    let mut aligned_out = AlignedMemory64::new(v_cts.len() * rows * cols * params.poly_len);
    let mut iter = aligned_out
        .as_mut_slice()
        .chunks_exact_mut(rows * cols * params.poly_len);
    for ct in v_cts {
        let out = iter.next().unwrap();
        for row in 0..rows {
            for col in 0..cols {
                let out_offs = (row * cols + col) * params.poly_len;
                let inp_offs = (row * cols + col) * 2 * params.poly_len;
                for z in 0..params.poly_len {
                    out[out_offs + z] =
                        ct.data[inp_offs + z] | (ct.data[inp_offs + z + params.poly_len] << 32);
                }
            }
        }
    }
    aligned_out
}

pub fn unpack_vec_pm<'a>(
    params: &'a Params,
    rows: usize,
    cols: usize,
    data: &[u64],
) -> Vec<PolyMatrixNTT<'a>> {
    assert_eq!(params.crt_count, 2);
    let mut v_cts = Vec::with_capacity(data.len() / (rows * cols * params.poly_len));
    let mut iter = data.chunks_exact(rows * cols * params.poly_len);
    for _ in 0..v_cts.capacity() {
        let in_data = iter.next().unwrap();
        let mut ct = PolyMatrixNTT::zero(params, rows, cols);
        for row in 0..rows {
            for col in 0..cols {
                // this is (on purpose) not the inverse of pack_vec_pm
                let in_offs = (row * cols + col) * params.poly_len;
                let out_offs = (row * cols + col) * 2 * params.poly_len;
                for z in 0..params.poly_len {
                    ct.data[out_offs + z] = in_data[in_offs + z];
                }
            }
        }
        v_cts.push(ct);
    }
    v_cts
}

pub fn condense_matrix<'a>(params: &'a Params, a: &PolyMatrixNTT<'a>) -> PolyMatrixNTT<'a> {
    let mut res = PolyMatrixNTT::zero(params, a.rows, a.cols);
    for i in 0..a.rows {
        for j in 0..a.cols {
            let res_poly = &mut res.get_poly_mut(i, j);
            let a_poly = a.get_poly(i, j);
            for z in 0..params.poly_len {
                res_poly[z] = a_poly[z] | (a_poly[z + params.poly_len] << 32);
            }
        }
    }
    res
}

pub fn uncondense_matrix<'a>(params: &'a Params, a: &PolyMatrixNTT<'a>) -> PolyMatrixNTT<'a> {
    let mut res = PolyMatrixNTT::zero(params, a.rows, a.cols);
    for i in 0..a.rows {
        for j in 0..a.cols {
            let res_poly = &mut res.get_poly_mut(i, j);
            let a_poly = a.get_poly(i, j);
            for z in 0..params.poly_len {
                res_poly[z] = a_poly[z] & ((1u64 << 32) - 1);
                res_poly[z + params.poly_len] = a_poly[z] >> 32;
            }
        }
    }
    res
}

pub struct FilePtIter<R: Read + Seek> {
    file: R,
    pt_bits: usize,
    bytes_per_row: usize,
    db_cols: usize,
    row_idx: usize,
    col_idx: usize,
    buf_pos: usize,
    buf: Vec<u8>,
    buf_vals: Vec<u16>,
}

impl<R: Read + Seek> FilePtIter<R> {
    pub fn new(file: R, bytes_per_row: usize, db_cols: usize, pt_bits: usize) -> Self {
        let max_filled_col = (bytes_per_row * 8 + pt_bits - 1) / pt_bits;
        assert!(max_filled_col <= db_cols);

        Self {
            file,
            pt_bits,
            bytes_per_row,
            db_cols,
            col_idx: 0,
            row_idx: 0,
            buf_pos: 8,
            buf: vec![0; pt_bits * 16],
            buf_vals: vec![0; pt_bits * 8],
        }
    }
}

impl FilePtIter<BufReader<File>> {
    pub fn from_file(filename: &str, bytes_per_row: usize, db_cols: usize, pt_bits: usize) -> Self {
        println!("bytes_per_row: {}, pt_bits: {}", bytes_per_row, pt_bits);
        Self::new(
            BufReader::new(File::open(filename).unwrap()),
            bytes_per_row,
            db_cols,
            pt_bits,
        )
    }
}

impl<R: Read + Seek> Iterator for FilePtIter<R> {
    type Item = u16;

    fn next(&mut self) -> Option<Self::Item> {
        // reads file, pt_bits at a time

        // max_filled_col pt-bits sized words contain data in each row (rest are zeros)
        let max_filled_col = (self.bytes_per_row * 8 + self.pt_bits - 1) / self.pt_bits;
        assert!(max_filled_col <= self.db_cols);
        if self.col_idx >= self.db_cols {
            self.col_idx = 0;
            self.row_idx += 1;
            let seeked_to = self
                .file
                .seek(std::io::SeekFrom::Start(
                    self.row_idx as u64 * self.bytes_per_row as u64,
                ))
                .unwrap();
            assert_eq!(seeked_to, self.row_idx as u64 * self.bytes_per_row as u64);
            self.buf_vals.fill(0);
            self.buf_pos = 8;
        } else if self.col_idx >= max_filled_col {
            self.col_idx += 1;
            return Some(0);
        }

        if self.buf_pos == 8 {
            self.buf_pos = 0;
            self.buf.fill(0);

            let bytes_consumed = (self.col_idx / 8) * self.pt_bits;
            let remaining_in_row = self.bytes_per_row - bytes_consumed;
            let to_read = remaining_in_row.min(self.pt_bits);

            let read = self.file.read_exact(&mut self.buf[..to_read]);
            if read.is_err() {
                self.buf_pos = 8;
                return Some(0);
            }

            // now, populate buf_vals with the (up to) 8 pt_bits-sized words
            self.buf_vals.fill(0);
            let mut bit_offs = 0;
            for i in 0..8 {
                let val = read_arbitrary_bits(&self.buf, bit_offs, self.pt_bits);
                self.buf_vals[i] = val as u16;
                bit_offs += self.pt_bits;
            }
        }

        // return the next value in buf_vals
        let val = self.buf_vals[self.buf_pos];
        self.buf_pos += 1;
        self.col_idx += 1;
        Some(val)
    }
}

#[cfg(all(test, feature = "server"))]
mod test {
    use std::io::Cursor;

    use crate::{
        bits::u64s_to_contiguous_bytes,
        params::{params_for_scenario, params_for_scenario_simplepir, DbRowsCols, PtModulusBits},
        server::{ToU64, YServer},
    };

    use super::*;

    #[test]
    fn test_pack_unpack_vec_pm() {
        let params = params_for_scenario(1 << 10, 1);
        let rows = 5;
        let cols = 3;
        let len = 7;
        let mut v_cts = Vec::new();
        for _ in 0..len {
            v_cts.push(PolyMatrixRaw::random(&params, rows, cols).ntt());
        }
        let data = pack_vec_pm(&params, rows, cols, &v_cts);
        let v_cts2_weird = unpack_vec_pm(&params, rows, cols, data.as_slice());
        let v_cts2 = v_cts2_weird
            .into_iter()
            .map(|ct| uncondense_matrix(&params, &ct))
            .collect::<Vec<_>>();
        for (ct1, ct2) in v_cts.iter().zip(v_cts2.iter()) {
            assert_eq!(ct1.raw().as_slice(), ct2.raw().as_slice());
        }
    }

    #[test]
    fn test_pt_iter() {
        let num_items = 1 << 14;
        let item_size_bytes = 16384;

        let params = params_for_scenario_simplepir(num_items, item_size_bytes as u64 * 8);
        let inp_data = (0..params.db_rows())
            .flat_map(|i| {
                let mut out = vec![0u8; item_size_bytes];
                out[0] = i as u8;
                (&mut out[1..5]).copy_from_slice(&(i as u32).to_be_bytes());
                out
            })
            .collect::<Vec<_>>();
        let cursor = Cursor::new(inp_data);
        let pt_iter = FilePtIter::new(
            cursor,
            item_size_bytes,
            params.db_cols_simplepir(),
            params.pt_modulus_bits(),
        );
        let y_server = YServer::<u16>::new(&params, pt_iter, true, false, true);

        for i in 0..params.db_rows() {
            let row = y_server
                .get_row(i)
                .iter()
                .map(|x| x.to_u64())
                .collect::<Vec<_>>();
            let ci_bytes = u64s_to_contiguous_bytes(&row, params.pt_modulus_bits());
            assert_eq!(ci_bytes[0], i as u8);
            assert_eq!(
                i as u32,
                u32::from_be_bytes(ci_bytes[1..5].try_into().unwrap())
            );
        }
    }

    #[test]
    fn test_pt_iter_unaligned_rows() {
        use crate::bits::u64s_to_contiguous_bytes;

        let pt_bits = 14;
        let bytes_per_row = 20;
        let max_filled_col = (bytes_per_row * 8 + pt_bits - 1) / pt_bits;
        let db_cols = max_filled_col + 5;
        let num_rows = 4;

        let mut data = vec![0u8; num_rows * bytes_per_row];
        for row in 0..num_rows {
            for b in 0..bytes_per_row {
                data[row * bytes_per_row + b] = ((row + 1) * 50 + b) as u8;
            }
        }

        let cursor = Cursor::new(data.clone());
        let mut iter = FilePtIter::new(cursor, bytes_per_row, db_cols, pt_bits);

        for row in 0..num_rows {
            let expected_bytes = &data[row * bytes_per_row..(row + 1) * bytes_per_row];
            let mut pt_vals = Vec::new();
            for _ in 0..db_cols {
                pt_vals.push(iter.next().unwrap());
            }

            for col in max_filled_col..db_cols {
                assert_eq!(pt_vals[col], 0, "row {row} col {col}: expected zero padding");
            }

            let as_u64s: Vec<u64> = pt_vals[..max_filled_col].iter().map(|&v| v as u64).collect();
            let reconstructed = u64s_to_contiguous_bytes(&as_u64s, pt_bits);
            assert_eq!(
                &reconstructed[..bytes_per_row],
                expected_bytes,
                "row {row}: round-trip mismatch — cross-row contamination?"
            );
        }
    }
}
