// Most AVX-512 intrinsics are stable on stable Rust, but the conversion
// family (`_mm512_cvtepu{8,16,32}_epi64` used in `server.rs`) is still
// gated behind the `stdarch_x86_avx512` library feature.  Enabling the
// `explicit_avx512` Cargo feature therefore requires a nightly toolchain
// for now; this `cfg_attr` opts in to the unstable library feature only
// when needed so stable builds without `explicit_avx512` continue to
// work.  See rust-lang/rust#111137 for the tracking issue.
#![cfg_attr(feature = "explicit_avx512", feature(stdarch_x86_avx512))]

pub mod bits;
pub mod client;
pub mod constants;
pub mod convolution;
pub mod lwe;
pub mod measurement;
pub mod modulus_switch;
pub mod noise_analysis;
pub mod params;
pub mod seed;
pub mod serialize;
pub mod transpose;
pub mod util;

#[cfg(feature = "server")]
pub mod kernel;
#[cfg(feature = "server")]
pub mod matmul;
#[cfg(feature = "server")]
pub mod packing;
#[cfg(feature = "server")]
pub mod scheme;
#[cfg(feature = "server")]
pub mod server;

#[cfg(feature = "test_data")]
pub mod data;
