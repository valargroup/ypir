use log::debug;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::{error::Error, fmt, panic};

use sha1::{Digest, Sha1};
use spiral_rs::aligned_memory::AlignedMemory64;
use spiral_rs::{
    arith::*, client::*, discrete_gaussian::*, gadget::*, number_theory::*, params::*, poly::*,
};

use crate::bits::{read_bits, u64s_to_contiguous_bytes};
use crate::measurement::get_vec_pm_size_bytes;
use crate::modulus_switch::ModulusSwitch;
use crate::params::*;
use crate::seed::{generate_secure_random_seed, Seed};
use crate::serialize::*;

use super::convolution::negacyclic_matrix_u32;
use super::{constants::*, lwe::*, noise_analysis::measure_noise_width_squared, util::*};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum YPIRDecodeError {
    Panic(String),
}

impl fmt::Display for YPIRDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            YPIRDecodeError::Panic(msg) => write!(f, "response decoding panicked: {msg}"),
        }
    }
}

impl Error for YPIRDecodeError {}

fn panic_payload_message(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(msg) => *msg,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(msg) => (*msg).to_string(),
            Err(_) => "unknown panic".to_string(),
        },
    }
}

pub fn rlwe_to_lwe<'a>(params: &'a Params, ct: &PolyMatrixRaw<'a>) -> Vec<u64> {
    let a = ct.get_poly(0, 0);
    let mut negacylic_a = negacyclic_matrix(&a, params.modulus);
    negacylic_a.extend(ct.get_poly(1, 0));

    negacylic_a
}

pub fn pack_query(params: &Params, query: &[u64]) -> AlignedMemory64 {
    let query_packed = query
        .iter()
        .enumerate()
        .map(|(_i, x)| {
            let crt0 = (*x) % params.moduli[0];
            let crt1 = (*x) % params.moduli[1];
            crt0 | (crt1 << 32)
        })
        .collect::<Vec<_>>();
    let mut aligned_query_packed = AlignedMemory64::new(query_packed.len());
    aligned_query_packed
        .as_mut_slice()
        .copy_from_slice(&query_packed);
    aligned_query_packed
}

pub fn get_reg_sample<'a>(
    params: &'a Params,
    sk_reg: &PolyMatrixRaw<'a>,
    rng: &mut ChaCha20Rng,
    rng_pub: &mut ChaCha20Rng,
) -> PolyMatrixNTT<'a> {
    let a = PolyMatrixRaw::random_rng(params, 1, 1, rng_pub);
    let e = PolyMatrixRaw::noise(
        params,
        1,
        1,
        &DiscreteGaussian::init(params.noise_width),
        rng,
    );
    let b_p = &sk_reg.ntt() * &a.ntt();
    let b = &e.ntt() + &b_p;
    let mut p = PolyMatrixNTT::zero(params, 2, 1);
    p.copy_into(&(-&a).ntt(), 0, 0);
    p.copy_into(&b, 1, 0);
    p
}

pub fn get_fresh_reg_public_key<'a>(
    params: &'a Params,
    sk_reg: &PolyMatrixRaw<'a>,
    m: usize,
    rng: &mut ChaCha20Rng,
    rng_pub: &mut ChaCha20Rng,
) -> PolyMatrixNTT<'a> {
    let mut p = PolyMatrixNTT::zero(params, 2, m);

    for i in 0..m {
        p.copy_into(&get_reg_sample(params, sk_reg, rng, rng_pub), 0, i);
    }
    p
}

pub fn raw_generate_expansion_params<'a>(
    params: &'a Params,
    sk_reg: &PolyMatrixRaw<'a>,
    num_exp: usize,
    m_exp: usize,
    rng: &mut ChaCha20Rng,
    rng_pub: &mut ChaCha20Rng,
) -> Vec<PolyMatrixNTT<'a>> {
    let g_exp = build_gadget(params, 1, m_exp);
    debug!("using gadget base {}", g_exp.get_poly(0, 1)[0]);
    let g_exp_ntt = g_exp.ntt();
    let mut res = Vec::new();

    for i in 0..num_exp {
        let t = (params.poly_len / (1 << i)) + 1;
        let tau_sk_reg = automorph_alloc(&sk_reg, t);
        let prod = &tau_sk_reg.ntt() * &g_exp_ntt;

        // let w_exp_i = client.encrypt_matrix_reg(&prod, rng, rng_pub);
        let sample = get_fresh_reg_public_key(params, &sk_reg, m_exp, rng, rng_pub);
        let w_exp_i = &sample + &prod.pad_top(1);
        res.push(w_exp_i);
    }

    res
}

pub fn decrypt_ct_reg_measured<'a>(
    client: &Client<'a>,
    params: &'a Params,
    ct: &PolyMatrixNTT<'a>,
    coeffs_to_measure: usize,
) -> PolyMatrixRaw<'a> {
    let dec_result = client.decrypt_matrix_reg(ct).raw();

    let mut dec_rescaled = PolyMatrixRaw::zero(&params, dec_result.rows, dec_result.cols);
    for z in 0..dec_rescaled.data.len() {
        dec_rescaled.data[z] = rescale(dec_result.data[z], params.modulus, params.pt_modulus);
    }

    // measure noise width
    let s_2 = measure_noise_width_squared(params, client, ct, &dec_rescaled, coeffs_to_measure);
    debug!("log2(measured noise): {}", s_2.log2());

    dec_rescaled
}

pub struct YClient<'a> {
    inner: &'a mut Client<'a>,
    params: &'a Params,
    lwe_client: LWEClient,
}

pub fn get_seed(public_seed_idx: u8) -> [u8; 32] {
    let mut seed = STATIC_PUBLIC_SEED;
    seed[0] = public_seed_idx;
    seed
}

pub fn generate_matrix_ring(
    rng_pub: &mut ChaCha20Rng,
    n: usize,
    rows: usize,
    cols: usize,
) -> Vec<u32> {
    assert_eq!(rows % n, 0);
    assert_eq!(cols % n, 0);
    let rows_outer = rows / n;
    let cols_outer = cols / n;

    let mut out = vec![0u32; rows * cols];
    for i in 0..rows_outer {
        for j in 0..cols_outer {
            let mut a = vec![0u32; n];
            for idx in 0..n {
                a[idx] = rng_pub.sample::<u32, _>(rand::distributions::Standard);
            }

            let mat = negacyclic_matrix_u32(&a);
            for k in 0..n {
                for l in 0..n {
                    let idx = (i * n + k) * cols + (j * n + l);
                    out[idx] = mat[k * n + l];
                }
            }
        }
    }

    out
}

impl<'a> YClient<'a> {
    pub(crate) fn new(inner: &'a mut Client<'a>, params: &'a Params) -> Self {
        Self {
            inner,
            params,
            lwe_client: LWEClient::new(LWEParams::default()),
        }
    }

    fn from_seed(inner: &'a mut Client<'a>, params: &'a Params, client_seed: Seed) -> Self {
        Self {
            inner,
            params,
            lwe_client: LWEClient::from_seed(LWEParams::default(), client_seed),
        }
    }

    fn lwe_client(&self) -> &LWEClient {
        &self.lwe_client
    }

    fn rlwes_to_lwes(&self, ct: &[PolyMatrixRaw<'a>]) -> Vec<u64> {
        let v = ct
            .iter()
            .map(|ct| rlwe_to_lwe(self.params, ct))
            .collect::<Vec<_>>();
        concat_horizontal(&v, self.params.poly_len + 1, self.params.poly_len)
    }

    pub(crate) fn generate_query_impl(
        &self,
        public_seed_idx: u8,
        dim_log2: usize,
        packing: bool,
        index: usize,
    ) -> Vec<PolyMatrixRaw<'a>> {
        // let db_cols = 1 << (self.params.db_dim_2 + self.params.poly_len_log2);
        // let idx_dim1 = index / db_cols;

        let multiply_ct = true;

        let mut rng_pub = ChaCha20Rng::from_seed(get_seed(public_seed_idx));

        // Generate dim1_bits LWE samples under public randomness
        let mut out = Vec::new();

        let scale_k = self.params.modulus / self.params.pt_modulus;

        for i in 0..(1 << dim_log2) {
            let mut scalar = PolyMatrixRaw::zero(self.params, 1, 1);
            let is_nonzero = i == (index / self.params.poly_len);

            if is_nonzero {
                scalar.data[index % self.params.poly_len] = scale_k;
            }

            if packing {
                let factor =
                    invert_uint_mod(self.params.poly_len as u64, self.params.modulus).unwrap();
                scalar = scalar_multiply_alloc(
                    &PolyMatrixRaw::single_value(self.params, factor).ntt(),
                    &scalar.ntt(),
                )
                .raw();
            }

            // if public_seed_idx == SEED_0 {
            //     out.push(scalar.pad_top(1));
            //     continue;
            // }

            let ct = if multiply_ct {
                let factor =
                    invert_uint_mod(self.params.poly_len as u64, self.params.modulus).unwrap();

                self.inner.encrypt_matrix_scaled_reg(
                    &scalar.ntt(),
                    &mut ChaCha20Rng::from_entropy(),
                    &mut rng_pub,
                    factor,
                )
            } else {
                self.inner.encrypt_matrix_reg(
                    &scalar.ntt(),
                    &mut ChaCha20Rng::from_entropy(),
                    &mut rng_pub,
                )
            };

            // let mut ct = self.inner.encrypt_matrix_reg(
            //     &scalar.ntt(),
            //     &mut ChaCha20Rng::from_entropy(),
            //     &mut rng_pub,
            // );

            // if multiply_ct && packing {
            //     let factor =
            //         invert_uint_mod(self.params.poly_len as u64, self.params.modulus).unwrap();
            //     ct = scalar_multiply_alloc(
            //         &PolyMatrixRaw::single_value(self.params, factor).ntt(),
            //         &ct,
            //     );
            // };

            // if multiply_error && is_nonzero && packing {
            //     let factor =
            //         invert_uint_mod(self.params.poly_len as u64, self.params.modulus).unwrap();
            //     ct = scalar_multiply_alloc(
            //         &PolyMatrixRaw::single_value(self.params, factor).ntt(),
            //         &ct,
            //     );
            // }

            let ct_raw = ct.raw();
            // let ct_0_nega = negacyclic_perm(ct_raw.get_poly(0, 0), 0, self.params.modulus);
            // let ct_1_nega = negacyclic_perm(ct_raw.get_poly(1, 0), 0, self.params.modulus);
            // let mut ct_nega = PolyMatrixRaw::zero(self.params, 2, 1);
            // ct_nega.get_poly_mut(0, 0).copy_from_slice(&ct_0_nega);
            // ct_nega.get_poly_mut(1, 0).copy_from_slice(&ct_1_nega);

            // self-test
            // {
            //     let test_ct = self.inner.encrypt_matrix_reg(
            //         &PolyMatrixRaw::single_value(self.params, scale_k * 7).ntt(),
            //         &mut ChaCha20Rng::from_entropy(),
            //         &mut ChaCha20Rng::from_entropy(),
            //     );
            //     let lwe = rlwe_to_lwe(self.params, &test_ct.raw());
            //     let result = self.decode_response(&lwe);
            //     assert_eq!(result[0], 7);
            // }

            out.push(ct_raw);
        }

        out
    }

    fn generate_query(
        &self,
        public_seed_idx: u8,
        dim_log2: usize,
        packing: bool,
        index_row: usize,
    ) -> Vec<u64> {
        if public_seed_idx == SEED_0 && !packing {
            let lwe_params = LWEParams::default();
            let dim = 1 << (dim_log2 + self.params.poly_len_log2);

            // lwes must be (n + 1) x (dim) matrix
            let mut lwes = vec![0u64; (lwe_params.n + 1) * dim];

            let scale_k = lwe_params.scale_k() as u32;
            let mut vals_to_encrypt = vec![0u32; dim];
            vals_to_encrypt[index_row] = scale_k;

            let mut rng_pub = ChaCha20Rng::from_seed(get_seed(public_seed_idx));

            for i in (0..dim).step_by(lwe_params.n) {
                let out = self
                    .lwe_client
                    .encrypt_many(&mut rng_pub, &vals_to_encrypt[i..i + lwe_params.n])
                    .iter()
                    .map(|x| *x as u64)
                    .collect::<Vec<_>>();
                assert_eq!(out.len(), (lwe_params.n + 1) * lwe_params.n);
                for r in 0..lwe_params.n + 1 {
                    for c in 0..lwe_params.n {
                        lwes[r * dim + i + c] = out[r * lwe_params.n + c];
                    }
                }
            }

            lwes
        } else {
            let out = self.generate_query_impl(public_seed_idx, dim_log2, packing, index_row);
            let lwes = self.rlwes_to_lwes(&out);
            lwes
        }
    }

    fn generate_query_lwe_low_mem(
        &self,
        public_seed_idx: u8,
        dim_log2: usize,
        packing: bool,
        index_row: usize,
    ) -> Vec<u64> {
        let index = index_row;
        let multiply_ct = true;
        let mut rng_pub = ChaCha20Rng::from_seed(get_seed(public_seed_idx));

        // Generate dim1_bits LWE samples under public randomness
        let mut out = Vec::new();

        let scale_k = self.params.modulus / self.params.pt_modulus;

        for i in 0..(1 << dim_log2) {
            let mut scalar = PolyMatrixRaw::zero(self.params, 1, 1);
            let is_nonzero = i == (index / self.params.poly_len);

            if is_nonzero {
                scalar.data[index % self.params.poly_len] = scale_k;
            }

            if packing {
                let factor =
                    invert_uint_mod(self.params.poly_len as u64, self.params.modulus).unwrap();
                scalar = scalar_multiply_alloc(
                    &PolyMatrixRaw::single_value(self.params, factor).ntt(),
                    &scalar.ntt(),
                )
                .raw();
            }

            let ct = if multiply_ct {
                let factor =
                    invert_uint_mod(self.params.poly_len as u64, self.params.modulus).unwrap();

                self.inner.encrypt_matrix_scaled_reg(
                    &scalar.ntt(),
                    &mut ChaCha20Rng::from_entropy(),
                    &mut rng_pub,
                    factor,
                )
            } else {
                self.inner.encrypt_matrix_reg(
                    &scalar.ntt(),
                    &mut ChaCha20Rng::from_entropy(),
                    &mut rng_pub,
                )
            };

            let ct_raw = ct.raw();

            // only care about the last row
            let lwe_last_row = ct_raw.get_poly(1, 0);

            out.extend_from_slice(lwe_last_row);
        }
        out
    }

    fn generate_full_query(
        &self,
        target_idx: usize,
    ) -> (Vec<u32>, AlignedMemory64, AlignedMemory64) {
        // setup
        let db_rows = 1 << (self.params.db_dim_1 + self.params.poly_len_log2);
        let db_cols = 1 << (self.params.db_dim_2 + self.params.poly_len_log2);
        let target_row = target_idx / db_cols;
        let target_col = target_idx % db_cols;
        debug!(
            "Target item: {} ({}, {})",
            target_idx, target_row, target_col
        );

        // generate pub params
        let sk_reg = self.client().get_sk_reg();
        let pack_pub_params = raw_generate_expansion_params(
            self.params,
            &sk_reg,
            self.params.poly_len_log2,
            self.params.t_exp_left,
            &mut ChaCha20Rng::from_entropy(),
            &mut ChaCha20Rng::from_seed(STATIC_SEED_2),
        );
        // let pub_params_size = get_vec_pm_size_bytes(&pack_pub_params) / 2;
        let mut pack_pub_params_row_1s = pack_pub_params.to_vec();
        for i in 0..pack_pub_params.len() {
            pack_pub_params_row_1s[i] =
                pack_pub_params[i].submatrix(1, 0, 1, pack_pub_params[i].cols);
            pack_pub_params_row_1s[i] = condense_matrix(self.params, &pack_pub_params_row_1s[i]);
        }
        let pub_params_size = get_vec_pm_size_bytes(&pack_pub_params_row_1s);
        debug!("pub params size: {} bytes", pub_params_size);
        let pack_pub_params_row_1s_pm = pack_vec_pm(
            self.params,
            1,
            self.params.t_exp_left,
            &pack_pub_params_row_1s,
        );
        // 11*3*2048*8 bytes (technically 7)
        assert_eq!(
            pack_pub_params_row_1s_pm.len() * std::mem::size_of::<u64>(),
            self.params.poly_len_log2
                * self.params.t_exp_left
                * self.params.poly_len
                * std::mem::size_of::<u64>()
        );

        // generate query
        let query_row = self.generate_query(SEED_0, self.params.db_dim_1, false, target_row);
        let query_row_last_row: &[u64] = &query_row[self.lwe_params().n * db_rows..];
        let mut aligned_query_packed = AlignedMemory64::new(query_row_last_row.len());
        aligned_query_packed
            .as_mut_slice()
            .copy_from_slice(&query_row_last_row);
        let packed_query_row = aligned_query_packed;
        let packed_query_row_u32 = packed_query_row
            .as_slice()
            .iter()
            .map(|x| *x as u32)
            .chain(std::iter::repeat(0).take(self.params.db_rows_padded_normal() - db_rows))
            .collect::<Vec<_>>();
        assert_eq!(
            packed_query_row_u32.len(),
            self.params.db_rows_padded_normal()
        );

        let query_col = self.generate_query(SEED_1, self.params.db_dim_2, true, target_col);
        let query_col_last_row = &query_col[self.params.poly_len * db_cols..];
        let packed_query_col = pack_query(self.params, query_col_last_row);
        assert_eq!(packed_query_col.len(), db_cols);
        (
            packed_query_row_u32,
            packed_query_col,
            pack_pub_params_row_1s_pm,
        )
    }

    /// Build the YPIR `pack_pub_params` blob (`query.1` on the wire).
    ///
    /// Default behavior is identical to the inline construction previously
    /// embedded in [`generate_full_query_simplepir`]: the secret RNG is drawn
    /// from `OsRng` (`ChaCha20Rng::from_entropy()`), so two calls under the
    /// same `client_seed` produce *different* `pp` blobs (each batch gets a
    /// fresh `pp`). The public RNG is seeded from the deterministic
    /// [`STATIC_SEED_2`] so the server's view of the public stream is fixed.
    fn build_pack_pub_params(&self) -> AlignedMemory64 {
        self.build_pack_pub_params_with_secret_rng(&mut ChaCha20Rng::from_entropy())
    }

    /// [`build_pack_pub_params`] with the secret RNG injected. Intended for
    /// tests that need byte-deterministic `pp` output. Production callers go
    /// through [`build_pack_pub_params`] which uses `OsRng` entropy.
    fn build_pack_pub_params_with_secret_rng(
        &self,
        secret_rng: &mut ChaCha20Rng,
    ) -> AlignedMemory64 {
        let sk_reg = self.client().get_sk_reg();
        let pack_pub_params = raw_generate_expansion_params(
            self.params,
            &sk_reg,
            self.params.poly_len_log2,
            self.params.t_exp_left,
            secret_rng,
            &mut ChaCha20Rng::from_seed(STATIC_SEED_2),
        );
        // let pub_params_size = get_vec_pm_size_bytes(&pack_pub_params) / 2;
        let mut pack_pub_params_row_1s = pack_pub_params.to_vec();
        for i in 0..pack_pub_params.len() {
            pack_pub_params_row_1s[i] =
                pack_pub_params[i].submatrix(1, 0, 1, pack_pub_params[i].cols);
            pack_pub_params_row_1s[i] = condense_matrix(self.params, &pack_pub_params_row_1s[i]);
        }
        let pub_params_size = get_vec_pm_size_bytes(&pack_pub_params_row_1s);
        debug!("pub params size: {} bytes", pub_params_size);
        let pack_pub_params_row_1s_pm = pack_vec_pm(
            self.params,
            1,
            self.params.t_exp_left,
            &pack_pub_params_row_1s,
        );
        assert_eq!(
            pack_pub_params_row_1s_pm.len() * std::mem::size_of::<u64>(),
            self.params.poly_len_log2
                * self.params.t_exp_left
                * self.params.poly_len
                * std::mem::size_of::<u64>()
        );
        pack_pub_params_row_1s_pm
    }

    fn generate_full_query_simplepir(
        &self,
        target_idx: u64,
    ) -> (AlignedMemory64, AlignedMemory64) {
        // setup
        let db_rows = 1 << (self.params.db_dim_1 + self.params.poly_len_log2);
        let db_cols = self.params.instances * self.params.poly_len;

        let target_row = (target_idx / db_cols as u64) as usize;
        let target_col = (target_idx % db_cols as u64) as usize;
        debug!(
            "Target item: {} ({}, {})",
            target_idx, target_row, target_col
        );

        let pack_pub_params_row_1s_pm = self.build_pack_pub_params();

        // generate query
        // NB: made this low memory
        // let query_row = self.generate_query(SEED_0, self.params.db_dim_1, true, target_row);
        // assert_eq!(query_row.len(), (self.params.poly_len + 1) * db_rows);
        // let query_row_last_row: &[u64] = &query_row[self.params.poly_len * db_rows..];
        // assert_eq!(query_row_last_row.len(), db_rows);
        let query_row_last_row =
            self.generate_query_lwe_low_mem(SEED_0, self.params.db_dim_1, true, target_row);
        assert_eq!(query_row_last_row.len(), db_rows);
        let packed_query_row = pack_query(self.params, &query_row_last_row);
        assert_eq!(packed_query_row.len(), self.params.db_rows());

        (packed_query_row, pack_pub_params_row_1s_pm)
    }

    /// Batched analogue of [`generate_full_query_simplepir`]. Builds
    /// `pack_pub_params` once for the whole batch and one `q.0` per target
    /// row, all under the same `sk_reg` (same `client_seed`).
    ///
    /// Each per-row `q.0` still draws its own fresh `from_entropy()` secret
    /// RNG inside [`generate_query_lwe_low_mem`], so the LWE error vectors
    /// `e_k` remain independent across the K queries even though `s` is
    /// shared.
    fn generate_full_query_simplepir_batch(
        &self,
        target_rows: &[usize],
    ) -> (Vec<AlignedMemory64>, AlignedMemory64) {
        let db_rows = 1 << (self.params.db_dim_1 + self.params.poly_len_log2);

        let pack_pub_params_row_1s_pm = self.build_pack_pub_params();

        let queries = target_rows
            .iter()
            .map(|&target_row| {
                let q_last_row = self.generate_query_lwe_low_mem(
                    SEED_0,
                    self.params.db_dim_1,
                    true,
                    target_row,
                );
                assert_eq!(q_last_row.len(), db_rows);
                let packed = pack_query(self.params, &q_last_row);
                assert_eq!(packed.len(), self.params.db_rows());
                packed
            })
            .collect();

        (queries, pack_pub_params_row_1s_pm)
    }

    fn lwe_params(&self) -> &LWEParams {
        self.lwe_client().lwe_params()
    }

    fn decode_response(&self, response: &[u64]) -> Vec<u64> {
        debug!("Decoding response: {:?}", &response[..16]);
        let db_cols = 1 << (self.params.db_dim_2 + self.params.poly_len_log2);

        let sk = self.inner.get_sk_reg().as_slice().to_vec();

        let mut out = Vec::new();
        for col in 0..db_cols {
            let mut sum = 0u128;
            for i in 0..self.params.poly_len {
                let v1 = response[i * db_cols + col];
                let v2 = sk[i];
                sum += v1 as u128 * v2 as u128;
            }

            sum += response[self.params.poly_len * db_cols + col] as u128;

            let result = (sum % self.params.modulus as u128) as u64;
            let result_rescaled = rescale(result, self.params.modulus, self.params.pt_modulus);
            out.push(result_rescaled);
        }

        out
    }

    fn client(&self) -> &Client<'a> {
        self.inner
    }
}

pub struct YPIRClient {
    params: Params,
}

pub type YPIRQuery = (Vec<u32>, AlignedMemory64, AlignedMemory64);
pub type YPIRSimpleQuery = (AlignedMemory64, AlignedMemory64);
/// Output of [`YPIRClient::generate_query_simplepir_batch`]: K SimplePIR
/// `q.0` query vectors plus a single shared `pack_pub_params` (`q.1`)
/// generated under one `client_seed`.
pub type YPIRSimpleBatchQuery = (Vec<AlignedMemory64>, AlignedMemory64);

pub const SHA1_HASH_BYTES: usize = 20;

impl YPIRClient {
    pub fn new(params: &Params) -> Self {
        Self {
            params: params.clone(),
        }
    }

    pub fn hash(target_item: &str) -> [u8; SHA1_HASH_BYTES] {
        let mut hasher = Sha1::new();
        hasher.update(target_item.as_bytes());
        let item_hash = hasher.finalize();
        item_hash.into()
    }

    pub fn bucket(log2_num_items: usize, target_item: &str) -> usize {
        let item_hash = Self::hash(target_item);

        let top_idx = u32::from_be_bytes(item_hash[0..4].try_into().unwrap());
        let bucket = top_idx >> (32 - log2_num_items);
        bucket as usize
    }

    pub fn from_db_sz(num_items: u64, item_size_bits: u64, is_simplepir: bool) -> Self {
        let params = if is_simplepir {
            params_for_scenario_simplepir(num_items, item_size_bits)
        } else {
            params_for_scenario(num_items, item_size_bits)
        };
        Self::new(&params)
    }

    pub fn generate_query_normal(&self, target_idx: usize) -> (YPIRQuery, Seed) {
        let client_seed = generate_secure_random_seed();
        let mut client = Client::init(&self.params);
        client.generate_secret_keys_from_seed(client_seed);
        let y_client = YClient::from_seed(&mut client, &self.params, client_seed);
        let query = y_client.generate_full_query(target_idx);
        (query, client_seed)
    }

    pub fn generate_query_simplepir(&self, target_row: usize) -> (YPIRSimpleQuery, Seed) {
        assert!(target_row < self.params.db_rows());
        let target_idx = target_row as u64 * self.params.db_cols_simplepir() as u64;
        let client_seed = generate_secure_random_seed();
        let mut client = Client::init(&self.params);
        client.generate_secret_keys_from_seed(client_seed);
        let y_client = YClient::from_seed(&mut client, &self.params, client_seed);
        let query = y_client.generate_full_query_simplepir(target_idx);
        (query, client_seed)
    }

    pub fn decode_response_normal(&self, client_seed: Seed, response_data: &[u8]) -> u64 {
        let mut client = Client::init(&self.params);
        client.generate_secret_keys_from_seed(client_seed);
        let y_client = YClient::from_seed(&mut client, &self.params, client_seed);
        let out =
            YPIRClient::decode_response_normal_yclient(&self.params, &y_client, response_data);
        out
    }

    pub fn decode_response_simplepir(&self, client_seed: Seed, response_data: &[u8]) -> Vec<u8> {
        let decoded = self.decode_response_simplepir_raw(client_seed, response_data);
        u64s_to_contiguous_bytes(&decoded, self.params.pt_modulus_bits())
    }

    pub fn decode_response_simplepir_raw(
        &self,
        client_seed: Seed,
        response_data: &[u8],
    ) -> Vec<u64> {
        let mut client = Client::init(&self.params);
        client.generate_secret_keys_from_seed(client_seed);
        let y_client = YClient::from_seed(&mut client, &self.params, client_seed);
        YPIRClient::decode_response_simplepir_yclient(&self.params, &y_client, response_data)
    }

    /// Batched analogue of [`generate_query_simplepir`]. Generates K SimplePIR
    /// queries that share one `pack_pub_params` and one `client_seed` (one
    /// `s`). Per-query LWE error vectors `e_k` remain independent because
    /// each `q.0` is generated with a fresh `OsRng` secret RNG.
    ///
    /// Callers (e.g. `pir-client`'s upcoming `client_batch_query`) should
    /// generate a fresh batch — and therefore a fresh `client_seed` — for
    /// every delegation, never reusing `client_seed` across batches.
    pub fn generate_query_simplepir_batch(
        &self,
        target_rows: &[usize],
    ) -> (YPIRSimpleBatchQuery, Seed) {
        for &row in target_rows {
            assert!(row < self.params.db_rows());
        }
        let client_seed = generate_secure_random_seed();
        let mut client = Client::init(&self.params);
        client.generate_secret_keys_from_seed(client_seed);
        let y_client = YClient::from_seed(&mut client, &self.params, client_seed);
        let query = y_client.generate_full_query_simplepir_batch(target_rows);
        (query, client_seed)
    }

    /// Batched analogue of [`decode_response_simplepir`]. Decodes K
    /// independent SimplePIR responses under one shared `client_seed`,
    /// returning the per-query plaintext byte streams or per-query errors.
    ///
    /// Each response chunk decodes independently. The K decodes share one
    /// `YClient` (one `s`) but each slot is wrapped in `catch_unwind`, so a
    /// malformed or adversarial response in one slot returns `Err` for that
    /// slot without preventing later slots from decoding.
    pub fn decode_response_simplepir_batch(
        &self,
        client_seed: Seed,
        responses: &[&[u8]],
    ) -> Vec<Result<Vec<u8>, YPIRDecodeError>> {
        let raws = self.decode_response_simplepir_batch_raw(client_seed, responses);
        raws.into_iter()
            .map(|r| r.map(|raw| u64s_to_contiguous_bytes(&raw, self.params.pt_modulus_bits())))
            .collect()
    }

    /// Raw (`Vec<u64>`-per-response) variant of
    /// [`decode_response_simplepir_batch`].
    pub fn decode_response_simplepir_batch_raw(
        &self,
        client_seed: Seed,
        responses: &[&[u8]],
    ) -> Vec<Result<Vec<u64>, YPIRDecodeError>> {
        let mut client = Client::init(&self.params);
        client.generate_secret_keys_from_seed(client_seed);
        let y_client = YClient::from_seed(&mut client, &self.params, client_seed);
        responses
            .iter()
            .map(|r| {
                panic::catch_unwind(panic::AssertUnwindSafe(|| {
                    YPIRClient::decode_response_simplepir_yclient(&self.params, &y_client, r)
                }))
                .map_err(|payload| YPIRDecodeError::Panic(panic_payload_message(payload)))
            })
            .collect()
    }

    fn decode_response_normal_yclient(
        params: &Params,
        y_client: &YClient,
        response_data: &[u8],
    ) -> u64 {
        let num_rlwe_outputs = params.rho();
        assert_eq!(response_data.len() % num_rlwe_outputs, 0);
        let response_vecs = response_data
            .chunks_exact(response_data.len() / num_rlwe_outputs)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();

        // rescale
        let rlwe_q_prime_1 = params.get_q_prime_1();
        let rlwe_q_prime_2 = params.get_q_prime_2();
        let mut response: Vec<PolyMatrixRaw> = Vec::new();
        for ct_bytes in response_vecs.iter() {
            let ct = PolyMatrixRaw::recover(&params, rlwe_q_prime_1, rlwe_q_prime_2, ct_bytes);
            response.push(ct);
        }

        let lwe_params = LWEParams::default();
        let lwe_q_prime_bits = lwe_params.q2_bits as usize;
        let pt_bits = (params.pt_modulus as f64).log2().floor() as usize;
        let blowup_factor = lwe_q_prime_bits as f64 / pt_bits as f64;

        // decrypt
        let outer_ct = response
            .iter()
            .flat_map(|ct| {
                decrypt_ct_reg_measured(y_client.client(), &params, &ct.ntt(), params.poly_len)
                    .as_slice()
                    .to_vec()
            })
            .collect::<Vec<_>>();
        // assert_eq!(outer_ct.len(), out_rows);
        // debug!("outer_ct: {:?}", &outer_ct[..]);
        let outer_ct_t_u8 = u64s_to_contiguous_bytes(&outer_ct, pt_bits);

        let mut inner_ct = PolyMatrixRaw::zero(&params, 2, 1);
        let mut bit_offs = 0;
        let lwe_q_prime = lwe_params.get_q_prime_2();
        let special_offs =
            ((lwe_params.n * lwe_q_prime_bits) as f64 / pt_bits as f64).ceil() as usize;
        for z in 0..lwe_params.n {
            let val = read_bits(&outer_ct_t_u8, bit_offs, lwe_q_prime_bits);
            bit_offs += lwe_q_prime_bits;
            assert!(
                val < lwe_q_prime,
                "val: {}, lwe_q_prime: {}",
                val,
                lwe_q_prime
            );
            inner_ct.data[z] = rescale(val, lwe_q_prime, lwe_params.modulus);
        }

        let mut val = 0;
        for i in 0..blowup_factor.ceil() as usize {
            val |= outer_ct[special_offs + i] << (i * pt_bits);
        }
        assert!(
            val < lwe_q_prime,
            "val: {}, lwe_q_prime: {}",
            val,
            lwe_q_prime
        );
        debug!("got b_val of: {}", val);
        inner_ct.data[lwe_params.n] = rescale(val, lwe_q_prime, lwe_params.modulus);

        debug!("decrypting inner ct...");
        // let plaintext = decrypt_ct_reg_measured(y_client.client(), &params, &inner_ct.ntt(), 1);
        // let final_result = plaintext.data[0];
        let inner_ct_as_u32 = inner_ct
            .as_slice()
            .iter()
            .take(lwe_params.n + 1)
            .map(|x| *x as u32)
            .collect::<Vec<_>>();
        let decrypted = y_client.lwe_client().decrypt(&inner_ct_as_u32);
        let final_result = rescale(decrypted as u64, lwe_params.modulus, lwe_params.pt_modulus);

        final_result
    }

    fn decode_response_simplepir_yclient(
        params: &Params,
        y_client: &YClient,
        response_data: &[u8],
    ) -> Vec<u64> {
        let db_cols = params.instances * params.poly_len;
        let num_rlwe_outputs = db_cols / params.poly_len;

        assert_eq!(response_data.len() % num_rlwe_outputs, 0);
        let response_vecs = response_data
            .chunks_exact(response_data.len() / num_rlwe_outputs)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();

        // rescale
        let rlwe_q_prime_1 = params.get_q_prime_1();
        let rlwe_q_prime_2 = params.get_q_prime_2();
        let mut response = Vec::new();
        for ct_bytes in response_vecs.iter() {
            let ct = PolyMatrixRaw::recover(&params, rlwe_q_prime_1, rlwe_q_prime_2, ct_bytes);
            response.push(ct);
        }

        debug!("decrypting outer cts...");
        let outer_ct = response
            .iter()
            .flat_map(|ct| {
                decrypt_ct_reg_measured(y_client.client(), &params, &ct.ntt(), params.poly_len)
                    .as_slice()
                    .to_vec()
            })
            .collect::<Vec<_>>();
        assert_eq!(outer_ct.len(), num_rlwe_outputs * params.poly_len);
        // debug!("outer_ct: {:?}", &outer_ct[..]);
        // let outer_ct_t_u8 = u64s_to_contiguous_bytes(&outer_ct, pt_bits);
        outer_ct
    }

    pub fn params(&self) -> &Params {
        &self.params
    }
}

#[cfg(test)]
mod test {
    use spiral_rs::arith::barrett_reduction_u128;

    use super::*;

    #[test]
    fn test_lwe() {
        let lwe_params = LWEParams::default();
        let client = LWEClient::new(lwe_params.clone());
        let pt = fastrand::u32(0..lwe_params.pt_modulus as u32);
        let scaled_pt = pt.wrapping_mul(lwe_params.scale_k() as u32);
        let ct = client.encrypt(&mut ChaCha20Rng::from_entropy(), scaled_pt);
        let pt_dec = client.decrypt(&ct);
        let result = rescale(pt_dec as u64, lwe_params.modulus, lwe_params.pt_modulus) as u32;
        assert_eq!(result, pt);
    }

    #[test]
    #[ignore]
    fn test_linear_accumulation_noise() {
        let params = params_for_scenario(1 << 43, 1);
        let upper_n = 1 << (11 + 6);

        let mut client = Client::init(&params);
        client.generate_secret_keys();
        let y_client = YClient::new(&mut client, &params);
        let target_idx = 0;
        let query = y_client.generate_query(SEED_0, params.db_dim_1, false, target_idx);

        let db = (0..upper_n)
            .map(|_| fastrand::u64(0..params.pt_modulus))
            .collect::<Vec<_>>();

        let mut acc = vec![0u128; params.poly_len + 1];
        for idx in 0..upper_n {
            for dim in 0..params.poly_len + 1 {
                let query_val = query[dim * upper_n + idx];
                let db_val = db[idx];
                let product = query_val as u128 * db_val as u128;
                acc[dim] += product;
            }
        }

        let mut ct = PolyMatrixRaw::zero(&params, 2, 1);
        for dim in 0..params.poly_len + 1 {
            ct.data[dim] = barrett_reduction_u128(&params, acc[dim]);
        }

        let _plaintext =
            decrypt_ct_reg_measured(y_client.client(), &params, &ct.ntt(), params.poly_len);
        todo!("problem w test: negacyclic");
    }
}

// ---------------------------------------------------------------------------
// Area 1: Malformed server response handling
// ---------------------------------------------------------------------------
#[cfg(test)]
mod malformed_response_tests {
    use super::*;
    use crate::params::{params_for_scenario, GetQPrime, GetRho};

    fn make_test_params() -> Params {
        params_for_scenario(1 << 20, 1)
    }

    fn make_yclient_from_seed(params: &Params, seed: Seed) -> (Client, Seed) {
        let mut client = Client::init(params);
        client.generate_secret_keys_from_seed(seed);
        (client, seed)
    }

    fn fixed_seed() -> Seed {
        [42u8; 32]
    }

    // -- YClient::decode_response --

    #[test]
    #[should_panic]
    fn decode_response_empty_input_panics() {
        let params = make_test_params();
        let (mut client, seed) = make_yclient_from_seed(&params, fixed_seed());
        let y_client = YClient::from_seed(&mut client, &params, seed);
        let _ = y_client.decode_response(&[]);
    }

    #[test]
    #[should_panic]
    fn decode_response_truncated_input_panics() {
        let params = make_test_params();
        let (mut client, seed) = make_yclient_from_seed(&params, fixed_seed());
        let y_client = YClient::from_seed(&mut client, &params, seed);
        let db_cols = 1usize << (params.db_dim_2 + params.poly_len_log2);
        let expected_len = (params.poly_len + 1) * db_cols;
        let _ = y_client.decode_response(&vec![0u64; expected_len / 2]);
    }

    #[test]
    fn decode_response_all_zeros() {
        let params = make_test_params();
        let (mut client, seed) = make_yclient_from_seed(&params, fixed_seed());
        let y_client = YClient::from_seed(&mut client, &params, seed);
        let db_cols = 1usize << (params.db_dim_2 + params.poly_len_log2);
        let expected_len = (params.poly_len + 1) * db_cols;
        let result = y_client.decode_response(&vec![0u64; expected_len]);
        assert_eq!(result.len(), db_cols);
        for val in &result {
            assert_eq!(*val, 0);
        }
    }

    #[test]
    fn decode_response_max_modular_values_does_not_panic() {
        let params = make_test_params();
        let (mut client, seed) = make_yclient_from_seed(&params, fixed_seed());
        let y_client = YClient::from_seed(&mut client, &params, seed);
        let db_cols = 1usize << (params.db_dim_2 + params.poly_len_log2);
        let expected_len = (params.poly_len + 1) * db_cols;
        let result = y_client.decode_response(&vec![params.modulus - 1; expected_len]);
        assert_eq!(result.len(), db_cols);
    }

    #[test]
    fn decode_response_u64_max_overflows_accumulator() {
        let params = make_test_params();
        let (mut client, seed) = make_yclient_from_seed(&params, fixed_seed());
        let y_client = YClient::from_seed(&mut client, &params, seed);
        let db_cols = 1usize << (params.db_dim_2 + params.poly_len_log2);
        let expected_len = (params.poly_len + 1) * db_cols;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            y_client.decode_response(&vec![u64::MAX; expected_len])
        }));
        assert!(result.is_err(), "u64::MAX inputs should overflow the u128 accumulator");
    }

    #[test]
    fn decode_response_oversized_input_returns_correct_length() {
        let params = make_test_params();
        let (mut client, seed) = make_yclient_from_seed(&params, fixed_seed());
        let y_client = YClient::from_seed(&mut client, &params, seed);
        let db_cols = 1usize << (params.db_dim_2 + params.poly_len_log2);
        let expected_len = (params.poly_len + 1) * db_cols;
        let result = y_client.decode_response(&vec![0u64; expected_len + 1000]);
        assert_eq!(result.len(), db_cols);
    }

    #[test]
    fn decode_response_random_data_does_not_panic() {
        let params = make_test_params();
        let (mut client, seed) = make_yclient_from_seed(&params, fixed_seed());
        let y_client = YClient::from_seed(&mut client, &params, seed);
        let db_cols = 1usize << (params.db_dim_2 + params.poly_len_log2);
        let expected_len = (params.poly_len + 1) * db_cols;
        let data: Vec<u64> = (0..expected_len).map(|_| fastrand::u64(..) % params.modulus).collect();
        let result = y_client.decode_response(&data);
        assert_eq!(result.len(), db_cols);
        for val in &result {
            assert!(*val < params.pt_modulus, "output exceeds plaintext modulus");
        }
    }

    #[test]
    fn decode_response_deterministic_across_calls() {
        let params = make_test_params();
        let seed = fixed_seed();
        let db_cols = 1usize << (params.db_dim_2 + params.poly_len_log2);
        let expected_len = (params.poly_len + 1) * db_cols;
        let data: Vec<u64> = (0..expected_len).map(|i| (i as u64 * 7) % params.modulus).collect();

        let r1 = {
            let (mut c, s) = make_yclient_from_seed(&params, seed);
            YClient::from_seed(&mut c, &params, s).decode_response(&data)
        };
        let r2 = {
            let (mut c, s) = make_yclient_from_seed(&params, seed);
            YClient::from_seed(&mut c, &params, s).decode_response(&data)
        };
        assert_eq!(r1, r2);
    }

    // -- decode_response_normal_yclient --

    fn expected_normal_response_byte_len(params: &Params) -> usize {
        let num_rlwe_outputs = params.rho();
        let q_prime_1 = params.get_q_prime_1();
        let q_prime_2 = params.get_q_prime_2();
        let q_1_bits = (q_prime_2 as f64).log2().ceil() as usize;
        let q_2_bits = (q_prime_1 as f64).log2().ceil() as usize;
        let per_ct_bits = (q_1_bits + q_2_bits) * params.poly_len;
        let per_ct_bytes = (per_ct_bits + 7) / 8;
        per_ct_bytes * num_rlwe_outputs
    }

    #[test]
    #[should_panic]
    fn decode_normal_empty_response_panics() {
        let params = make_test_params();
        let (mut client, _) = make_yclient_from_seed(&params, fixed_seed());
        let y_client = YClient::from_seed(&mut client, &params, fixed_seed());
        let _ = YPIRClient::decode_response_normal_yclient(&params, &y_client, &[]);
    }

    #[test]
    #[should_panic]
    fn decode_normal_single_byte_panics() {
        let params = make_test_params();
        let (mut client, _) = make_yclient_from_seed(&params, fixed_seed());
        let y_client = YClient::from_seed(&mut client, &params, fixed_seed());
        let _ = YPIRClient::decode_response_normal_yclient(&params, &y_client, &[0xFF]);
    }

    #[test]
    #[should_panic]
    fn decode_normal_misaligned_length_panics() {
        let params = make_test_params();
        let len = expected_normal_response_byte_len(&params);
        let (mut client, _) = make_yclient_from_seed(&params, fixed_seed());
        let y_client = YClient::from_seed(&mut client, &params, fixed_seed());
        let _ = YPIRClient::decode_response_normal_yclient(&params, &y_client, &vec![0u8; len + 1]);
    }

    #[test]
    #[should_panic]
    fn decode_normal_truncated_response_panics() {
        let params = make_test_params();
        let len = expected_normal_response_byte_len(&params);
        let (mut client, _) = make_yclient_from_seed(&params, fixed_seed());
        let y_client = YClient::from_seed(&mut client, &params, fixed_seed());
        let _ = YPIRClient::decode_response_normal_yclient(&params, &y_client, &vec![0u8; len / 2]);
    }

    #[test]
    fn decode_normal_all_zeros_does_not_leak_via_panic() {
        let params = make_test_params();
        let len = expected_normal_response_byte_len(&params);
        let zeros = vec![0u8; len];
        let (sa, sb) = ([1u8; 32], [2u8; 32]);

        let ra = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (mut c, _) = make_yclient_from_seed(&params, sa);
            let y = YClient::from_seed(&mut c, &params, sa);
            YPIRClient::decode_response_normal_yclient(&params, &y, &zeros)
        }));
        let rb = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (mut c, _) = make_yclient_from_seed(&params, sb);
            let y = YClient::from_seed(&mut c, &params, sb);
            YPIRClient::decode_response_normal_yclient(&params, &y, &zeros)
        }));
        assert_eq!(ra.is_ok(), rb.is_ok(),
            "All-zeros response must produce the same success/failure outcome regardless of secret key");
    }

    /// KNOWN VULNERABILITY: random adversarial response can trigger selective-failure side channel.
    /// The assert!(val < lwe_q_prime) in decode_response_normal_yclient panics for some keys
    /// but not others on the same random input.
    #[test]
    #[ignore = "documents selective-failure side channel (known vulnerability)"]
    fn decode_normal_random_response_same_outcome_different_keys() {
        let params = make_test_params();
        let len = expected_normal_response_byte_len(&params);
        let resp: Vec<u8> = (0..len).map(|_| fastrand::u8(..)).collect();
        let (sa, sb) = ([10u8; 32], [20u8; 32]);

        let ra = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (mut c, _) = make_yclient_from_seed(&params, sa);
            let y = YClient::from_seed(&mut c, &params, sa);
            YPIRClient::decode_response_normal_yclient(&params, &y, &resp)
        }));
        let rb = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (mut c, _) = make_yclient_from_seed(&params, sb);
            let y = YClient::from_seed(&mut c, &params, sb);
            YPIRClient::decode_response_normal_yclient(&params, &y, &resp)
        }));
        assert_eq!(ra.is_ok(), rb.is_ok(),
            "Random adversarial response must produce the same success/failure outcome regardless of secret key");
    }

    /// KNOWN VULNERABILITY: selective-failure side channel in decode_response_normal_yclient.
    /// The assert!(val < lwe_q_prime) checks panic for some keys but not others on the same input.
    #[test]
    #[ignore = "documents selective-failure side channel (known vulnerability)"]
    fn decode_normal_max_value_bytes_same_outcome_different_keys() {
        let params = make_test_params();
        let len = expected_normal_response_byte_len(&params);
        let resp = vec![0xFFu8; len];
        let (sa, sb) = ([30u8; 32], [40u8; 32]);

        let ra = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (mut c, _) = make_yclient_from_seed(&params, sa);
            let y = YClient::from_seed(&mut c, &params, sa);
            YPIRClient::decode_response_normal_yclient(&params, &y, &resp)
        }));
        let rb = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (mut c, _) = make_yclient_from_seed(&params, sb);
            let y = YClient::from_seed(&mut c, &params, sb);
            YPIRClient::decode_response_normal_yclient(&params, &y, &resp)
        }));
        assert_eq!(ra.is_ok(), rb.is_ok(),
            "Max-value response must produce the same success/failure outcome regardless of secret key");
    }

    #[test]
    fn decode_normal_deterministic_for_same_seed() {
        let params = make_test_params();
        let len = expected_normal_response_byte_len(&params);
        let resp: Vec<u8> = (0..len).map(|i| (i % 256) as u8).collect();
        let seed = fixed_seed();

        let r1 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (mut c, _) = make_yclient_from_seed(&params, seed);
            let y = YClient::from_seed(&mut c, &params, seed);
            YPIRClient::decode_response_normal_yclient(&params, &y, &resp)
        }));
        let r2 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (mut c, _) = make_yclient_from_seed(&params, seed);
            let y = YClient::from_seed(&mut c, &params, seed);
            YPIRClient::decode_response_normal_yclient(&params, &y, &resp)
        }));
        match (r1, r2) {
            (Ok(v1), Ok(v2)) => assert_eq!(v1, v2),
            (Err(_), Err(_)) => {}
            _ => panic!("Inconsistent panic behavior across identical decode calls"),
        }
    }

    // -- decode_response_simplepir_yclient --

    fn make_simplepir_params() -> Params {
        crate::params::params_for_scenario_simplepir(1 << 14, 2048 * 14)
    }

    fn expected_simplepir_response_byte_len(params: &Params) -> usize {
        let db_cols = params.instances * params.poly_len;
        let num_rlwe_outputs = db_cols / params.poly_len;
        let q_prime_1 = params.get_q_prime_1();
        let q_prime_2 = params.get_q_prime_2();
        let q_1_bits = (q_prime_2 as f64).log2().ceil() as usize;
        let q_2_bits = (q_prime_1 as f64).log2().ceil() as usize;
        let per_ct_bits = (q_1_bits + q_2_bits) * params.poly_len;
        let per_ct_bytes = (per_ct_bits + 7) / 8;
        per_ct_bytes * num_rlwe_outputs
    }

    #[test]
    #[should_panic]
    fn decode_simplepir_empty_response_panics() {
        let params = make_simplepir_params();
        let (mut client, _) = make_yclient_from_seed(&params, fixed_seed());
        let y_client = YClient::from_seed(&mut client, &params, fixed_seed());
        let _ = YPIRClient::decode_response_simplepir_yclient(&params, &y_client, &[]);
    }

    #[test]
    #[should_panic]
    fn decode_simplepir_truncated_response_panics() {
        let params = make_simplepir_params();
        let len = expected_simplepir_response_byte_len(&params);
        let (mut client, _) = make_yclient_from_seed(&params, fixed_seed());
        let y_client = YClient::from_seed(&mut client, &params, fixed_seed());
        let _ = YPIRClient::decode_response_simplepir_yclient(&params, &y_client, &vec![0u8; len / 2]);
    }

    #[test]
    #[should_panic]
    fn decode_simplepir_misaligned_length_panics() {
        let params = make_simplepir_params();
        let len = expected_simplepir_response_byte_len(&params);
        let (mut client, _) = make_yclient_from_seed(&params, fixed_seed());
        let y_client = YClient::from_seed(&mut client, &params, fixed_seed());
        let _ = YPIRClient::decode_response_simplepir_yclient(&params, &y_client, &vec![0u8; len + 1]);
    }

    #[test]
    fn decode_simplepir_all_zeros_does_not_leak_via_panic() {
        let params = make_simplepir_params();
        let len = expected_simplepir_response_byte_len(&params);
        let zeros = vec![0u8; len];
        let (sa, sb) = ([1u8; 32], [2u8; 32]);

        let ra = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (mut c, _) = make_yclient_from_seed(&params, sa);
            let y = YClient::from_seed(&mut c, &params, sa);
            YPIRClient::decode_response_simplepir_yclient(&params, &y, &zeros)
        }));
        let rb = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (mut c, _) = make_yclient_from_seed(&params, sb);
            let y = YClient::from_seed(&mut c, &params, sb);
            YPIRClient::decode_response_simplepir_yclient(&params, &y, &zeros)
        }));
        assert_eq!(ra.is_ok(), rb.is_ok(),
            "All-zeros SimplePIR response must be key-independent");
    }

    #[test]
    fn decode_simplepir_random_response_same_outcome_different_keys() {
        let params = make_simplepir_params();
        let len = expected_simplepir_response_byte_len(&params);
        let resp: Vec<u8> = (0..len).map(|_| fastrand::u8(..)).collect();
        let (sa, sb) = ([10u8; 32], [20u8; 32]);

        let ra = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (mut c, _) = make_yclient_from_seed(&params, sa);
            let y = YClient::from_seed(&mut c, &params, sa);
            YPIRClient::decode_response_simplepir_yclient(&params, &y, &resp)
        }));
        let rb = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (mut c, _) = make_yclient_from_seed(&params, sb);
            let y = YClient::from_seed(&mut c, &params, sb);
            YPIRClient::decode_response_simplepir_yclient(&params, &y, &resp)
        }));
        assert_eq!(ra.is_ok(), rb.is_ok(),
            "Random adversarial SimplePIR response must be key-independent");
    }

    #[test]
    fn decode_simplepir_deterministic_for_same_seed() {
        let params = make_simplepir_params();
        let len = expected_simplepir_response_byte_len(&params);
        let resp: Vec<u8> = (0..len).map(|i| (i % 256) as u8).collect();
        let seed = fixed_seed();

        let r1 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (mut c, _) = make_yclient_from_seed(&params, seed);
            let y = YClient::from_seed(&mut c, &params, seed);
            YPIRClient::decode_response_simplepir_yclient(&params, &y, &resp)
        }));
        let r2 = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (mut c, _) = make_yclient_from_seed(&params, seed);
            let y = YClient::from_seed(&mut c, &params, seed);
            YPIRClient::decode_response_simplepir_yclient(&params, &y, &resp)
        }));
        match (r1, r2) {
            (Ok(v1), Ok(v2)) => assert_eq!(v1, v2),
            (Err(_), Err(_)) => {}
            _ => panic!("Inconsistent panic behavior across identical decode calls"),
        }
    }

    #[test]
    #[should_panic]
    fn ypirclient_decode_normal_empty() {
        let c = YPIRClient::from_db_sz(1u64 << 20, 1, false);
        let _ = c.decode_response_normal(fixed_seed(), &[]);
    }

    #[test]
    #[should_panic]
    fn ypirclient_decode_simplepir_empty() {
        let c = YPIRClient::from_db_sz(1 << 14, 2048 * 14, true);
        let _ = c.decode_response_simplepir(fixed_seed(), &[]);
    }

    /// KNOWN VULNERABILITY: selective-failure side channel via crafted boundary response.
    #[test]
    #[ignore = "documents selective-failure side channel (known vulnerability)"]
    fn decode_normal_crafted_near_boundary_same_outcome() {
        let params = make_test_params();
        let len = expected_normal_response_byte_len(&params);
        let crafted = vec![0x7Fu8; len];
        let seeds: Vec<Seed> = (0..5u8).map(|i| [i + 50; 32]).collect();

        let outcomes: Vec<bool> = seeds.iter().map(|seed| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let (mut c, _) = make_yclient_from_seed(&params, *seed);
                let y = YClient::from_seed(&mut c, &params, *seed);
                YPIRClient::decode_response_normal_yclient(&params, &y, &crafted)
            })).is_ok()
        }).collect();

        let all_same = outcomes.windows(2).all(|w| w[0] == w[1]);
        assert!(all_same,
            "Crafted boundary response produced different outcomes across keys: {:?}", outcomes);
    }
}

// ---------------------------------------------------------------------------
// Area 2: SP response decoding pipeline unit tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod sp_decode_pipeline_tests {
    use super::*;
    use crate::bits::{contiguous_bytes_to_u64s, u64s_to_contiguous_bytes};
    use crate::modulus_switch::ModulusSwitch;
    use crate::params::{params_for_scenario_simplepir, GetQPrime, PtModulusBits};

    fn sp_params() -> Params {
        params_for_scenario_simplepir(1 << 14, 2048 * 14)
    }

    fn fixed_seed() -> Seed {
        [42u8; 32]
    }

    // -- RLWE encrypt-then-decrypt round-trip --

    #[test]
    fn rlwe_encrypt_decrypt_roundtrip_zero_plaintext() {
        let params = sp_params();
        let mut client = Client::init(&params);
        client.generate_secret_keys_from_seed(fixed_seed());

        let pt = PolyMatrixRaw::zero(&params, 1, 1);
        let ct = client.encrypt_matrix_reg(
            &pt.ntt(),
            &mut ChaCha20Rng::from_entropy(),
            &mut ChaCha20Rng::from_entropy(),
        );
        let dec = client.decrypt_matrix_reg(&ct).raw();

        for z in 0..params.poly_len {
            let val = rescale(dec.data[z], params.modulus, params.pt_modulus);
            assert_eq!(val, 0, "coeff {z} should decrypt to 0");
        }
    }

    #[test]
    fn rlwe_encrypt_decrypt_roundtrip_known_plaintext() {
        let params = sp_params();
        let mut client = Client::init(&params);
        client.generate_secret_keys_from_seed(fixed_seed());

        let scale_k = params.modulus / params.pt_modulus;
        let mut pt = PolyMatrixRaw::zero(&params, 1, 1);
        for z in 0..params.poly_len {
            let msg = (z as u64) % params.pt_modulus;
            pt.data[z] = msg * scale_k;
        }

        let ct = client.encrypt_matrix_reg(
            &pt.ntt(),
            &mut ChaCha20Rng::from_entropy(),
            &mut ChaCha20Rng::from_entropy(),
        );
        let dec = client.decrypt_matrix_reg(&ct).raw();

        for z in 0..params.poly_len {
            let expected = (z as u64) % params.pt_modulus;
            let got = rescale(dec.data[z], params.modulus, params.pt_modulus);
            assert_eq!(got, expected, "coeff {z}: expected {expected}, got {got}");
        }
    }

    #[test]
    fn rlwe_encrypt_decrypt_roundtrip_max_plaintext() {
        let params = sp_params();
        let mut client = Client::init(&params);
        client.generate_secret_keys_from_seed(fixed_seed());

        let scale_k = params.modulus / params.pt_modulus;
        let max_pt = params.pt_modulus - 1;
        let mut pt = PolyMatrixRaw::zero(&params, 1, 1);
        for z in 0..params.poly_len {
            pt.data[z] = max_pt * scale_k;
        }

        let ct = client.encrypt_matrix_reg(
            &pt.ntt(),
            &mut ChaCha20Rng::from_entropy(),
            &mut ChaCha20Rng::from_entropy(),
        );
        let dec = client.decrypt_matrix_reg(&ct).raw();

        for z in 0..params.poly_len {
            let got = rescale(dec.data[z], params.modulus, params.pt_modulus);
            assert_eq!(got, max_pt, "coeff {z}: expected {max_pt}, got {got}");
        }
    }

    #[test]
    fn rlwe_encrypt_decrypt_deterministic_with_fixed_rngs() {
        let params = sp_params();
        let seed = fixed_seed();

        let decrypt_with = |rng_seed: [u8; 32], pub_seed: [u8; 32]| -> Vec<u64> {
            let mut client = Client::init(&params);
            client.generate_secret_keys_from_seed(seed);
            let scale_k = params.modulus / params.pt_modulus;
            let mut pt = PolyMatrixRaw::zero(&params, 1, 1);
            pt.data[0] = 7 * scale_k;
            let ct = client.encrypt_matrix_reg(
                &pt.ntt(),
                &mut ChaCha20Rng::from_seed(rng_seed),
                &mut ChaCha20Rng::from_seed(pub_seed),
            );
            let dec = client.decrypt_matrix_reg(&ct).raw();
            (0..params.poly_len).map(|z| rescale(dec.data[z], params.modulus, params.pt_modulus)).collect()
        };

        let r1 = decrypt_with([99u8; 32], [100u8; 32]);
        let r2 = decrypt_with([99u8; 32], [100u8; 32]);
        assert_eq!(r1, r2, "same RNG seeds must produce identical decrypt results");
        assert_eq!(r1[0], 7);
    }

    // -- decrypt_ct_reg_measured --

    #[test]
    fn decrypt_ct_reg_measured_recovers_plaintext() {
        let params = sp_params();
        let mut client = Client::init(&params);
        client.generate_secret_keys_from_seed(fixed_seed());

        let scale_k = params.modulus / params.pt_modulus;
        let mut pt = PolyMatrixRaw::zero(&params, 1, 1);
        for z in 0..params.poly_len {
            pt.data[z] = ((z as u64 * 3) % params.pt_modulus) * scale_k;
        }

        let ct = client.encrypt_matrix_reg(
            &pt.ntt(),
            &mut ChaCha20Rng::from_entropy(),
            &mut ChaCha20Rng::from_entropy(),
        );
        let dec = decrypt_ct_reg_measured(&client, &params, &ct, params.poly_len);

        for z in 0..params.poly_len {
            let expected = (z as u64 * 3) % params.pt_modulus;
            assert_eq!(dec.data[z], expected, "coeff {z} mismatch");
        }
    }

    // -- Modulus switch round-trip with SP parameters --

    #[test]
    fn modulus_switch_roundtrip_sp_params() {
        let params = sp_params();
        let q1 = params.get_q_prime_1();
        let q2 = params.get_q_prime_2();

        let mut client = Client::init(&params);
        client.generate_secret_keys_from_seed(fixed_seed());

        let scale_k = params.modulus / params.pt_modulus;
        let mut pt = PolyMatrixRaw::zero(&params, 1, 1);
        for z in 0..params.poly_len {
            pt.data[z] = ((z as u64 * 5) % params.pt_modulus) * scale_k;
        }
        let ct_ntt = client.encrypt_matrix_reg(
            &pt.ntt(),
            &mut ChaCha20Rng::from_entropy(),
            &mut ChaCha20Rng::from_entropy(),
        );
        let ct_raw = ct_ntt.raw();

        let switched = ct_raw.switch(q1, q2);
        let recovered = PolyMatrixRaw::recover(&params, q1, q2, &switched);

        let dec = client.decrypt_matrix_reg(&recovered.ntt()).raw();
        for z in 0..params.poly_len {
            let expected = (z as u64 * 5) % params.pt_modulus;
            let got = rescale(dec.data[z], params.modulus, params.pt_modulus);
            assert_eq!(got, expected, "switch/recover round-trip failed at coeff {z}");
        }
    }

    // -- Full SP decode pipeline (synthetic response, no server needed) --

    #[test]
    fn sp_decode_synthetic_single_rlwe() {
        let params = sp_params();
        let seed = fixed_seed();
        let q1 = params.get_q_prime_1();
        let q2 = params.get_q_prime_2();

        let mut client = Client::init(&params);
        client.generate_secret_keys_from_seed(seed);

        let scale_k = params.modulus / params.pt_modulus;
        let expected_values: Vec<u64> = (0..params.poly_len)
            .map(|z| (z as u64 * 13) % params.pt_modulus)
            .collect();

        let mut pt = PolyMatrixRaw::zero(&params, 1, 1);
        for z in 0..params.poly_len {
            pt.data[z] = expected_values[z] * scale_k;
        }
        let ct = client.encrypt_matrix_reg(
            &pt.ntt(),
            &mut ChaCha20Rng::from_entropy(),
            &mut ChaCha20Rng::from_entropy(),
        );
        let ct_raw = ct.raw();
        let switched_bytes = ct_raw.switch(q1, q2);

        let y_client = YClient::from_seed(&mut client, &params, seed);
        let recovered = PolyMatrixRaw::recover(&params, q1, q2, &switched_bytes);
        let dec = decrypt_ct_reg_measured(y_client.client(), &params, &recovered.ntt(), params.poly_len);

        for z in 0..params.poly_len {
            assert_eq!(dec.data[z], expected_values[z],
                "SP synthetic decode failed at coeff {z}: expected {}, got {}",
                expected_values[z], dec.data[z]);
        }
    }

    #[test]
    fn sp_decode_full_pipeline_with_byte_conversion() {
        let params = sp_params();
        let seed = fixed_seed();
        let q1 = params.get_q_prime_1();
        let q2 = params.get_q_prime_2();
        let pt_bits = params.pt_modulus_bits();

        let mut client = Client::init(&params);
        client.generate_secret_keys_from_seed(seed);

        let scale_k = params.modulus / params.pt_modulus;
        let num_rlwe_outputs = params.instances;
        let mut all_expected: Vec<u64> = Vec::new();
        let mut response_bytes: Vec<u8> = Vec::new();

        for inst in 0..num_rlwe_outputs {
            let pt_vals: Vec<u64> = (0..params.poly_len)
                .map(|z| ((inst * params.poly_len + z) as u64 * 7) % params.pt_modulus)
                .collect();
            all_expected.extend_from_slice(&pt_vals);

            let mut pt = PolyMatrixRaw::zero(&params, 1, 1);
            for z in 0..params.poly_len {
                pt.data[z] = pt_vals[z] * scale_k;
            }
            let ct = client.encrypt_matrix_reg(
                &pt.ntt(),
                &mut ChaCha20Rng::from_entropy(),
                &mut ChaCha20Rng::from_entropy(),
            );
            response_bytes.extend_from_slice(&ct.raw().switch(q1, q2));
        }

        let y_client = YClient::from_seed(&mut client, &params, seed);
        let decoded = YPIRClient::decode_response_simplepir_yclient(&params, &y_client, &response_bytes);
        assert_eq!(decoded.len(), num_rlwe_outputs * params.poly_len);

        for (i, (got, expected)) in decoded.iter().zip(all_expected.iter()).enumerate() {
            assert_eq!(*got, *expected,
                "Full SP pipeline mismatch at index {i}: expected {expected}, got {got}");
        }

        let decoded_bytes = u64s_to_contiguous_bytes(&decoded, pt_bits);
        let round_tripped = contiguous_bytes_to_u64s(&decoded_bytes, pt_bits);
        for (i, (got, expected)) in round_tripped.iter().zip(all_expected.iter()).enumerate() {
            assert_eq!(*got, *expected,
                "Byte conversion round-trip mismatch at index {i}");
        }
    }

    #[test]
    fn sp_decode_output_length_invariant() {
        let params = sp_params();
        let seed = fixed_seed();
        let q1 = params.get_q_prime_1();
        let q2 = params.get_q_prime_2();

        let mut client = Client::init(&params);
        client.generate_secret_keys_from_seed(seed);

        let num_rlwe_outputs = params.instances;
        let mut response_bytes: Vec<u8> = Vec::new();
        for _ in 0..num_rlwe_outputs {
            let pt = PolyMatrixRaw::zero(&params, 1, 1);
            let ct = client.encrypt_matrix_reg(
                &pt.ntt(),
                &mut ChaCha20Rng::from_entropy(),
                &mut ChaCha20Rng::from_entropy(),
            );
            response_bytes.extend_from_slice(&ct.raw().switch(q1, q2));
        }

        let y_client = YClient::from_seed(&mut client, &params, seed);
        let decoded = YPIRClient::decode_response_simplepir_yclient(&params, &y_client, &response_bytes);
        assert_eq!(decoded.len(), params.instances * params.poly_len,
            "SP decode output must be instances * poly_len");
    }

    // -- 14-bit word packing round-trip --

    #[test]
    fn u64s_14bit_packing_roundtrip() {
        let pt_bits = 14;
        let n = 2048;
        let vals: Vec<u64> = (0..n).map(|i| (i as u64 * 997) % (1 << pt_bits)).collect();
        let bytes = u64s_to_contiguous_bytes(&vals, pt_bits);
        let recovered = contiguous_bytes_to_u64s(&bytes, pt_bits);
        assert_eq!(recovered.len(), vals.len());
        for (i, (got, expected)) in recovered.iter().zip(vals.iter()).enumerate() {
            assert_eq!(*got, *expected, "14-bit round-trip failed at index {i}");
        }
    }

    #[test]
    fn u64s_14bit_packing_edge_values() {
        let pt_bits = 14;
        let max_val = (1u64 << pt_bits) - 1;
        let vals = vec![0, 1, max_val, max_val - 1, (1 << 13), (1 << 13) - 1];
        let bytes = u64s_to_contiguous_bytes(&vals, pt_bits);
        let recovered = contiguous_bytes_to_u64s(&bytes, pt_bits);
        for (i, (got, expected)) in recovered.iter().zip(vals.iter()).enumerate() {
            assert_eq!(*got, *expected, "14-bit edge value failed at index {i}");
        }
    }

    #[test]
    fn u64s_14bit_packing_random_values() {
        let pt_bits = 14;
        let max_val = 1u64 << pt_bits;
        for _ in 0..10 {
            let n = fastrand::usize(1..4096);
            let vals: Vec<u64> = (0..n).map(|_| fastrand::u64(..) % max_val).collect();
            let bytes = u64s_to_contiguous_bytes(&vals, pt_bits);
            let recovered = contiguous_bytes_to_u64s(&bytes, pt_bits);
            assert_eq!(recovered.len(), vals.len());
            assert_eq!(recovered, vals);
        }
    }

    // -- pack_query CRT encoding --

    #[test]
    fn pack_query_crt_encoding() {
        let params = sp_params();
        let vals: Vec<u64> = (0..16).map(|i| (i as u64 * 12345) % params.modulus).collect();
        let packed = pack_query(&params, &vals);
        assert_eq!(packed.len(), vals.len());

        for (i, (&original, &packed_val)) in vals.iter().zip(packed.as_slice().iter()).enumerate() {
            let crt0 = packed_val & 0xFFFFFFFF;
            let crt1 = packed_val >> 32;
            assert_eq!(crt0, original % params.moduli[0],
                "CRT component 0 mismatch at index {i}");
            assert_eq!(crt1, original % params.moduli[1],
                "CRT component 1 mismatch at index {i}");
        }
    }

    #[test]
    fn pack_query_zero_input() {
        let params = sp_params();
        let zeros = vec![0u64; 32];
        let packed = pack_query(&params, &zeros);
        assert_eq!(packed.len(), 32);
        for &val in packed.as_slice() {
            assert_eq!(val, 0);
        }
    }

    // -- rlwe_to_lwe extraction --

    #[test]
    fn rlwe_to_lwe_output_length() {
        let params = sp_params();
        let ct = PolyMatrixRaw::zero(&params, 2, 1);
        let lwe = rlwe_to_lwe(&params, &ct);
        let expected_len = params.poly_len * params.poly_len + params.poly_len;
        assert_eq!(lwe.len(), expected_len,
            "rlwe_to_lwe output should be poly_len^2 + poly_len");
    }

    #[test]
    fn rlwe_to_lwe_zero_ciphertext() {
        let params = sp_params();
        let ct = PolyMatrixRaw::zero(&params, 2, 1);
        let lwe = rlwe_to_lwe(&params, &ct);
        for &val in &lwe {
            assert_eq!(val, 0, "rlwe_to_lwe of zero ct should be all zeros");
        }
    }

    // -- Multi-trial statistical correctness for SP decode --

    #[test]
    fn sp_decode_multi_trial_correctness() {
        let params = sp_params();
        let q1 = params.get_q_prime_1();
        let q2 = params.get_q_prime_2();
        let scale_k = params.modulus / params.pt_modulus;

        for trial in 0..5u8 {
            let seed = [trial + 1; 32];
            let mut client = Client::init(&params);
            client.generate_secret_keys_from_seed(seed);

            let expected_val = (trial as u64 * 37) % params.pt_modulus;
            let mut response_bytes = Vec::new();
            for _ in 0..params.instances {
                let mut pt = PolyMatrixRaw::zero(&params, 1, 1);
                pt.data[0] = expected_val * scale_k;
                let ct = client.encrypt_matrix_reg(
                    &pt.ntt(),
                    &mut ChaCha20Rng::from_entropy(),
                    &mut ChaCha20Rng::from_entropy(),
                );
                response_bytes.extend_from_slice(&ct.raw().switch(q1, q2));
            }

            let y_client = YClient::from_seed(&mut client, &params, seed);
            let decoded = YPIRClient::decode_response_simplepir_yclient(&params, &y_client, &response_bytes);

            assert_eq!(decoded[0], expected_val,
                "trial {trial}: first coeff should be {expected_val}, got {}", decoded[0]);
        }
    }
}

// ---------------------------------------------------------------------------
// Area 3: Batch-path tests (shared `pp` + shared `s` per batch)
// ---------------------------------------------------------------------------
//
// * The new batch path is purely additive — the single-query path is
//   not exercised here, but the `build_pack_pub_params` hoist is byte-faithful
//   given the same secret RNG.
// * One `pack_pub_params` is returned per batch and K independent `q.0`
//   vectors share it (and one `client_seed`).
// * Each per-row `q.0` draws fresh `OsRng` entropy so the LWE error vectors
//   `e_k` remain independent under the shared `s`.
// * Per-chunk decoding is independent: decoding three chunks together gives
//   the same per-chunk plaintexts as decoding each chunk alone.
#[cfg(test)]
mod batch_path_tests {
    use super::*;
    use crate::bits::contiguous_bytes_to_u64s;
    use crate::modulus_switch::ModulusSwitch;
    use crate::params::{params_for_scenario_simplepir, GetQPrime, PtModulusBits};

    fn sp_params() -> Params {
        params_for_scenario_simplepir(1 << 14, 2048 * 14)
    }

    fn fixed_seed() -> Seed {
        [42u8; 32]
    }

    /// Synthesise a SimplePIR server response that decodes (under
    /// `client_seed`) to the given `expected_vals` (length must equal
    /// `params.instances * params.poly_len`). Used in place of running an
    /// actual server roundtrip — same approach as
    /// `sp_decode_full_pipeline_with_byte_conversion`.
    fn synth_simplepir_response(params: &Params, client_seed: Seed, expected_vals: &[u64]) -> Vec<u8> {
        let q1 = params.get_q_prime_1();
        let q2 = params.get_q_prime_2();
        let scale_k = params.modulus / params.pt_modulus;
        let num_rlwe_outputs = params.instances;
        assert_eq!(expected_vals.len(), num_rlwe_outputs * params.poly_len,
            "synth_simplepir_response expects instances * poly_len plaintexts");

        let mut client = Client::init(params);
        client.generate_secret_keys_from_seed(client_seed);

        let mut response_bytes = Vec::new();
        for inst in 0..num_rlwe_outputs {
            let mut pt = PolyMatrixRaw::zero(params, 1, 1);
            for z in 0..params.poly_len {
                pt.data[z] = expected_vals[inst * params.poly_len + z] * scale_k;
            }
            let ct = client.encrypt_matrix_reg(
                &pt.ntt(),
                &mut ChaCha20Rng::from_entropy(),
                &mut ChaCha20Rng::from_entropy(),
            );
            response_bytes.extend_from_slice(&ct.raw().switch(q1, q2));
        }
        response_bytes
    }

    #[test]
    fn batch_query_returns_k_queries_and_one_pp() {
        let params = sp_params();
        let ypir = YPIRClient::new(&params);
        let target_rows = vec![0usize, 1, 2, 7, 13];
        let ((q_vec, pp), _seed) = ypir.generate_query_simplepir_batch(&target_rows);

        assert_eq!(q_vec.len(), target_rows.len(),
            "batch should return K query vectors");

        let expected_pp_words = params.poly_len_log2 * params.t_exp_left * params.poly_len;
        assert_eq!(pp.as_slice().len(), expected_pp_words,
            "pp must satisfy the build_pack_pub_params length invariant");

        for q in &q_vec {
            assert_eq!(q.as_slice().len(), params.db_rows(),
                "each per-row q.0 should have db_rows entries");
        }
    }

    #[test]
    fn batch_query_pp_matches_under_fixed_secret_rng() {
        // The §0.5.2 hoist of `build_pack_pub_params` is byte-faithful: given
        // the same client_seed and the same secret-RNG seed, two calls
        // produce identical pp bytes. This guards against accidental
        // behavior change in the legacy single-query path.
        let params = sp_params();
        let seed = fixed_seed();
        let secret_rng_seed = [99u8; 32];

        let pp1 = {
            let mut client = Client::init(&params);
            client.generate_secret_keys_from_seed(seed);
            let y_client = YClient::from_seed(&mut client, &params, seed);
            let mut rng = ChaCha20Rng::from_seed(secret_rng_seed);
            y_client.build_pack_pub_params_with_secret_rng(&mut rng)
        };
        let pp2 = {
            let mut client = Client::init(&params);
            client.generate_secret_keys_from_seed(seed);
            let y_client = YClient::from_seed(&mut client, &params, seed);
            let mut rng = ChaCha20Rng::from_seed(secret_rng_seed);
            y_client.build_pack_pub_params_with_secret_rng(&mut rng)
        };
        assert_eq!(pp1.as_slice(), pp2.as_slice(),
            "build_pack_pub_params_with_secret_rng must be deterministic given (client_seed, secret_rng_seed)");

        // Sanity: a different secret RNG seed produces a different pp blob.
        let pp3 = {
            let mut client = Client::init(&params);
            client.generate_secret_keys_from_seed(seed);
            let y_client = YClient::from_seed(&mut client, &params, seed);
            let mut rng = ChaCha20Rng::from_seed([100u8; 32]);
            y_client.build_pack_pub_params_with_secret_rng(&mut rng)
        };
        assert_ne!(pp1.as_slice(), pp3.as_slice(),
            "different secret RNG seeds should produce different pp blobs");
    }

    #[test]
    fn batch_query_each_q_decodes_independently() {
        let params = sp_params();
        let ypir = YPIRClient::new(&params);
        let pt_modulus = params.pt_modulus;
        let poly_len = params.poly_len;
        let num_rlwe_outputs = params.instances;
        let pt_bits = params.pt_modulus_bits();

        // K different expected plaintext patterns — one per query in the batch.
        const K: usize = 4;
        let expected_per_query: Vec<Vec<u64>> = (0..K).map(|k| {
            (0..(num_rlwe_outputs * poly_len))
                .map(|i| ((k as u64 + 1) * 17 * (i as u64 + 1)) % pt_modulus)
                .collect()
        }).collect();

        // We only need the client_seed from the batch query — pp and q vectors
        // are not used because we synthesise responses out-of-band, exactly
        // like sp_decode_pipeline_tests does.
        let target_rows: Vec<usize> = (0..K).map(|k| k * 3).collect();
        let (_query, client_seed) = ypir.generate_query_simplepir_batch(&target_rows);

        let response_bytes_per_query: Vec<Vec<u8>> = expected_per_query.iter()
            .map(|expected| synth_simplepir_response(&params, client_seed, expected))
            .collect();
        let response_refs: Vec<&[u8]> = response_bytes_per_query.iter()
            .map(|v| v.as_slice())
            .collect();

        let raws = ypir.decode_response_simplepir_batch_raw(client_seed, &response_refs);
        assert_eq!(raws.len(), K);
        for (k, (got, expected)) in raws.iter().zip(expected_per_query.iter()).enumerate() {
            let got = got
                .as_ref()
                .unwrap_or_else(|err| panic!("batch decode slot {k} failed: {err}"));
            assert_eq!(got, expected, "batch decode mismatch on query {k}");
        }

        let bytes = ypir.decode_response_simplepir_batch(client_seed, &response_refs);
        assert_eq!(bytes.len(), K);
        for (k, (got_bytes, expected)) in bytes.iter().zip(expected_per_query.iter()).enumerate() {
            let got_bytes = got_bytes
                .as_ref()
                .unwrap_or_else(|err| panic!("batch byte decode slot {k} failed: {err}"));
            let recovered = contiguous_bytes_to_u64s(got_bytes, pt_bits);
            assert_eq!(&recovered[..expected.len()], expected.as_slice(),
                "batch decode (byte variant) mismatch on query {k}");
        }
    }

    #[test]
    fn batch_query_distinct_secret_rngs_produce_distinct_q() {
        // Same target row queried K times in one batch must yield K different
        // `q.0` byte streams because each per-row call to
        // `generate_query_lwe_low_mem` draws fresh OsRng entropy. This pins
        // the noise-freshness invariant the shared-`s` security argument
        // relies on (§0.5.4 / earlier discussion).
        let params = sp_params();
        let ypir = YPIRClient::new(&params);
        let target_rows = vec![5usize; 3];
        let ((q_vec, _pp), _seed) = ypir.generate_query_simplepir_batch(&target_rows);
        assert_ne!(q_vec[0].as_slice(), q_vec[1].as_slice(),
            "two queries for the same row must differ (independent e_k)");
        assert_ne!(q_vec[0].as_slice(), q_vec[2].as_slice());
        assert_ne!(q_vec[1].as_slice(), q_vec[2].as_slice());
    }

    #[test]
    fn batch_decode_chunk_independence() {
        // Decoding three chunks together must produce the same per-chunk
        // plaintexts as decoding each chunk alone. This pins the "no
        // coupling" property the §0.5.5 batch error-oracle argument depends
        // on: under shared `s`, a malformed/different chunk in one slot
        // must not change decode output of any other slot.
        let params = sp_params();
        let ypir = YPIRClient::new(&params);
        let pt_modulus = params.pt_modulus;
        let poly_len = params.poly_len;
        let num_rlwe_outputs = params.instances;
        let n = num_rlwe_outputs * poly_len;

        let expected_0: Vec<u64> = (0..n).map(|i| (i as u64) % pt_modulus).collect();
        let expected_1: Vec<u64> = (0..n).map(|i| ((i as u64 * 5) + 1) % pt_modulus).collect();
        let expected_2: Vec<u64> = (0..n).map(|i| ((i as u64 * 7) + 3) % pt_modulus).collect();

        let target_rows = vec![0usize, 1, 2];
        let (_query, client_seed) = ypir.generate_query_simplepir_batch(&target_rows);
        let r0 = synth_simplepir_response(&params, client_seed, &expected_0);
        let r1 = synth_simplepir_response(&params, client_seed, &expected_1);
        let r2 = synth_simplepir_response(&params, client_seed, &expected_2);

        let alone_0 = ypir.decode_response_simplepir_raw(client_seed, &r0);
        let alone_1 = ypir.decode_response_simplepir_raw(client_seed, &r1);
        let alone_2 = ypir.decode_response_simplepir_raw(client_seed, &r2);

        let response_refs: Vec<&[u8]> = vec![r0.as_slice(), r1.as_slice(), r2.as_slice()];
        let batch = ypir.decode_response_simplepir_batch_raw(client_seed, &response_refs);

        assert_eq!(batch.len(), 3);
        let batch_0 = batch[0].as_ref().expect("batch slot 0 should decode");
        let batch_1 = batch[1].as_ref().expect("batch slot 1 should decode");
        let batch_2 = batch[2].as_ref().expect("batch slot 2 should decode");
        assert_eq!(batch_0, &alone_0, "batch slot 0 must match standalone decode");
        assert_eq!(batch_1, &alone_1, "batch slot 1 must match standalone decode");
        assert_eq!(batch_2, &alone_2, "batch slot 2 must match standalone decode");

        assert_eq!(batch_0, &expected_0);
        assert_eq!(batch_1, &expected_1);
        assert_eq!(batch_2, &expected_2);
    }

    #[test]
    fn batch_decode_returns_per_slot_error_for_malformed_chunk() {
        let params = sp_params();
        let ypir = YPIRClient::new(&params);
        let pt_modulus = params.pt_modulus;
        let n = params.instances * params.poly_len;

        let expected_0: Vec<u64> = (0..n).map(|i| ((i as u64 * 3) + 2) % pt_modulus).collect();
        let expected_2: Vec<u64> = (0..n).map(|i| ((i as u64 * 11) + 7) % pt_modulus).collect();

        let target_rows = vec![0usize, 1, 2];
        let (_query, client_seed) = ypir.generate_query_simplepir_batch(&target_rows);
        let r0 = synth_simplepir_response(&params, client_seed, &expected_0);
        let malformed = vec![0xFFu8];
        let r2 = synth_simplepir_response(&params, client_seed, &expected_2);

        let response_refs: Vec<&[u8]> = vec![r0.as_slice(), malformed.as_slice(), r2.as_slice()];
        let batch = ypir.decode_response_simplepir_batch_raw(client_seed, &response_refs);

        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0].as_ref().expect("slot 0 should decode"), &expected_0);
        assert!(batch[1].is_err(), "malformed slot should return an error");
        assert_eq!(batch[2].as_ref().expect("slot 2 should decode"), &expected_2);
    }
}
