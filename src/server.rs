#[cfg(target_feature = "avx2")]
use std::arch::x86_64::*;
use std::{marker::PhantomData, ops::Range, time::Instant};

use log::debug;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use spiral_rs::aligned_memory::AlignedMemory64;
use spiral_rs::{arith::*, client::*, params::*, poly::*};

use crate::measurement::Measurement;
use crate::serialize::*;

use super::{
    bits::*,
    client::*,
    constants::*,
    convolution::{negacyclic_perm_u32, Convolution},
    kernel::*,
    lwe::*,
    matmul::matmul_vec_packed,
    modulus_switch::ModulusSwitch,
    packing::*,
    params::*,
    transpose::*,
    util::*,
};

pub fn generate_y_constants<'a>(
    params: &'a Params,
) -> (Vec<PolyMatrixNTT<'a>>, Vec<PolyMatrixNTT<'a>>) {
    let mut y_constants = Vec::new();
    let mut neg_y_constants = Vec::new();
    for num_cts_log2 in 1..params.poly_len_log2 + 1 {
        let num_cts = 1 << num_cts_log2;

        // Y = X^(poly_len / num_cts)
        let mut y_raw = PolyMatrixRaw::zero(params, 1, 1);
        y_raw.data[params.poly_len / num_cts] = 1;
        let y = y_raw.ntt();

        let mut neg_y_raw = PolyMatrixRaw::zero(params, 1, 1);
        neg_y_raw.data[params.poly_len / num_cts] = params.modulus - 1;
        let neg_y = neg_y_raw.ntt();

        y_constants.push(y);
        neg_y_constants.push(neg_y);
    }

    (y_constants, neg_y_constants)
}

/// Takes a matrix of u64s and returns a matrix of T's.
///
/// Input is row x cols u64's.
/// Output is out_rows x cols T's.
pub fn split_alloc(
    buf: &[u64],
    special_bit_offs: usize,
    rows: usize,
    cols: usize,
    out_rows: usize,
    inp_mod_bits: usize,
    pt_bits: usize,
) -> Vec<u16> {
    let mut out = vec![0u16; out_rows * cols];

    assert!(out_rows >= rows);
    assert!(inp_mod_bits >= pt_bits);

    for j in 0..cols {
        let mut bytes_tmp = vec![0u8; out_rows * inp_mod_bits / 8];

        // read this column
        let mut bit_offs = 0;
        for i in 0..rows {
            let inp = buf[i * cols + j];
            // if j < 10 {
            //     debug!("({},{}) inp: {}", i, j, inp);
            // }

            if i == rows - 1 {
                bit_offs = special_bit_offs;
            }

            // if j == 4095 {
            //     debug!("write: {}/{} {}/{}", j, cols, i, rows);
            // }
            write_bits(&mut bytes_tmp, inp, bit_offs, inp_mod_bits);
            bit_offs += inp_mod_bits;
        }

        // debug!("stretch: {}", j);

        // now, 'stretch' the column vertically
        let mut bit_offs = 0;
        for i in 0..out_rows {
            // if j == 4095 {
            //     debug!("stretch: {}/{}", i, out_rows);
            //     debug!("reading at offs: {}, {} bits", bit_offs, pt_bits);
            //     debug!("into byte buffer of len: {}", bytes_tmp.len());
            //     debug!("writing at {} in out of len {}", i * cols + j, out.len());
            // }
            let out_val = read_bits(&bytes_tmp, bit_offs, pt_bits);
            out[i * cols + j] = out_val as u16;
            // if j == 4095 {
            //     debug!("wrote at {} in out of len {}", i * cols + j, out.len());
            // }
            bit_offs += pt_bits;
            if bit_offs >= out_rows * inp_mod_bits {
                break;
            }
        }

        // debug!("here {}", j);
        // debug!(
        //     "out {}",
        //     out[(special_bit_offs / pt_bits) * cols + j].to_u64()
        // );
        // debug!("buf {}", buf[(rows - 1) * cols + j] & ((1 << pt_bits) - 1));

        assert_eq!(
            out[(special_bit_offs / pt_bits) * cols + j] as u64,
            buf[(rows - 1) * cols + j] & ((1 << pt_bits) - 1)
        );
    }

    out
}

pub fn generate_fake_pack_pub_params<'a>(params: &'a Params) -> Vec<PolyMatrixNTT<'a>> {
    let pack_pub_params = raw_generate_expansion_params(
        &params,
        &PolyMatrixRaw::zero(&params, 1, 1),
        params.poly_len_log2,
        params.t_exp_left,
        &mut ChaCha20Rng::from_entropy(),
        &mut ChaCha20Rng::from_seed(STATIC_SEED_2),
    );
    pack_pub_params
}

#[derive(Clone)]
pub struct YServer<'a, T> {
    params: &'a Params,
    db_buf_aligned: AlignedMemory64, // db_buf: Vec<u8>, // stored transposed
    phantom: PhantomData<T>,
    pad_rows: bool,
}

impl<'a, T> YServer<'a, T>
where
    T: Sized + Copy + ToU64 + Default,
    *const T: ToM512 + ToU64,
{
    pub fn new<'b, I>(params: &'a Params, mut db: I, inp_transposed: bool, pad_rows: bool) -> Self
    where
        I: Iterator<Item = T>,
    {
        let bytes_per_pt_el = std::mem::size_of::<T>(); //1; //((lwe_params.pt_modulus as f64).log2() / 8.).ceil() as usize;

        let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
        let db_rows_padded = if pad_rows {
            params.db_rows_padded_simplepir()
        } else {
            db_rows
        };
        let db_cols = params.instances * params.poly_len;

        let sz_bytes = db_rows_padded * db_cols * bytes_per_pt_el;

        let mut db_buf_aligned = AlignedMemory64::new(sz_bytes / 8);
        let db_buf_mut = as_bytes_mut(&mut db_buf_aligned);
        let db_buf_ptr = db_buf_mut.as_mut_ptr() as *mut T;

        for i in 0..db_rows {
            for j in 0..db_cols {
                let idx = if inp_transposed {
                    i * db_cols + j
                } else {
                    j * db_rows_padded + i
                };

                unsafe {
                    *db_buf_ptr.add(idx) = db.next().unwrap();
                    // *db_buf_ptr.add(idx) = if i < db_rows {
                    //     db.next().unwrap()
                    // } else {
                    //     T::default()
                    // };
                }
            }
        }

        Self {
            params,
            db_buf_aligned,
            phantom: PhantomData,
            pad_rows,
        }
    }

    pub fn db_rows(&self) -> usize {
        1 << (self.params.db_dim_1 + self.params.poly_len_log2)
    }

    pub fn db_rows_padded(&self) -> usize {
        if self.pad_rows {
            self.params.db_rows_padded_simplepir()
        } else {
            1 << (self.params.db_dim_1 + self.params.poly_len_log2)
        }
    }

    pub fn db_cols(&self) -> usize {
        self.params.instances * self.params.poly_len
    }

    pub fn multiply_batched_with_db_packed<const K: usize>(
        &self,
        aligned_query_packed: &[u64],
        query_rows: usize,
    ) -> AlignedMemory64 {
        // let db_rows = 1 << (self.params.db_dim_1 + self.params.poly_len_log2);
        let db_rows_padded = self.db_rows_padded();
        let db_cols = self.db_cols();
        assert_eq!(aligned_query_packed.len(), K * query_rows * db_rows_padded);
        assert_eq!(query_rows, 1);

        let now = Instant::now();
        let mut result = AlignedMemory64::new(K * db_cols);
        fast_batched_dot_product::<K, _>(
            self.params,
            result.as_mut_slice(),
            aligned_query_packed,
            db_rows_padded,
            &self.db(),
            db_rows_padded,
            db_cols,
        );
        debug!("Fast dot product in {} us", now.elapsed().as_micros());

        result
    }

    pub fn lwe_multiply_batched_with_db_packed<const K: usize>(
        &self,
        aligned_query_packed: &[u32],
    ) -> Vec<u32> {
        let _db_rows = 1 << (self.params.db_dim_1 + self.params.poly_len_log2);
        let db_cols = self.db_cols();
        let db_rows_padded = self.db_rows_padded();
        assert_eq!(aligned_query_packed.len(), K * db_rows_padded);
        // assert_eq!(aligned_query_packed[db_rows + 1], 0);

        let mut result = vec![0u32; (db_cols + 8) * K];
        let now = Instant::now();
        // let mut result = AlignedMemory64::new(K * db_cols + 8);
        // lwe_fast_batched_dot_product::<K, _>(
        //     self.params,
        //     result.as_mut_slice(),
        //     aligned_query_packed,
        //     db_rows,
        //     &self.db(),
        //     db_rows,
        //     db_cols,
        // );
        let a_rows = db_cols;
        let a_true_cols = db_rows_padded;
        let a_cols = a_true_cols / 4; // order is inverted on purpose, because db is transposed
        let b_rows = a_true_cols;
        let b_cols = K;
        matmul_vec_packed(
            result.as_mut_slice(),
            self.db_u32(),
            aligned_query_packed,
            a_rows,
            a_cols,
            b_rows,
            b_cols,
        );
        let t = Instant::now();
        let result = transpose_generic(&result, db_cols, K);
        debug!("Transpose in {} us", t.elapsed().as_micros());
        debug!("Fast dot product in {} us", now.elapsed().as_micros());

        result
    }

    pub fn multiply_with_db_ring(
        &self,
        preprocessed_query: &[PolyMatrixNTT],
        col_range: Range<usize>,
        _seed_idx: u8,
    ) -> Vec<u64> {
        let db_rows_poly = 1 << (self.params.db_dim_1);
        let db_rows = 1 << (self.params.db_dim_1 + self.params.poly_len_log2);
        assert_eq!(preprocessed_query.len(), db_rows_poly);

        // assert_eq!(db_rows_poly, 1); // temporary restriction

        // let mut preprocessed_query = Vec::new();
        // for query_el in query {
        //     let query_raw = query_el.raw();
        //     let query_raw_transformed =
        //         negacyclic_perm(query_raw.get_poly(0, 0), 0, self.params.modulus);
        //     let mut query_transformed_pol = PolyMatrixRaw::zero(self.params, 1, 1);
        //     query_transformed_pol
        //         .as_mut_slice()
        //         .copy_from_slice(&query_raw_transformed);
        //     preprocessed_query.push(query_transformed_pol.ntt());
        // }

        let mut result = Vec::new();
        let db = self.db();

        let mut prod = PolyMatrixNTT::zero(self.params, 1, 1);
        let mut db_elem_poly = PolyMatrixRaw::zero(self.params, 1, 1);
        let mut db_elem_ntt = PolyMatrixNTT::zero(self.params, 1, 1);

        for col in col_range.clone() {
            let mut sum = PolyMatrixNTT::zero(self.params, 1, 1);

            for row in 0..db_rows_poly {
                for z in 0..self.params.poly_len {
                    db_elem_poly.data[z] =
                        db[col * db_rows + row * self.params.poly_len + z].to_u64();
                }
                to_ntt(&mut db_elem_ntt, &db_elem_poly);

                multiply(&mut prod, &preprocessed_query[row], &db_elem_ntt);

                if row == db_rows_poly - 1 {
                    add_into(&mut sum, &prod);
                } else {
                    add_into_no_reduce(&mut sum, &prod);
                }
            }

            let sum_raw = sum.raw();
            result.extend(sum_raw.as_slice());
        }

        // result
        let now = Instant::now();
        let res = transpose_generic(&result, col_range.len(), self.params.poly_len);
        debug!("transpose in {} us", now.elapsed().as_micros());
        res
    }

    pub fn generate_pseudorandom_query(&self, public_seed_idx: u8) -> Vec<PolyMatrixNTT<'a>> {
        let mut client = Client::init(&self.params);
        client.generate_secret_keys();
        let y_client = YClient::new(&mut client, &self.params);
        let query = y_client.generate_query_impl(public_seed_idx, self.params.db_dim_1, true, 0);
        let query_mapped = query
            .iter()
            .map(|x| x.submatrix(0, 0, 1, 1))
            .collect::<Vec<_>>();

        let mut preprocessed_query = Vec::new();
        for query_raw in query_mapped {
            // let query_raw_transformed =
            //     negacyclic_perm(query_raw.get_poly(0, 0), 0, self.params.modulus);
            // let query_raw_transformed = query_raw.get_poly(0, 0);
            let query_raw_transformed = if public_seed_idx == SEED_0 {
                negacyclic_perm(query_raw.get_poly(0, 0), 0, self.params.modulus)
                // query_raw.get_poly(0, 0).to_owned()
            } else {
                negacyclic_perm(query_raw.get_poly(0, 0), 0, self.params.modulus)
            };
            let mut query_transformed_pol = PolyMatrixRaw::zero(self.params, 1, 1);
            query_transformed_pol
                .as_mut_slice()
                .copy_from_slice(&query_raw_transformed);
            preprocessed_query.push(query_transformed_pol.ntt());
        }

        preprocessed_query
    }

    pub fn answer_hint_ring(&self, public_seed_idx: u8, cols: usize) -> Vec<u64> {
        let preprocessed_query = self.generate_pseudorandom_query(public_seed_idx);

        let res = self.multiply_with_db_ring(&preprocessed_query, 0..cols, public_seed_idx);

        res
    }

    pub fn generate_hint_0(&self) -> Vec<u64> {
        let _db_rows = 1 << (self.params.db_dim_1 + self.params.poly_len_log2);
        let db_cols = self.db_cols();

        let mut rng_pub = ChaCha20Rng::from_seed(get_seed(SEED_0));
        let lwe_params = LWEParams::default();

        // pseudorandom LWE query is n x db_rows
        let psuedorandom_query =
            generate_matrix_ring(&mut rng_pub, lwe_params.n, lwe_params.n, db_cols);

        // db is db_cols x db_rows (!!!)
        // hint_0 is n x db_cols
        let hint_0 = naive_multiply_matrices(
            &psuedorandom_query,
            lwe_params.n,
            db_cols,
            &self.db(),
            self.db_rows_padded(), // TODO: doesn't quite work
            db_cols,
            true,
        );
        hint_0.iter().map(|&x| x as u64).collect::<Vec<_>>()
    }

    pub fn generate_hint_0_ring(&self) -> Vec<u64> {
        let db_rows = 1 << (self.params.db_dim_1 + self.params.poly_len_log2);
        let db_cols = self.db_cols();

        let lwe_params = LWEParams::default();
        let n = lwe_params.n;
        let conv = Convolution::new(n);

        let mut hint_0 = vec![0u64; n * db_cols];

        let convd_len = conv.params().crt_count * conv.params().poly_len;

        let mut rng_pub = ChaCha20Rng::from_seed(get_seed(SEED_0));

        let mut v_nega_perm_a = Vec::new();
        for _ in 0..db_rows / n {
            let mut a = vec![0u32; n];
            for idx in 0..n {
                a[idx] = rng_pub.sample::<u32, _>(rand::distributions::Standard);
            }
            let nega_perm_a = negacyclic_perm_u32(&a);
            let nega_perm_a_ntt = conv.ntt(&nega_perm_a);
            v_nega_perm_a.push(nega_perm_a_ntt);
        }

        // limit on the number of times we can add results modulo M before we wrap
        let log2_conv_output =
            log2(lwe_params.modulus) + log2(lwe_params.n as u64) + log2(lwe_params.pt_modulus);
        let log2_modulus = log2(conv.params().modulus);
        let log2_max_adds = log2_modulus - log2_conv_output - 1;
        assert!(log2_max_adds > 0);
        let max_adds = 1 << log2_max_adds;

        for col in 0..db_cols {
            let mut tmp_col = vec![0u64; convd_len];
            for outer_row in 0..db_rows / n {
                let start_idx = col * self.db_rows_padded() + outer_row * n;
                let pt_col = &self.db()[start_idx..start_idx + n];
                let pt_col_u32 = pt_col
                    .iter()
                    .map(|&x| x.to_u64() as u32)
                    .collect::<Vec<_>>();
                assert_eq!(pt_col_u32.len(), n);
                let pt_ntt = conv.ntt(&pt_col_u32);

                let convolved_ntt = conv.pointwise_mul(&v_nega_perm_a[outer_row], &pt_ntt);

                for r in 0..convd_len {
                    tmp_col[r] += convolved_ntt[r] as u64;
                }

                if outer_row % max_adds == max_adds - 1 || outer_row == db_rows / n - 1 {
                    let mut col_poly_u32 = vec![0u32; convd_len];
                    for i in 0..conv.params().crt_count {
                        for j in 0..conv.params().poly_len {
                            let val = barrett_coeff_u64(
                                conv.params(),
                                tmp_col[i * conv.params().poly_len + j],
                                i,
                            );
                            col_poly_u32[i * conv.params().poly_len + j] = val as u32;
                        }
                    }
                    let col_poly_raw = conv.raw(&col_poly_u32);
                    for i in 0..n {
                        hint_0[i * db_cols + col] += col_poly_raw[i] as u64;
                        hint_0[i * db_cols + col] %= 1u64 << 32;
                    }
                    tmp_col.fill(0);
                }
            }
        }

        hint_0
    }

    pub fn answer_query(&self, aligned_query_packed: &[u64]) -> AlignedMemory64 {
        self.multiply_batched_with_db_packed::<1>(aligned_query_packed, 1)
    }

    pub fn answer_batched_queries<const K: usize>(
        &self,
        aligned_queries_packed: &[u64],
    ) -> AlignedMemory64 {
        self.multiply_batched_with_db_packed::<K>(aligned_queries_packed, 1)
    }

    pub fn perform_offline_precomputation_simplepir(
        &self,
        measurement: Option<&mut Measurement>,
        hint_0_load: Option<&Vec<u64>>,
        hint_0_store: Option<&String>,
    ) -> OfflinePrecomputedValues<'a> {
        // Set up some parameters

        let params = self.params;
        let db_cols = params.instances * params.poly_len;
        let num_rlwe_outputs = db_cols / params.poly_len;

        // Begin offline precomputation

        let now = Instant::now();
        let hint_0 = if let Some(hint_0_load) = hint_0_load {
            hint_0_load.to_vec()
        } else {
            let hint_0_res = self.answer_hint_ring(SEED_0, db_cols);
            if let Some(hint_0_store) = hint_0_store {
                write_vec_u64_to_file(hint_0_store, &hint_0_res);
            }
            hint_0_res
        };
        // hint_0 is poly_len x db_cols
        let simplepir_prep_time_ms = now.elapsed().as_millis();
        if let Some(measurement) = measurement {
            measurement.offline.simplepir_prep_time_ms = simplepir_prep_time_ms as usize;
        }

        let now = Instant::now();
        let y_constants = generate_y_constants(&params);

        let combined = [&hint_0[..], &vec![0u64; db_cols]].concat();
        assert_eq!(combined.len(), db_cols * (params.poly_len + 1));
        let prepacked_lwe = prep_pack_many_lwes(&params, &combined, num_rlwe_outputs);

        let fake_pack_pub_params = generate_fake_pack_pub_params(&params);

        let mut precomp: Precomp = Vec::new();
        for i in 0..prepacked_lwe.len() {
            let tup = precompute_pack(
                params,
                params.poly_len_log2,
                &prepacked_lwe[i],
                &fake_pack_pub_params,
                &y_constants,
            );
            precomp.push(tup);
        }
        debug!("Precomp in {} us", now.elapsed().as_micros());

        OfflinePrecomputedValues {
            hint_0,
            y_constants,
            prepacked_lwe,
            fake_pack_pub_params,
            precomp,
        }
    }

    /// Perform SimplePIR-style YPIR
    pub fn perform_online_computation_simplepir(
        &self,
        first_dim_queries_packed: &[u64],
        offline_vals: &OfflinePrecomputedValues<'a>,
        pack_pub_params_row_1s: &[&[u64]],
        mut measurement: Option<&mut Measurement>,
    ) -> Vec<u8> {
        // Set up some parameters

        let params = self.params;

        let y_constants = &offline_vals.y_constants;
        let prepacked_lwe = &offline_vals.prepacked_lwe;
        let precomp = &offline_vals.precomp;

        // RLWE reduced moduli
        let rlwe_q_prime_1 = params.get_q_prime_1();
        let rlwe_q_prime_2 = params.get_q_prime_2();

        let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
        let db_cols = params.instances * params.poly_len;

        assert_eq!(
            first_dim_queries_packed.len(),
            params.db_rows_padded_simplepir()
        );

        // Begin online computation

        let first_pass = Instant::now();
        // debug!("Performing mul...");
        let mut intermediate = AlignedMemory64::new(db_cols);
        fast_batched_dot_product::<1, T>(
            &params,
            intermediate.as_mut_slice(),
            first_dim_queries_packed,
            db_rows,
            self.db(),
            db_rows,
            db_cols,
        );
        // debug!("Done w mul...");
        let first_pass_time_ms = first_pass.elapsed().as_millis();
        if let Some(ref mut m) = measurement {
            m.online.first_pass_time_ms = first_pass_time_ms as usize;
        }

        let ring_packing = Instant::now();
        let num_rlwe_outputs = db_cols / params.poly_len;
        let pack_pub_params_row_1s_pms =
            unpack_vec_pm(&params, 1, params.t_exp_left, pack_pub_params_row_1s[0]);
        let packed = pack_many_lwes(
            &params,
            &prepacked_lwe,
            &precomp,
            intermediate.as_slice(),
            num_rlwe_outputs,
            &pack_pub_params_row_1s_pms,
            &y_constants,
        );
        // debug!("Packed...");
        if let Some(m) = measurement {
            m.online.ring_packing_time_ms = ring_packing.elapsed().as_millis() as usize;
        }

        let mut packed_mod_switched = Vec::with_capacity(packed.len());
        for ct in packed.iter() {
            let res = ct.raw();
            let res_switched = res.switch(rlwe_q_prime_1, rlwe_q_prime_2);
            packed_mod_switched.push(res_switched);
        }

        assert_eq!(packed_mod_switched.len(), num_rlwe_outputs);

        let concated = packed_mod_switched
            .iter()
            .map(|x| x.as_slice())
            .flatten()
            .copied()
            .collect::<Vec<_>>();

        concated
    }

    pub fn perform_full_online_computation_simplepir(
        &self,
        offline_vals: &OfflinePrecomputedValues<'a>,
        query: &[u8],
    ) -> Vec<u8> {
        let first_dim_bytes_sz = self.params.db_rows() * std::mem::size_of::<u64>();
        let pub_param_bytes_sz = self.params.poly_len_log2
            * self.params.t_exp_left
            * self.params.poly_len
            * std::mem::size_of::<u64>();
        assert_eq!(query.len(), first_dim_bytes_sz + pub_param_bytes_sz);

        let first_dim_bytes = &query[..first_dim_bytes_sz];
        let pub_param_bytes = &query[first_dim_bytes_sz..];

        let first_dim = AlignedMemory64::from_bytes(first_dim_bytes);
        let pub_params = AlignedMemory64::from_bytes(pub_param_bytes);

        self.perform_online_computation_simplepir(
            first_dim.as_slice(),
            offline_vals,
            &[pub_params.as_slice()],
            None,
        )
    }

    // generic function that returns a u8 or u16:
    pub fn db(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(
                self.db_buf_aligned.as_ptr() as *const T,
                self.db_buf_aligned.len() * 8 / std::mem::size_of::<T>(),
            )
        }
    }

    pub fn db_mut(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.db_buf_aligned.as_ptr() as *mut T,
                self.db_buf_aligned.len() * 8 / std::mem::size_of::<T>(),
            )
        }
    }

    pub fn db_u16(&self) -> &[u16] {
        unsafe {
            std::slice::from_raw_parts(
                self.db_buf_aligned.as_ptr() as *const u16,
                self.db_buf_aligned.len() * 8 / std::mem::size_of::<u16>(),
            )
        }
    }

    pub fn db_u32(&self) -> &[u32] {
        unsafe {
            std::slice::from_raw_parts(
                self.db_buf_aligned.as_ptr() as *const u32,
                self.db_buf_aligned.len() * 8 / std::mem::size_of::<u32>(),
            )
        }
    }

    pub fn get_elem(&self, target_idx: usize) -> T {
        let db_cols = self.db_cols();
        let (target_row, target_col) = (target_idx / db_cols, target_idx % db_cols);
        self.get_elem_row_col(target_row, target_col)
    }

    pub fn get_elem_row_col(&self, row: usize, col: usize) -> T {
        self.db()[col * self.db_rows_padded() + row] // stored transposed
    }

    pub fn get_row(&self, row: usize) -> Vec<T> {
        let db_cols = self.db_cols();
        let mut res = Vec::with_capacity(db_cols);
        for col in 0..db_cols {
            res.push(self.get_elem_row_col(row, col));
        }
        res
        // // convert to u8 contiguously
        // let mut res_u8 = Vec::with_capacity(db_cols * std::mem::size_of::<T>());
        // for &x in res.iter() {
        //     res_u8.extend_from_slice(&x.to_u64().to_le_bytes()[..std::mem::size_of::<T>()]);
        // }
        // res_u8
    }
}

#[cfg(not(target_feature = "avx2"))]
#[allow(non_camel_case_types)]
type __m512i = u64;

pub trait ToM512 {
    fn to_m512(self) -> __m512i;
}

#[cfg(feature = "explicit_avx512")]
mod m512_impl {
    use super::*;

    impl ToM512 for *const u8 {
        #[inline(always)]
        fn to_m512(self) -> __m512i {
            unsafe { _mm512_cvtepu8_epi64(_mm_loadl_epi64(self as *const _)) }
        }
    }

    impl ToM512 for *const u16 {
        #[inline(always)]
        fn to_m512(self) -> __m512i {
            unsafe { _mm512_cvtepu16_epi64(_mm_load_si128(self as *const _)) }
        }
    }

    impl ToM512 for *const u32 {
        #[inline(always)]
        fn to_m512(self) -> __m512i {
            unsafe { _mm512_cvtepu32_epi64(_mm256_load_si256(self as *const _)) }
        }
    }
}

#[cfg(not(feature = "explicit_avx512"))]
mod m512_impl {
    use super::*;

    impl ToM512 for *const u8 {
        #[inline(always)]
        fn to_m512(self) -> __m512i {
            panic!("AVX512 implementation not being used")
        }
    }

    impl ToM512 for *const u16 {
        #[inline(always)]
        fn to_m512(self) -> __m512i {
            panic!("AVX512 implementation not being used")
        }
    }

    impl ToM512 for *const u32 {
        #[inline(always)]
        fn to_m512(self) -> __m512i {
            panic!("AVX512 implementation not being used")
        }
    }
}

pub trait ToU64 {
    fn to_u64(self) -> u64;
}

impl ToU64 for u8 {
    fn to_u64(self) -> u64 {
        self as u64
    }
}

impl ToU64 for u16 {
    fn to_u64(self) -> u64 {
        self as u64
    }
}

impl ToU64 for u32 {
    fn to_u64(self) -> u64 {
        self as u64
    }
}

impl ToU64 for u64 {
    fn to_u64(self) -> u64 {
        self
    }
}

impl ToU64 for *const u8 {
    fn to_u64(self) -> u64 {
        (unsafe { *self }) as u64
    }
}

impl ToU64 for *const u16 {
    fn to_u64(self) -> u64 {
        (unsafe { *self }) as u64
    }
}

impl ToU64 for *const u64 {
    fn to_u64(self) -> u64 {
        unsafe { *self }
    }
}

pub fn naive_multiply_matrices<T: ToU64 + Copy>(
    a: &[u32],
    a_rows: usize,
    a_cols: usize,
    b_t: &[T], // transposed
    b_rows: usize,
    b_cols: usize,
    is_b_transposd: bool,
) -> Vec<u32> {
    // performs wrapping arithmetic

    assert_eq!(a_cols, b_rows);

    // debug!("Multiplying {}x{} by {}x{}", a_rows, a_cols, b_rows, b_cols);

    let mut result = vec![0u32; a_rows * b_cols];
    for i in 0..a_rows {
        for j in 0..b_cols {
            for k in 0..a_cols {
                let a_idx = i * a_cols + k;
                let b_idx = if is_b_transposd {
                    j * b_rows + k // on purpose, since transposed
                } else {
                    k * b_cols + j
                };
                let res_idx = i * b_cols + j;

                unsafe {
                    let a_val = *a.get_unchecked(a_idx);
                    let b_val = (b_t.get_unchecked(b_idx)).to_u64() as u32;

                    result[res_idx] = result[res_idx].wrapping_add(a_val.wrapping_mul(b_val));
                }
            }
        }
    }

    result
}

// ── Cache I/O for YServer<u16> (warm-restart precompute cache) ───────────────
//
// Checked binary dump/load for the YPIR-formatted database (`db_buf_aligned`)
// and identifying flags. See `serialize::cache_io` for the format conventions
// and safety contract. Restricted to `T = u16` with padded SimplePIR rows,
// the only configuration the consumer (vote-nullifier-pir) uses today.

/// YServer dump format version. Bump on any change that breaks cache
/// validity (wire-format change OR algorithm change that produces different
/// bytes for the same input). Same convention as `PAYLOAD_FORMAT_V1` in
/// `crate::serialize`.
const YSERVER_DUMP_V1: u8 = 2;

impl<'a> YServer<'a, u16> {
    /// Dump the YPIR-formatted database (and identifying flags) to a writer.
    /// Format is binary-stable within a major version of `valar-ypir`.
    /// Intended for warm-restart caching only; see
    /// [`crate::serialize::cache_io`] for the safety contract.
    ///
    /// **Endianness:** the on-wire format is little-endian. Only compiles on
    /// little-endian targets; same constraint as
    /// `OfflinePrecomputedValues::dump_into`.
    ///
    /// # Errors
    ///
    /// Returns the underlying I/O error if the writer fails. Panics if the
    /// server was constructed with `pad_rows = false` (this dump is for the
    /// consumer's only configuration today).
    pub fn dump_into<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        assert!(self.pad_rows, "dump_into requires pad_rows = true");

        crate::serialize::cache_io::write_u8(w, YSERVER_DUMP_V1)?;
        // flags: bit 0 = pad_rows. This cache format is SP-only.
        crate::serialize::cache_io::write_u8(w, 0b01)?;
        crate::serialize::cache_io::dump_aligned_memory64(w, &self.db_buf_aligned)?;
        Ok(())
    }

    /// Load a YServer from a reader. Bounds-checks every access; returns
    /// [`crate::serialize::CacheError`] on truncation, malformed data, or
    /// version mismatch. Never panics on disk-derived input.
    ///
    /// `params` must match the params the cache was originally produced with;
    /// the consumer is responsible for verifying that via its own header.
    ///
    /// **Trailing-byte contract:** consumes exactly the bytes produced by
    /// one [`Self::dump_into`] call and stops; does NOT check for trailing
    /// bytes on the reader. Same rationale as
    /// `OfflinePrecomputedValues::load_from`'s contract.
    ///
    /// **Validation order:** the on-disk `db_buf_aligned` length is checked
    /// against the expected size derived from `params` BEFORE any
    /// allocation. A corrupted length prefix can't trigger a multi-GB
    /// allocation before being rejected.
    pub fn load_from<R: std::io::Read>(
        r: &mut R,
        params: &'a Params,
    ) -> Result<Self, crate::serialize::CacheError> {
        use crate::serialize::{cache_io, CacheError};

        let v = cache_io::read_u8(r, "YServer dump version")?;
        if v != YSERVER_DUMP_V1 {
            return Err(CacheError::Malformed {
                what: "YServer dump version",
                detail: format!("unknown version {v}, expected {YSERVER_DUMP_V1}"),
            });
        }
        // Known flag bits for YSERVER_DUMP_V1:
        //   bit 0: pad_rows  (must be 1 — only configuration we dump)
        // Bits 1..=7 are reserved; reject if set so a future format that
        // assigns them isn't silently mis-loaded by older code.
        const KNOWN_FLAGS: u8 = 0b0000_0001;
        let flags = cache_io::read_u8(r, "YServer flags")?;
        if flags & !KNOWN_FLAGS != 0 {
            return Err(CacheError::Malformed {
                what: "YServer flags",
                detail: format!(
                    "unknown flag bits set: 0x{flags:02x} (known mask 0x{KNOWN_FLAGS:02x})"
                ),
            });
        }
        let pad_rows = flags & 0b01 != 0;
        if !pad_rows {
            return Err(CacheError::Malformed {
                what: "YServer flags",
                detail: format!("this loader requires pad_rows=true; got pad_rows={pad_rows}"),
            });
        }

        // Compute the expected `db_buf_aligned` length from params and pass
        // it to the `_exact` loader so the on-disk length prefix is checked
        // BEFORE allocating. `bytes_per_pt_el = 2` for u16; db_buf_aligned
        // holds u64s, so length-in-u64 = (db_rows_padded * db_cols * 2) / 8.
        let db_rows_padded = params.db_rows_padded_simplepir();
        let db_cols = params.instances * params.poly_len;
        let expected_u64s = (db_rows_padded * db_cols * 2) / 8;
        let db_buf_aligned =
            cache_io::load_aligned_memory64_exact(r, expected_u64s, "db_buf_aligned")?;

        Ok(YServer {
            params,
            db_buf_aligned,
            phantom: std::marker::PhantomData,
            pad_rows: true,
        })
    }
}
