use spiral_rs::{arith::*, params::*, poly::*, util::*};

pub trait ModulusSwitch<'a> {
    fn switch(&self, q_1: u64, q_2: u64) -> Vec<u8>;
    fn recover(params: &'a Params, q_1: u64, q_2: u64, ciphertext: &[u8]) -> Self;
}

/// Expected byte length of a modulus-switched ciphertext.
pub fn expected_ct_byte_len(params: &Params, q_1: u64, q_2: u64) -> usize {
    let q_1_bits = (q_2 as f64).log2().ceil() as usize;
    let q_2_bits = (q_1 as f64).log2().ceil() as usize;
    let total_sz_bits = (q_1_bits + q_2_bits) * params.poly_len;
    (total_sz_bits + 7) / 8
}

impl<'a> ModulusSwitch<'a> for PolyMatrixRaw<'a> {
    fn switch(&self, q_1: u64, q_2: u64) -> Vec<u8> {
        assert_eq!(self.rows, 2);
        assert_eq!(self.cols, 1);
        let q_1_bits = (q_2 as f64).log2().ceil() as usize;
        let q_2_bits = (q_1 as f64).log2().ceil() as usize;
        let total_sz_bits = (q_1_bits + q_2_bits) * self.params.poly_len;
        let total_sz_bytes = (total_sz_bits + 7) / 8;

        let mut res = vec![0u8; total_sz_bytes];
        let mut bit_offs = 0;
        let (row_0, row_1) = (self.get_poly(0, 0), self.get_poly(1, 0));
        for z in 0..self.params.poly_len {
            let val = row_0[z];
            let val_rescaled = rescale(val, self.params.modulus, q_2);
            write_arbitrary_bits(&mut res, val_rescaled, bit_offs, q_1_bits);
            bit_offs += q_1_bits;
        }
        for z in 0..self.params.poly_len {
            let val = row_1[z];
            let val_rescaled = rescale(val, self.params.modulus, q_1);
            write_arbitrary_bits(&mut res, val_rescaled, bit_offs, q_2_bits);
            bit_offs += q_2_bits;
        }

        res
        // self.as_slice().to_vec()
    }

    fn recover(params: &'a Params, q_1: u64, q_2: u64, ciphertext: &[u8]) -> Self {
        let q_1_bits = (q_2 as f64).log2().ceil() as usize;
        let q_2_bits = (q_1 as f64).log2().ceil() as usize;
        let total_sz_bits = (q_1_bits + q_2_bits) * params.poly_len;
        let total_sz_bytes = (total_sz_bits + 7) / 8;
        assert_eq!(ciphertext.len(), total_sz_bytes);

        let mut res = PolyMatrixRaw::zero(params, 2, 1);
        let mut bit_offs = 0;
        let (row_0, row_1) = res.data.as_mut_slice().split_at_mut(params.poly_len);
        for z in 0..params.poly_len {
            let val = read_arbitrary_bits(&ciphertext, bit_offs, q_1_bits);
            row_0[z] = rescale(val, q_2, params.modulus);
            bit_offs += q_1_bits;
        }
        for z in 0..params.poly_len {
            let val = read_arbitrary_bits(&ciphertext, bit_offs, q_2_bits);
            row_1[z] = rescale(val, q_1, params.modulus);
            bit_offs += q_2_bits;
        }

        res
        // let mut res = PolyMatrixRaw::zero(params, 2, 1);
        // res.as_mut_slice().copy_from_slice(ciphertext);
        // res
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::params::{params_for_scenario_simplepir, GetQPrime};

    fn test_params() -> Params {
        params_for_scenario_simplepir(1 << 14, 2048 * 14)
    }

    #[test]
    #[should_panic(expected = "assert")]
    fn recover_empty_ciphertext_panics() {
        let params = test_params();
        let q1 = params.get_q_prime_1();
        let q2 = params.get_q_prime_2();
        let _ = PolyMatrixRaw::recover(&params, q1, q2, &[]);
    }

    #[test]
    #[should_panic(expected = "assert")]
    fn recover_single_byte_panics() {
        let params = test_params();
        let q1 = params.get_q_prime_1();
        let q2 = params.get_q_prime_2();
        let _ = PolyMatrixRaw::recover(&params, q1, q2, &[0xFF]);
    }

    #[test]
    #[should_panic(expected = "assert")]
    fn recover_too_short_panics() {
        let params = test_params();
        let q1 = params.get_q_prime_1();
        let q2 = params.get_q_prime_2();
        let expected_len = expected_ct_byte_len(&params, q1, q2);
        let short = vec![0u8; expected_len - 1];
        let _ = PolyMatrixRaw::recover(&params, q1, q2, &short);
    }

    #[test]
    #[should_panic(expected = "assert")]
    fn recover_too_long_panics() {
        let params = test_params();
        let q1 = params.get_q_prime_1();
        let q2 = params.get_q_prime_2();
        let expected_len = expected_ct_byte_len(&params, q1, q2);
        let long = vec![0u8; expected_len + 1];
        let _ = PolyMatrixRaw::recover(&params, q1, q2, &long);
    }

    #[test]
    fn recover_exact_length_all_zeros() {
        let params = test_params();
        let q1 = params.get_q_prime_1();
        let q2 = params.get_q_prime_2();
        let expected_len = expected_ct_byte_len(&params, q1, q2);
        let zeros = vec![0u8; expected_len];
        let result = PolyMatrixRaw::recover(&params, q1, q2, &zeros);
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 1);
    }

    #[test]
    fn recover_exact_length_all_ff() {
        let params = test_params();
        let q1 = params.get_q_prime_1();
        let q2 = params.get_q_prime_2();
        let expected_len = expected_ct_byte_len(&params, q1, q2);
        let ff = vec![0xFFu8; expected_len];
        let result = PolyMatrixRaw::recover(&params, q1, q2, &ff);
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 1);
        for val in result.as_slice() {
            assert!(*val < params.modulus, "recovered value must be < modulus");
        }
    }

    #[test]
    fn switch_then_recover_preserves_structure() {
        let params = test_params();
        let q1 = params.get_q_prime_1();
        let q2 = params.get_q_prime_2();

        let mut ct = PolyMatrixRaw::zero(&params, 2, 1);
        for i in 0..params.poly_len {
            ct.data[i] = (i as u64 * 1000) % params.modulus;
            ct.data[params.poly_len + i] = (i as u64 * 777) % params.modulus;
        }

        let switched = ct.switch(q1, q2);
        let recovered = PolyMatrixRaw::recover(&params, q1, q2, &switched);

        assert_eq!(recovered.rows, 2);
        assert_eq!(recovered.cols, 1);
        for val in recovered.as_slice() {
            assert!(*val < params.modulus, "recovered value must be < modulus");
        }
    }

    #[test]
    fn recover_random_bytes_produces_in_range_values() {
        let params = test_params();
        let q1 = params.get_q_prime_1();
        let q2 = params.get_q_prime_2();
        let expected_len = expected_ct_byte_len(&params, q1, q2);
        let random_bytes: Vec<u8> = (0..expected_len).map(|_| fastrand::u8(..)).collect();
        let result = PolyMatrixRaw::recover(&params, q1, q2, &random_bytes);
        for val in result.as_slice() {
            assert!(
                *val < params.modulus,
                "adversarial bytes must still produce in-range coefficients after rescale"
            );
        }
    }

    // -- SP-specific modulus switch tests --

    fn sp_params() -> Params {
        crate::params::params_for_scenario_simplepir(1 << 14, 2048 * 14)
    }

    #[test]
    fn sp_switch_recover_preserves_structure() {
        let params = sp_params();
        let q1 = params.get_q_prime_1();
        let q2 = params.get_q_prime_2();

        let mut ct = PolyMatrixRaw::zero(&params, 2, 1);
        for i in 0..params.poly_len {
            ct.data[i] = (i as u64 * 3001) % params.modulus;
            ct.data[params.poly_len + i] = (i as u64 * 4999) % params.modulus;
        }

        let switched = ct.switch(q1, q2);
        assert_eq!(switched.len(), expected_ct_byte_len(&params, q1, q2));

        let recovered = PolyMatrixRaw::recover(&params, q1, q2, &switched);
        assert_eq!(recovered.rows, 2);
        assert_eq!(recovered.cols, 1);
        for val in recovered.as_slice() {
            assert!(*val < params.modulus);
        }
    }

    #[test]
    fn sp_switch_byte_length_matches_expected() {
        let params = sp_params();
        let q1 = params.get_q_prime_1();
        let q2 = params.get_q_prime_2();

        let ct = PolyMatrixRaw::zero(&params, 2, 1);
        let switched = ct.switch(q1, q2);
        assert_eq!(switched.len(), expected_ct_byte_len(&params, q1, q2));
    }

    #[test]
    fn sp_recover_random_bytes_in_range() {
        let params = sp_params();
        let q1 = params.get_q_prime_1();
        let q2 = params.get_q_prime_2();
        let len = expected_ct_byte_len(&params, q1, q2);
        let random_bytes: Vec<u8> = (0..len).map(|_| fastrand::u8(..)).collect();
        let result = PolyMatrixRaw::recover(&params, q1, q2, &random_bytes);
        for val in result.as_slice() {
            assert!(*val < params.modulus);
        }
    }
}
