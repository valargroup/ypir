use std::f64::consts::PI;

use log::debug;

use serde::Serialize;
use spiral_rs::{arith::rescale, client::Client, gadget::get_bits_per, params::Params, poly::*};

use super::{
    lwe::LWEParams,
    params::{assert_valid_ypir_sp_params, params_for_scenario, DbRowsCols, GetQPrime},
};

/*
    \taudouble &= \frac{\qtilde_{2, 2}}{2 p} - (\qtilde_{2, 2} \bmod p) - \frac{1}{2}
        \left( 2 + (\qtilde_{2, 2} \bmod p) + (\qtilde_{2, 2} / q_2)(q_2 \bmod p) \right) \\
    \sigmadouble^2 &\le (\qtilde_{2, 2} / \qtilde_{2, 1})^2 d_2 \sigma_2^2 / 4 +
        (\qtilde_{2, 2} / q_2)^2 (\sigma_2^2 / 4) (\ell_2 p^2 + (d_2^2 - 1)(t d_2 z^2) / 3) \\
    \tausimple &= \frac{\qtilde_1}{2N} - (\qtilde_1 \bmod N) -
      \frac{1}{2} \left( 2 + \qtilde_1 \bmod N + (\qtilde_1 / q_1) (q_1 \bmod N) \right) / 2 \\
    \sigmasimple &\le d_1 \sigma_1^2 / 4 + (\qtilde_1 / q_1)^2 \ell_1 N^2 \sigma_1^2 / 4\iftoggle{fullversion}{.}{}

    t = \lfloor \log_z q_2 \rfloor + 1
*/

fn tau_double(q_2: f64, qtilde_2_2: f64, p: f64) -> f64 {
    let term1 = qtilde_2_2 / (2.0 * p);
    let term2 = -(qtilde_2_2 % p);
    let term3 = -(1.0 / 2.0) * (2.0 + (qtilde_2_2 % p) + (qtilde_2_2 / q_2) * (q_2 % p)) / 2.0;

    term1 + term2 + term3
}

fn sigma_2_double(
    d_2: f64,
    q_2: f64,
    qtilde_2_2: f64,
    qtilde_2_1: f64,
    sigma_2: f64,
    p: f64,
    z: f64,
    l_2: f64,
) -> f64 {
    let t = (q_2.log2() / z.log2()).floor() + 1.0;
    let term1 = (qtilde_2_2 / qtilde_2_1).powi(2) * d_2 * sigma_2.powi(2) / 4.0;
    let term2 = (qtilde_2_2 / q_2).powi(2)
        * (sigma_2.powi(2) / 4.0)
        * (l_2 * p.powi(2) + (d_2.powi(2) - 1.0) * (t * d_2 * z.powi(2)) / 3.0);

    term1 + term2
}

fn delta_double(
    d_2: f64,
    q_2: f64,
    qtilde_2_2: f64,
    qtilde_2_1: f64,
    sigma_2: f64,
    p: f64,
    z: f64,
    l_2: f64,
) -> (f64, f64) {
    let tau = tau_double(q_2, qtilde_2_2, p);
    let sigma_2 = sigma_2_double(d_2, q_2, qtilde_2_2, qtilde_2_1, sigma_2, p, z, l_2);
    let delta = 2.0 * (-PI * tau.powi(2) / sigma_2).exp();
    (delta, sigma_2)
}

fn log2_delta(tau: f64, sigma_2: f64) -> f64 {
    1.0 - PI * tau.powi(2) / (sigma_2 * std::f64::consts::LN_2)
}

fn log2_union_bound(log2_probability: f64, event_count: usize) -> f64 {
    assert!(event_count > 0);
    log2_probability + (event_count as f64).log2()
}

fn tau_simple(q_1: f64, qtilde_1: f64, n: f64) -> f64 {
    let term1 = qtilde_1 / (2.0 * n);
    let term2 = -(qtilde_1 % n);
    let term3 = -(1.0 / 2.0) * (2.0 + qtilde_1 % n + (qtilde_1 / q_1) * (q_1 % n)) / 2.0;

    term1 + term2 + term3
}

fn sigma_2_simple(d_1: f64, q_1: f64, qtilde_1: f64, sigma_1: f64, n: f64, l_1: f64) -> f64 {
    let term1 = d_1 * sigma_1.powi(2) / 4.0;
    let term2 = (qtilde_1 / q_1).powi(2) * l_1 * n.powi(2) * sigma_1.powi(2) / 4.0;
    term1 + term2
}

fn delta_simple(d_1: f64, q_1: f64, qtilde_1: f64, sigma_1: f64, n: f64, l_1: f64) -> (f64, f64) {
    let tau = tau_simple(q_1, qtilde_1, n);
    let sigma_2 = sigma_2_simple(d_1, q_1, qtilde_1, sigma_1, n, l_1);
    let delta = 2.0 * (-PI * tau.powi(2) / sigma_2).exp();
    (delta, sigma_2)
}

pub fn simplepir_correctness(_lwe_n: f64, lwe_q: f64, lwe_s: f64, lwe_p: f64, upper_n: f64) -> f64 {
    let log2_sqrt_upper_n = (upper_n as f64).sqrt().log2();
    let pt_term = lwe_p.log2() * 2.0 - 2.0; // (p/2)^2
    let noise_term = lwe_s.log2() * 2.0;
    let log2_s_2 = log2_sqrt_upper_n + pt_term + noise_term; // log2(s'^2)
    debug!("log2_s_2: {}", log2_s_2);

    // get probability that noise is less than q/2p
    // subgaussian formula: Pr[noise > T] = 2 \exp(-\pi T^2 / s'^2)
    let noise_threshold = lwe_q / (2.0 * lwe_p);
    let noise_threshold_term = -PI * noise_threshold.powi(2) / 2f64.powf(log2_s_2);
    let err_prob = 2.0 * noise_threshold_term.exp();
    let log2_err_prob = err_prob.log2();
    debug!("log2_err_prob: {}", log2_err_prob);
    log2_err_prob
}

pub struct YPIRSchemeParams {
    pub l1: f64,
    pub d1: f64,
    pub p1: f64,
    pub s1: f64,
    pub q1: f64,
    pub q1_prime: f64,

    pub d2: f64,
    pub l2: f64,
    pub p2: f64,
    pub s2: f64,
    pub q2: f64,
    pub q2_1_prime: f64,
    pub q2_2_prime: f64,
    pub t: f64,
}

impl Default for YPIRSchemeParams {
    fn default() -> Self {
        let max_db_bits = 64 * (1 << 33); // 64 GB
        let lwe_params = LWEParams::default();
        let params = params_for_scenario(max_db_bits, 1);
        Self::from_params(&params, &lwe_params)
    }
}

impl YPIRSchemeParams {
    pub fn from_params(params: &Params, lwe_params: &LWEParams) -> Self {
        let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
        let db_cols = 1 << (params.db_dim_2 + params.poly_len_log2);

        // Warning: the paper uses reversed reference to q_prime_1 and q_prime_2
        let q2_1_prime = params.get_q_prime_2() as f64;
        let q2_2_prime: f64 = params.get_q_prime_1() as f64;

        assert!(q2_1_prime >= q2_2_prime);

        Self {
            l1: db_rows as f64, // max size for 64 GB
            d1: lwe_params.n as f64,
            p1: lwe_params.pt_modulus as f64,
            s1: lwe_params.noise_width,
            q1: lwe_params.modulus as f64,
            q1_prime: lwe_params.get_q_prime_1() as f64,

            d2: params.poly_len as f64,
            l2: db_cols as f64,
            p2: params.pt_modulus as f64,
            s2: params.noise_width,
            q2: params.modulus as f64,
            q2_1_prime,
            q2_2_prime,
            t: params.t_exp_left as f64,
        }
    }

    /// Returns the modulus for plaintext database elements, N in the paper, and p_1 in the implementation.
    pub fn n(&self) -> f64 {
        self.p1
    }

    /// Returns the gadget decomposition base, z in the paper. In the implementation, we just set t = 3.
    pub fn z(&self) -> f64 {
        let modulus_log2 = self.q2.log2().ceil() as usize;
        let t = self.t as usize;
        let bits_per = if t == modulus_log2 {
            1
        } else {
            modulus_log2 / t + 1
        };
        (1u64 << bits_per) as f64
    }

    pub fn delta_simple(&self) -> (f64, f64) {
        delta_simple(self.d1, self.q1, self.q1_prime, self.s1, self.n(), self.l1)
    }

    pub fn delta_double(&self) -> (f64, f64) {
        delta_double(
            self.d2,
            self.q2,
            self.q2_2_prime,
            self.q2_1_prime,
            self.s2,
            self.p2,
            self.z(),
            self.l2,
        )
    }

    pub fn delta(&self) -> f64 {
        let (delta_simple, _) = self.delta_simple();
        let (delta_double, _) = self.delta_double();
        delta_simple + delta_double
    }

    /// The square of the subgaussian parameter for the outer ciphertext noise.
    pub fn expected_outer_noise(&self) -> f64 {
        let (_, sigma_2) = self.delta_double();
        (self.q2 / self.q2_2_prime).powi(2) * sigma_2
    }
}

/// Multiplicative slack on the analytical automorphism-packing term.
///
/// The `(d^2 - 1)(t*d*z^2)/3` expression is the YPIR analysis' bound on the
/// key-switching noise of a single automorphism; composing it across
/// `log2(poly_len)` packing levels the way this model does is a heuristic, and
/// measurement puts the real contribution roughly 2.3x above it (visible at
/// 2048, where that term dominates; invisible at 4096, where the modulus
/// switch dominates). The slack makes the estimate conservative against the
/// measurements sampled so far, but does not make the composition a proof.
///
/// `noise_bound_dominates_measurement` in `scheme.rs` fails if sampled noise
/// crosses the resulting model, so this constant cannot silently rot.
pub const PACKING_TERM_SLACK: f64 = 2.5;

/// A term-by-term model of the noise of a decoded YPIR-SP response, together
/// with its estimated failure probability.
///
/// All noise quantities are subgaussian *width squared* (`sigma^2 * 2*pi`) in
/// the modulus-switched domain, i.e. the domain in which `tau` is the decoding
/// half-window. Multiply by `(q / q_prime_1)^2` to compare against
/// [`measure_noise_width_squared`], which measures in the `q` domain.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct YPIRSPNoiseReport {
    pub poly_len: usize,
    pub instances: usize,
    /// Total coefficients decoded for one SimplePIR response.
    pub decoded_coefficients: usize,
    pub db_rows: usize,
    pub gadget_digits: usize,
    pub gadget_bits_per_digit: usize,
    pub gadget_base: u64,
    /// Row 0 is switched to `q_prime_2`; its rounding error is multiplied by
    /// the secret key when the client decrypts.
    pub switch_secret_term: f64,
    /// Row 1 is switched to `q_prime_1`; its rounding error enters directly.
    pub switch_rounding_term: f64,
    /// SimplePIR first dimension: `db_rows` query noises scaled by database
    /// entries. This is the only `db_rows`-dependent term.
    pub first_dim_term: f64,
    /// Automorphism key-switching during ring packing, including
    /// [`PACKING_TERM_SLACK`].
    pub packing_term: f64,
    pub packing_term_slack: f64,
    /// Decoding half-window in the switched domain: decryption is correct iff
    /// the noise stays inside `+/- tau`.
    pub tau: f64,
    pub modeled_noise_width_squared: f64,
    /// Modeled two-sided failure probability for one decoded coefficient.
    pub modeled_coefficient_failure_log2: f64,
    /// Union-bound estimate for any decoded coefficient failing in one response.
    pub modeled_failure_log2: f64,
}

/// Models the decryption-failure probability of the YPIR+SimplePIR pipeline.
///
/// Unlike a DoublePIR bound this has no second-matmul term (YPIR-SP packs the
/// first-dimension result and sends it), and it does depend on `db_rows`, which
/// is where the first dimension's noise enters. Panics unless `params` is a
/// supported YPIR-SP set, since the terms are only calibrated for those.
pub fn ypir_sp_noise_report(params: &Params) -> YPIRSPNoiseReport {
    assert_valid_ypir_sp_params(params);

    let q = params.modulus as f64;
    let q_prime_large = params.get_q_prime_2() as f64;
    let q_prime_small = params.get_q_prime_1() as f64;
    let d = params.poly_len as f64;
    let p = params.pt_modulus as f64;
    let sigma_e_2 = params.noise_width.powi(2);
    let db_rows = params.db_rows();
    let gadget_bits_per_digit = get_bits_per(params, params.t_exp_left);
    let gadget_base = 1u64 << gadget_bits_per_digit;
    let z = gadget_base as f64;
    let t = params.t_exp_left as f64;

    // Scales a pre-switch (mod q) width^2 into the switched domain.
    let to_switched = (q_prime_small / q).powi(2);

    let switch_secret_term = (q_prime_small / q_prime_large).powi(2) * d * sigma_e_2 / 4.0;
    // Uniform over one quantisation step of the switched domain: +/- 1/2, so
    // variance 1/12 and width^2 = 2*pi/12.
    let switch_rounding_term = 2.0 * PI / 12.0;
    // Query noise has width sigma_e: `get_scaled_regev_sample` scales the error
    // by d^-1, which ring packing's factor of d undoes. Database entries are
    // bounded by p (worst case, not the p^2/3 of a uniform database).
    let first_dim_term = to_switched * sigma_e_2 * db_rows as f64 * p.powi(2);
    let packing_term =
        to_switched * (sigma_e_2 / 4.0) * (d.powi(2) - 1.0) * (t * d * z.powi(2)) / 3.0
            * PACKING_TERM_SLACK;

    let modeled_noise_width_squared =
        switch_secret_term + switch_rounding_term + first_dim_term + packing_term;
    let tau = tau_double(q, q_prime_small, p);
    let decoded_coefficients = params
        .instances
        .checked_mul(params.poly_len)
        .expect("decoded coefficient count overflow");
    let modeled_coefficient_failure_log2 = log2_delta(tau, modeled_noise_width_squared);

    YPIRSPNoiseReport {
        poly_len: params.poly_len,
        instances: params.instances,
        decoded_coefficients,
        db_rows,
        gadget_digits: params.t_exp_left,
        gadget_bits_per_digit,
        gadget_base,
        switch_secret_term,
        switch_rounding_term,
        first_dim_term,
        packing_term,
        packing_term_slack: PACKING_TERM_SLACK,
        tau,
        modeled_noise_width_squared,
        modeled_coefficient_failure_log2,
        modeled_failure_log2: log2_union_bound(
            modeled_coefficient_failure_log2,
            decoded_coefficients,
        ),
    }
}

impl YPIRSPNoiseReport {
    /// The modeled width squared expressed in the `q` domain, directly comparable with
    /// [`measure_noise_width_squared`].
    pub fn noise_width_squared_bound_q_domain(&self, params: &Params) -> f64 {
        let scale = (params.modulus as f64 / params.get_q_prime_1() as f64).powi(2);
        self.modeled_noise_width_squared * scale
    }

    /// Ratio of the bound to the `sigma^2` that would put the response-level
    /// union bound at exactly `2^-40`. Below 1.0 means the target is met; the
    /// reciprocal is how much noise headroom the parameter set has.
    pub fn utilisation_at_2_pow_minus_40(&self) -> f64 {
        let exponent_at_target = 41.0 + (self.decoded_coefficients as f64).log2();
        let sigma_2_at_target =
            PI * self.tau.powi(2) / (exponent_at_target * std::f64::consts::LN_2);
        self.modeled_noise_width_squared / sigma_2_at_target
    }
}

pub fn measure_noise_width_squared<'a>(
    params: &Params,
    client: &Client<'a>,
    ct: &PolyMatrixNTT<'a>,
    pt: &PolyMatrixRaw<'a>,
    coeffs_to_measure: usize,
) -> f64 {
    assert!(coeffs_to_measure > 0);
    let dec_result = client.decrypt_matrix_reg(ct).raw();
    assert!(coeffs_to_measure <= dec_result.data.len());
    assert!(coeffs_to_measure <= pt.data.len());
    let mut total = 0f64;
    for i in 0..coeffs_to_measure {
        // let decrypted_val = wrapped(dec_result.data[i], params.modulus);
        // let true_val = wrapped(
        //     rescale(pt.data[i], params.pt_modulus, params.modulus),
        //     params.modulus,
        // );
        let decrypted_val = dec_result.data[i];
        let true_val = rescale(pt.data[i], params.pt_modulus, params.modulus);
        let diff = decrypted_val.abs_diff(true_val);
        let centered_diff = diff.min(params.modulus - diff);
        let noise_2 = (centered_diff as f64).powi(2);
        assert!(noise_2 >= 0.0);
        // if noise_2.log2() >= 77.0 {
        //     debug!(
        //         "i: {}, noise_2: {}, diff: {}, diff_mod: {}, decrypted_val: {}, true_val: {}",
        //         i, noise_2, diff, diff_mod, decrypted_val, true_val
        //     );
        // }
        total += noise_2;
    }
    let variance = total / coeffs_to_measure as f64;
    assert!(variance >= 0.0);

    // noise_standard_deviation * sqrt(2*pi) == subg_width
    // variance * 2*pi == subg_width^2
    let subg_width_2 = variance * 2.0 * PI;
    subg_width_2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{params_for_scenario_simplepir_with_config, YPIRSPConfig};

    #[test]
    fn test_new_calc() {
        let ysp = YPIRSchemeParams::default();
        let (delta_simple, _) = ysp.delta_simple();
        debug!("log2(delta_simple): {}", delta_simple.log2());

        let (delta_double, _) = ysp.delta_double();
        debug!("log2(delta_double): {}", delta_double.log2());

        let expected_outer_noise = ysp.expected_outer_noise();
        debug!(
            "log2(expected_outer_noise): {}",
            expected_outer_noise.log2()
        );
    }

    #[test]
    fn test_ypir_scheme_params() {
        let ysp = YPIRSchemeParams::default();
        let simple_log2_delta = ysp.delta_simple().0.log2();
        let double_log2_delta = ysp.delta_double().0.log2();
        let total_log2_delta = ysp.delta().log2();

        // Using the production gadget base (2^19) exposes that the existing
        // YPIR-double parameters do not meet the previous 2^-40 target. Keep
        // this characterization explicit rather than silently dropping the
        // two-dimensional path's regression coverage.
        assert!(
            (simple_log2_delta - -96.70).abs() < 0.05,
            "simple_log2_delta: {simple_log2_delta}"
        );
        assert!(
            (double_log2_delta - -26.74).abs() < 0.05,
            "double_log2_delta: {double_log2_delta}"
        );
        assert!(
            (total_log2_delta - -26.74).abs() < 0.05,
            "total_log2_delta: {total_log2_delta}"
        );
        assert!(
            total_log2_delta > -40.0,
            "update this characterization if the default parameters are strengthened"
        );
    }

    #[test]
    fn test_ypir_sp_noise_reports_match_gadget_configuration() {
        let params_2048 =
            params_for_scenario_simplepir_with_config(1 << 14, 1 << 17, YPIRSPConfig::degree_2048());
        let params_4096 =
            params_for_scenario_simplepir_with_config(1 << 14, 1 << 17, YPIRSPConfig::degree_4096());
        let report_2048 = ypir_sp_noise_report(&params_2048);
        let report_4096 = ypir_sp_noise_report(&params_4096);

        assert_eq!(report_2048.gadget_bits_per_digit, 19);
        assert_eq!(report_2048.gadget_base, 1 << 19);
        assert_eq!(report_4096.gadget_bits_per_digit, 15);
        assert_eq!(report_4096.gadget_base, 1 << 15);
        assert_eq!(
            report_2048.decoded_coefficients,
            report_2048.instances * report_2048.poly_len
        );
        assert_eq!(
            report_4096.decoded_coefficients,
            report_4096.instances * report_4096.poly_len
        );
        assert!(
            report_2048.modeled_failure_log2 > -40.0,
            "2048 with calibrated packing slack must not claim response-wide 2^-40: {}",
            report_2048.modeled_failure_log2
        );
        assert!(report_4096.modeled_failure_log2 < -40.0);
    }

    #[test]
    fn test_ypir_sp_failure_bound_covers_the_full_response() {
        let small = ypir_sp_noise_report(&params_for_scenario_simplepir_with_config(
            1 << 14,
            1,
            YPIRSPConfig::degree_2048(),
        ));
        let one_mib = ypir_sp_noise_report(&params_for_scenario_simplepir_with_config(
            1 << 14,
            1 << 23,
            YPIRSPConfig::degree_2048(),
        ));

        for report in [&small, &one_mib] {
            let union_factor_log2 = (report.decoded_coefficients as f64).log2();
            assert!(
                (report.modeled_failure_log2
                    - report.modeled_coefficient_failure_log2
                    - union_factor_log2)
                    .abs()
                    < 1e-12,
                "response bound must include all {} decoded coefficients",
                report.decoded_coefficients
            );
        }

        assert!(one_mib.instances > small.instances);
        assert!(
            one_mib.modeled_failure_log2 > -40.0,
            "2048-degree 1 MiB responses must not be reported as meeting 2^-40: {}",
            one_mib.modeled_failure_log2
        );
        assert!(
            one_mib.utilisation_at_2_pow_minus_40() > 1.0,
            "response-aware utilization must show that the target is missed"
        );
    }

    /// The extra gadget digit is supposed to more than pay for the larger ring.
    /// With measurement-calibrated packing slack the 2048 set misses response-wide
    /// 2^-40 on this shape, while 4096 still clears it by a wide margin.
    #[test]
    fn test_4096_has_more_noise_headroom_than_2048() {
        let params_2048 =
            params_for_scenario_simplepir_with_config(1 << 14, 1 << 17, YPIRSPConfig::degree_2048());
        let params_4096 =
            params_for_scenario_simplepir_with_config(1 << 14, 1 << 17, YPIRSPConfig::degree_4096());
        let report_2048 = ypir_sp_noise_report(&params_2048);
        let report_4096 = ypir_sp_noise_report(&params_4096);

        let util_2048 = report_2048.utilisation_at_2_pow_minus_40();
        let util_4096 = report_4096.utilisation_at_2_pow_minus_40();
        let headroom_4096 = 1.0 / util_4096;
        debug!("2^-40 utilisation: 2048 = {util_2048:.2}, 4096 = {util_4096:.2}");

        assert!(
            util_2048 > 1.0,
            "2048 with calibrated packing slack should miss response-wide 2^-40: util={util_2048}"
        );
        assert!(headroom_4096 > 1.0, "4096 misses 2^-40: {headroom_4096}");
        assert!(
            report_4096.modeled_failure_log2 < report_2048.modeled_failure_log2 - 100.0,
            "4096 should remain far safer than 2048, got {} vs {}",
            report_4096.modeled_failure_log2,
            report_2048.modeled_failure_log2
        );
    }

    /// The first dimension is the only `db_rows`-dependent contribution, and the
    /// old model omitted it entirely, making the report identical for a 4096-row
    /// and a 2^30-row database. Assert the dependence now exists and that it
    /// stays sub-dominant for the shipped sets, since that is what makes the
    /// 4096 margin insensitive to database size.
    #[test]
    fn test_noise_bound_depends_on_db_rows() {
        for config in [YPIRSPConfig::degree_2048(), YPIRSPConfig::degree_4096()] {
            let small = ypir_sp_noise_report(&params_for_scenario_simplepir_with_config(
                1 << 14,
                1 << 17,
                config,
            ));
            let large = ypir_sp_noise_report(&params_for_scenario_simplepir_with_config(
                1 << 30,
                1 << 17,
                config,
            ));

            assert!(large.db_rows > small.db_rows);
            assert!(
                large.first_dim_term > small.first_dim_term,
                "poly_len {}: first-dimension term must grow with db_rows",
                config.poly_len()
            );
            assert!(
                large.modeled_noise_width_squared > small.modeled_noise_width_squared,
                "poly_len {}: total bound must grow with db_rows",
                config.poly_len()
            );
            // Sub-dominant: even at 2^30 rows the first dimension must not be
            // what decides correctness, else the margin would erode with scale.
            assert!(
                large.first_dim_term < 0.1 * large.modeled_noise_width_squared,
                "poly_len {}: first-dimension term became dominant at 2^30 rows ({} of {})",
                config.poly_len(),
                large.first_dim_term,
                large.modeled_noise_width_squared
            );
            if config.poly_len() >= 4096 {
                assert!(
                    large.modeled_failure_log2 < -40.0,
                    "poly_len {}: 2^30 rows misses 2^-40 ({})",
                    config.poly_len(),
                    large.modeled_failure_log2
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "unsupported YPIR-SP parameters")]
    fn test_noise_report_rejects_unvalidated_params() {
        let mut params =
            params_for_scenario_simplepir_with_config(1 << 14, 1 << 17, YPIRSPConfig::degree_4096());
        params.t_exp_left = 3;
        let _ = ypir_sp_noise_report(&params);
    }
}
