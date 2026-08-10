use log::debug;
use serde_json::Value;

use spiral_rs::{arith::*, params::*};

use super::lwe::LWEParams;

static DEFAULT_MODULI: [u64; 2] = [268369921u64, 249561089u64];
const DEF_MOD_STR: &str = "[\"268369921\", \"249561089\"]";
pub const DEFAULT_POLY_LEN: usize = 2048;

/// Client-selectable parameters for the YPIR+SimplePIR construction.
///
/// Only the audited 2048-degree set and the experimental 4096-degree set are
/// accepted. The latter uses an extra gadget digit to offset the additional
/// packing noise introduced by the larger ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct YPIRSPConfig {
    poly_len: usize,
    t_exp_left: usize,
}

impl YPIRSPConfig {
    pub const fn degree_2048() -> Self {
        Self {
            poly_len: 2048,
            t_exp_left: 3,
        }
    }

    pub const fn degree_4096() -> Self {
        Self {
            poly_len: 4096,
            t_exp_left: 4,
        }
    }

    pub fn for_poly_len(poly_len: usize) -> Self {
        match poly_len {
            2048 => Self::degree_2048(),
            4096 => Self::degree_4096(),
            _ => panic!("YPIR-SP poly_len must be 2048 or 4096"),
        }
    }

    pub const fn poly_len(self) -> usize {
        self.poly_len
    }

    pub const fn t_exp_left(self) -> usize {
        self.t_exp_left
    }

    fn validated_values(self) -> (usize, usize) {
        assert!(
            matches!((self.poly_len, self.t_exp_left), (2048, 3) | (4096, 4)),
            "YPIR-SP configuration must be (poly_len=2048, t_exp_left=3) or \
             (poly_len=4096, t_exp_left=4)"
        );
        (self.poly_len, self.t_exp_left)
    }
}

impl Default for YPIRSPConfig {
    fn default() -> Self {
        Self::degree_2048()
    }
}

fn ext_params_from_json(json_str: &str) -> Params {
    let v: Value = serde_json::from_str(json_str).unwrap();

    let n = v["n"].as_u64().unwrap() as usize;
    let db_dim_1 = v["nu_1"].as_u64().unwrap() as usize;
    let db_dim_2 = v["nu_2"].as_u64().unwrap() as usize;
    let instances = v["instances"].as_u64().unwrap_or(1) as usize;
    let p = v["p"].as_u64().unwrap();
    let q2_bits = u64::max(v["q2_bits"].as_u64().unwrap(), MIN_Q2_BITS);
    let t_gsw = v["t_gsw"].as_u64().unwrap() as usize;
    let t_conv = v["t_conv"].as_u64().unwrap() as usize;
    let t_exp_left = v["t_exp_left"].as_u64().unwrap() as usize;
    let t_exp_right = v["t_exp_right"].as_u64().unwrap() as usize;
    let do_expansion = v.get("direct_upload").is_none();

    let poly_len = v["poly_len"].as_u64().unwrap_or(DEFAULT_POLY_LEN as u64) as usize;
    let mut db_item_size = v["db_item_size"].as_u64().unwrap_or(0) as usize;
    if db_item_size == 0 {
        db_item_size = instances * n * n;
        db_item_size = db_item_size * poly_len * log2_ceil(p) as usize / 8;
    }

    let version = v["version"].as_u64().unwrap_or(0) as usize;

    let moduli = v["moduli"]
        .as_array()
        .map(|x| {
            x.as_slice()
                .iter()
                .map(|y| {
                    y.as_u64()
                        .unwrap_or_else(|| y.as_str().unwrap().parse().unwrap())
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or(DEFAULT_MODULI.to_vec());
    let noise_width = v["noise_width"].as_f64().unwrap_or(6.4);

    Params::init(
        poly_len,
        &moduli,
        noise_width,
        n,
        p,
        q2_bits,
        t_conv,
        t_exp_left,
        t_exp_right,
        t_gsw,
        do_expansion,
        db_dim_1,
        db_dim_2,
        instances,
        db_item_size,
        version,
    )
}

fn internal_params_for(
    nu_1: usize,
    nu_2: usize,
    p: u64,
    q2_bits: usize,
    t_exp_left: usize,
    moduli: &str,
    poly_len: usize,
) -> Params {
    ext_params_from_json(&format!(
        r#"
        {{
            "n": 1,
            "nu_1": {},
            "nu_2": {},
            "p": {},
            "q2_bits": {},
            "t_gsw": 3,
            "t_conv": 4,
            "t_exp_left": {},
            "t_exp_right": 2,
            "instances": 1,
            "db_item_size": 0,
            "moduli": {},
            "noise_width": 16.042421,
            "poly_len": {}
        }}
        "#,
        nu_1, nu_2, p, q2_bits, t_exp_left, moduli, poly_len
    ))
}

pub fn params_for_scenario(num_items: u64, item_size_bits: u64) -> Params {
    let total_db_bytes = num_items * item_size_bits / 8;
    let lwe_pt_word_bytes = 1;
    let num_items = total_db_bytes / lwe_pt_word_bytes;
    let num_tiles = num_items as f64 / (DEFAULT_POLY_LEN * DEFAULT_POLY_LEN) as f64;
    let num_tiles_usize = num_tiles.ceil() as usize;
    let num_tiles_log2 = (num_tiles_usize as f64).log2().ceil() as usize;

    let (nu_1, nu_2) = if num_tiles_log2 % 2 == 0 {
        (num_tiles_log2 / 2, num_tiles_log2 / 2)
    } else {
        ((num_tiles_log2 + 1) / 2, (num_tiles_log2 - 1) / 2)
    };

    debug!("chose nu_1: {}, nu_2: {}", nu_1, nu_2);

    let p = 32768;
    let q2_bits = 28;
    let t_exp_left = 3;

    internal_params_for(
        nu_1,
        nu_2,
        p,
        q2_bits,
        t_exp_left,
        DEF_MOD_STR,
        DEFAULT_POLY_LEN,
    )
}

pub fn params_for_scenario_simplepir(num_items: u64, item_size_bits: u64) -> Params {
    params_for_scenario_simplepir_with_config(num_items, item_size_bits, YPIRSPConfig::default())
}

pub fn params_for_scenario_simplepir_with_config(
    num_items: u64,
    item_size_bits: u64,
    config: YPIRSPConfig,
) -> Params {
    assert!(num_items > 0, "YPIR-SP requires at least one item");
    assert!(item_size_bits > 0, "YPIR-SP items must not be empty");
    let (poly_len, t_exp_left) = config.validated_values();

    let padded_rows = num_items.next_power_of_two().max(poly_len as u64);
    let bits_per_instance = poly_len as u64 * 14;
    let db_cols = item_size_bits.div_ceil(bits_per_instance) as usize;

    debug!("db_rows: {}, db_cols: {}", padded_rows, db_cols);

    let nu_1 = padded_rows.trailing_zeros() as usize - poly_len.trailing_zeros() as usize;
    debug!("chose nu_1: {}", nu_1);

    let p = 1 << 14;
    let q2_bits = 28;

    let mut params = internal_params_for(nu_1, 1, p, q2_bits, t_exp_left, DEF_MOD_STR, poly_len);
    params.instances = db_cols;
    params
}

pub trait GetQPrime {
    /// The smaller reduced modulus, used on the second row of the encoding
    fn get_q_prime_1(&self) -> u64;

    /// The larger reduced modulus, used on the first row of the encoding
    fn get_q_prime_2(&self) -> u64;
}

impl GetQPrime for Params {
    fn get_q_prime_1(&self) -> u64 {
        1 << 20
    }

    fn get_q_prime_2(&self) -> u64 {
        if self.q2_bits == self.modulus_log2 {
            self.modulus
        } else {
            Q2_VALUES[self.q2_bits as usize]
        }
    }
}

impl GetQPrime for LWEParams {
    fn get_q_prime_1(&self) -> u64 {
        u64::MAX // unsupported
    }

    fn get_q_prime_2(&self) -> u64 {
        if self.q2_bits == (self.modulus as f64).log2().ceil() as usize {
            self.modulus
        } else {
            Q2_VALUES[self.q2_bits as usize]
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct YPIRParams {
    pub is_simplepir: bool,
}

pub trait GetRho {
    fn rho(&self) -> usize;
}

impl GetRho for Params {
    fn rho(&self) -> usize {
        let lwe_params = LWEParams::default();
        let lwe_q_prime_bits = lwe_params.q2_bits as usize;
        let pt_bits = (self.pt_modulus as f64).log2().floor() as usize;
        let blowup_factor = lwe_q_prime_bits as f64 / pt_bits as f64;
        let smaller_params_db_dim_2 = ((blowup_factor * (lwe_params.n + 1) as f64)
            / self.poly_len as f64)
            .log2()
            .ceil() as usize;

        let rho = 1 << smaller_params_db_dim_2;
        rho
    }
}

pub trait GetNumDbItems {
    fn num_db_items(&self, is_simplepir: bool) -> usize;
}

impl GetNumDbItems for Params {
    fn num_db_items(&self, is_simplepir: bool) -> usize {
        if is_simplepir {
            let db_rows = 1 << (self.db_dim_1 + self.poly_len_log2);
            let db_cols = self.instances * self.poly_len;
            db_rows * db_cols
        } else {
            let db_rows = 1 << (self.db_dim_1 + self.poly_len_log2);
            let db_cols = 1 << (self.db_dim_2 + self.poly_len_log2);
            db_rows * db_cols
        }
    }
}

pub trait PtModulusBits {
    fn pt_modulus_bits(&self) -> usize;
}

impl PtModulusBits for Params {
    fn pt_modulus_bits(&self) -> usize {
        (self.pt_modulus as f64).log2().ceil() as usize
    }
}

pub trait DbRowsCols {
    fn db_rows(&self) -> usize;
    fn db_rows_padded_normal(&self) -> usize;
    fn db_rows_padded_simplepir(&self) -> usize;
    fn db_cols_normal(&self) -> usize;
    fn db_cols_simplepir(&self) -> usize;
}

impl DbRowsCols for Params {
    fn db_rows(&self) -> usize {
        let db_rows = 1 << (self.db_dim_1 + self.poly_len_log2);
        db_rows
    }
    fn db_rows_padded_normal(&self) -> usize {
        let db_rows = 1 << (self.db_dim_1 + self.poly_len_log2);
        db_rows + db_rows / 128
    }

    fn db_rows_padded_simplepir(&self) -> usize {
        let db_rows = 1 << (self.db_dim_1 + self.poly_len_log2);
        db_rows
    }

    fn db_cols_normal(&self) -> usize {
        let db_cols = 1 << (self.db_dim_2 + self.poly_len_log2);
        db_cols
    }

    fn db_cols_simplepir(&self) -> usize {
        self.instances * self.poly_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_simplepir_builder_uses_default_config() {
        let legacy = params_for_scenario_simplepir(1 << 14, 1 << 17);
        let explicit = params_for_scenario_simplepir_with_config(
            1 << 14,
            1 << 17,
            YPIRSPConfig::degree_2048(),
        );

        assert_eq!(legacy, explicit);
    }

    #[test]
    fn simplepir_4096_config_derives_geometry() {
        let params =
            params_for_scenario_simplepir_with_config(512, 32_768, YPIRSPConfig::degree_4096());

        assert_eq!(params.poly_len, 4096);
        assert_eq!(params.poly_len_log2, 12);
        assert_eq!(params.t_exp_left, 4);
        assert_eq!(params.db_rows(), 4096);
        assert_eq!(params.instances, 1);
        assert_eq!(params.db_cols_simplepir(), 4096);
    }

    #[test]
    #[should_panic(expected = "YPIR-SP poly_len must be 2048 or 4096")]
    fn unsupported_simplepir_degree_is_rejected() {
        let _ = YPIRSPConfig::for_poly_len(1024);
    }

    #[test]
    #[should_panic(
        expected = "YPIR-SP configuration must be (poly_len=2048, t_exp_left=3) or (poly_len=4096, t_exp_left=4)"
    )]
    fn mismatched_simplepir_degree_and_gadget_digits_are_rejected() {
        let invalid = YPIRSPConfig {
            poly_len: 4096,
            t_exp_left: 3,
        };

        let _ = params_for_scenario_simplepir_with_config(512, 32_768, invalid);
    }
}
