use clap::Parser;
use serde::Serialize;
use valar_ypir::{
    noise_analysis::{ypir_sp_noise_report, YPIRSPNoiseReport},
    params::{params_for_scenario_simplepir_with_config, GetQPrime, YPIRSPConfig},
};

#[derive(Parser, Debug)]
#[command(version, about = "Analyze a YPIR-SP parameter set")]
struct Args {
    /// Number of logical database rows
    num_items: u64,
    /// Size of each row in bits
    item_size_bits: u64,
    /// RLWE polynomial degree (2048 or 4096)
    #[arg(long, default_value_t = 4096)]
    poly_len: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Report {
    poly_len: usize,
    modulus: u64,
    modulus_bits: u64,
    noise_width: f64,
    noise_standard_deviation: f64,
    plaintext_modulus: u64,
    reduced_modulus_small: u64,
    reduced_modulus_large: u64,
    noise: YPIRSPNoiseReport,
}

fn main() {
    let args = Args::parse();
    let config = YPIRSPConfig::for_poly_len(args.poly_len);
    let params =
        params_for_scenario_simplepir_with_config(args.num_items, args.item_size_bits, config);
    let report = Report {
        poly_len: params.poly_len,
        modulus: params.modulus,
        modulus_bits: params.modulus_log2,
        noise_width: params.noise_width,
        noise_standard_deviation: params.noise_width / (2.0 * std::f64::consts::PI).sqrt(),
        plaintext_modulus: params.pt_modulus,
        reduced_modulus_small: params.get_q_prime_1(),
        reduced_modulus_large: params.get_q_prime_2(),
        noise: ypir_sp_noise_report(&params),
    };

    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}
