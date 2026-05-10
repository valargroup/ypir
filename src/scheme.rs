use std::time::Instant;

use log::debug;
use rand::{thread_rng, Rng};

use spiral_rs::aligned_memory::AlignedMemory64;
use spiral_rs::params::*;

use super::{client::*, measurement::*, params::*, server::*};

pub fn run_ypir_batched(num_items: usize, item_size_bits: usize, trials: usize) -> Measurement {
    let params = params_for_scenario_simplepir(num_items as u64, item_size_bits as u64);
    let measurement = run_simple_ypir_on_params::<1>(params, trials);
    debug!("{:#?}", measurement);
    measurement
}

pub trait Sample {
    fn sample() -> Self;
}

impl Sample for u8 {
    fn sample() -> Self {
        fastrand::u8(..)
    }
}

impl Sample for u16 {
    fn sample() -> Self {
        fastrand::u16(..)
    }
}

pub fn run_simple_ypir_on_params<const K: usize>(params: Params, trials: usize) -> Measurement {
    assert_eq!(K, 1); // for now

    let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
    let db_rows_padded = params.db_rows_padded_simplepir();
    let db_cols = params.instances * params.poly_len;

    let mut rng = thread_rng();

    // --

    let now = Instant::now();
    type T = u16;
    let pt_iter = std::iter::repeat_with(|| (T::sample() as u64 % params.pt_modulus) as T);
    let y_server = YServer::<T>::new(&params, pt_iter, false, true);
    debug!("Created server in {} us", now.elapsed().as_micros());
    debug!(
        "Database of {} bytes",
        y_server.db().len() * (params.pt_modulus as f64).log2().ceil() as usize / 8
    );
    // assert_eq!(
    //     y_server.db().len() * std::mem::size_of::<T>(),
    //     db_rows_padded * db_cols * (params.pt_modulus as f64).log2().ceil() as usize / 8
    // );
    assert_eq!(y_server.db().len(), db_rows_padded * db_cols);

    // ================================================================
    // OFFLINE PHASE
    // ================================================================
    let mut measurements = vec![Measurement::default(); trials + 1];

    let start_offline_comp = Instant::now();
    let offline_values =
        y_server.perform_offline_precomputation_simplepir(Some(&mut measurements[0]), None, None);
    let offline_server_time_ms = start_offline_comp.elapsed().as_millis();

    let packed_query_row_sz = params.db_rows_padded_simplepir();
    // let mut all_queries_packed = AlignedMemory64::new(K * packed_query_row_sz);

    for trial in 0..trials + 1 {
        debug!("trial: {}", trial);
        let mut measurement = &mut measurements[trial];
        measurement.offline.server_time_ms = offline_server_time_ms as usize;

        // ================================================================
        // QUERY GENERATION PHASE
        // ================================================================
        let mut online_upload_bytes = 0;
        let mut queries = Vec::new();

        let ypir_client = YPIRClient::new(&params);

        for _batch in 0..K {
            let target_idx: usize = rng.gen::<usize>() % (db_rows * db_cols);
            let target_row = target_idx / db_cols;
            let target_col = target_idx % db_cols;
            debug!(
                "Target item: {} ({}, {})",
                target_idx, target_row, target_col
            );

            let start = Instant::now();
            let ((packed_query_row, pack_pub_params_row_1s), client_seed) =
                ypir_client.generate_query_simplepir(target_row);

            let query_size = ((packed_query_row.len() as f64 * params.modulus_log2 as f64) / 8.0)
                .ceil() as usize;
            let pub_params_size = pack_pub_params_row_1s.len() * params.modulus_log2 as usize / 8;
            let total_query_size = query_size + pub_params_size;

            measurement.online.client_query_gen_time_ms = start.elapsed().as_millis() as usize;
            debug!("Generated query in {} us", start.elapsed().as_micros());

            online_upload_bytes = total_query_size;
            debug!("Query size: {} bytes", online_upload_bytes);

            queries.push((
                client_seed,
                target_idx,
                packed_query_row,
                pack_pub_params_row_1s,
            ));
        }

        let mut all_queries_packed = AlignedMemory64::new(K * packed_query_row_sz);
        for (i, chunk_mut) in all_queries_packed
            .as_mut_slice()
            .chunks_mut(packed_query_row_sz)
            .enumerate()
        {
            (&mut chunk_mut[..db_rows]).copy_from_slice(queries[i].2.as_slice());
        }

        let offline_values = offline_values.clone();

        // ================================================================
        // ONLINE PHASE
        // ================================================================

        let start_online_comp = Instant::now();
        let responses = vec![y_server.perform_online_computation_simplepir(
            all_queries_packed.as_slice(),
            &offline_values,
            &queries.iter().map(|x| x.3.as_slice()).collect::<Vec<_>>(),
            Some(&mut measurement),
        )];
        let online_server_time_ms = start_online_comp.elapsed().as_millis();
        let online_download_bytes = get_size_bytes(&[responses.clone()]); // TODO: this is not quite right for multiple clients

        // check correctness
        for (response_switched, (client_seed, target_idx, _, _)) in
            responses.iter().zip(queries.iter())
        {
            let (target_row, _target_col) = (target_idx / db_cols, target_idx % db_cols);
            let corr_result = y_server
                .get_row(target_row)
                .iter()
                .map(|x| x.to_u64())
                .collect::<Vec<_>>();

            // let scheme_params = YPIRSchemeParams::from_params(&params, &LWEParams::default());
            // let log2_corr_err = scheme_params.delta().log2();
            // let log2_expected_outer_noise = scheme_params.expected_outer_noise().log2();
            // debug!("log2_correctness_err: {}", log2_corr_err);
            // debug!("log2_expected_outer_noise: {}", log2_expected_outer_noise);

            let start_decode = Instant::now();
            let final_result =
                ypir_client.decode_response_simplepir_raw(*client_seed, response_switched);
            measurement.online.client_decode_time_ms = start_decode.elapsed().as_millis() as usize;

            // debug!("got      {:?}", &final_result[..256]);
            // debug!("expected {:?}", &corr_result[..256]);
            // debug!("got {:?}, expected {:?}", &final_result[..256], &corr_result[..256]);
            assert_eq!(final_result, corr_result);
        }

        measurement.online.upload_bytes = online_upload_bytes;
        measurement.online.download_bytes = online_download_bytes;
        measurement.online.server_time_ms = online_server_time_ms as usize;
    }

    // discard the first measurement (if there were multiple trials)
    // copy offline values from the first measurement to the second measurement
    if trials > 1 {
        measurements[1].offline = measurements[0].offline.clone();
        measurements.remove(0);
    }

    let mut final_measurement = measurements[0].clone();
    final_measurement.online.server_time_ms = mean(
        &measurements
            .iter()
            .map(|m| m.online.server_time_ms)
            .collect::<Vec<_>>(),
    )
    .round() as usize;
    final_measurement.online.all_server_times_ms = measurements
        .iter()
        .map(|m| m.online.server_time_ms)
        .collect::<Vec<_>>();
    final_measurement.online.std_dev_server_time_ms =
        std_dev(&final_measurement.online.all_server_times_ms);

    final_measurement
}

fn mean(xs: &[usize]) -> f64 {
    xs.iter().map(|x| *x as f64).sum::<f64>() / xs.len() as f64
}

fn std_dev(xs: &[usize]) -> f64 {
    let mean = mean(xs);
    let mut variance = 0.;
    for x in xs {
        variance += (*x as f64 - mean).powi(2);
    }
    (variance / xs.len() as f64).sqrt()
}

#[cfg(test)]
mod test {
    use crate::{bits::u64s_to_contiguous_bytes, serialize::ToBytes};

    use super::*;
    use test_log::test;

    #[test]
    #[ignore = "long test: full SP benchmark path"]
    fn test_ypir_basic() {
        run_ypir_batched(1 << 14, 16384 * 8, 1);
    }

    #[test]
    #[ignore = "long test: full SP client/server round trip"]
    fn test_ypirclient_simplepir() {
        let params = params_for_scenario_simplepir(1 << 14, 16384 * 8);
        let pt_iter = std::iter::repeat_with(|| (u16::sample() as u64 % params.pt_modulus) as u16);
        let y_server = YServer::<u16>::new(&params, pt_iter, false, true);
        let mut offline_values =
            y_server.perform_offline_precomputation_simplepir(None, None, None);

        let target_row = fastrand::usize(..params.db_rows());

        let client = YPIRClient::from_db_sz(1 << 14, 16384 * 8);
        let (query, client_seed) = client.generate_query_simplepir(target_row);
        let query_bytes = query.to_bytes();
        let response =
            y_server.perform_full_online_computation_simplepir(&mut offline_values, &query_bytes);

        let decoded = client.decode_response_simplepir(client_seed, &response);
        let corr_result = y_server
            .get_row(target_row)
            .iter()
            .map(|x| x.to_u64())
            .collect::<Vec<_>>();
        let corr_result_bytes = u64s_to_contiguous_bytes(&corr_result, params.pt_modulus_bits());

        assert_eq!(decoded, corr_result_bytes);
    }

    #[test]
    #[ignore]
    #[cfg(feature = "huge_tests")]
    fn test_ypir_simplepir_rectangle() {
        run_ypir_batched(1 << 16, 16384 * 8, 1);
    }

    #[test]
    #[ignore]
    #[cfg(feature = "huge_tests")]
    fn test_ypir_simplepir_rectangle_8gb() {
        run_ypir_batched(1 << 17, 65536 * 8, 1);
    }

    #[cfg(feature = "test_data")]
    #[test]
    #[ignore]
    fn test_with_test_data() {
        use crate::data::{RESP1, SEED1};

        let client = YPIRClient::from_db_sz(1 << 14, 16384 * 8);
        let decoded = client.decode_response_simplepir(SEED1, RESP1);
        println!("raw:  {:?}", &decoded[..32]);
        let bytes = u64s_to_contiguous_bytes(&decoded, client.params().pt_modulus_bits());
        println!("as u8: {:?}", &bytes[..32]);
    }

    // ── Precompute-cache round-trip + truncation fuzz ────────────────────────

    /// Build a small SimplePIR YServer + OfflinePrecomputedValues and a
    /// reference response for a fixed query. Returns
    /// (params, server_dump, offline_dump, query_bytes, reference_response).
    /// Used by the round-trip and truncation-fuzz tests.
    fn build_cache_fixture() -> (Params, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, usize) {
        // Same scenario as test_ypirclient_simplepir; small but real.
        let num_items: u64 = 1 << 14;
        let item_size_bits: u64 = 16384 * 8;
        let params = params_for_scenario_simplepir(num_items, item_size_bits);

        // Deterministic plaintext (so both fresh and round-tripped servers see
        // the same database).
        fastrand::seed(253);
        let db_size = (params.db_rows() * params.db_cols_simplepir()) as usize;
        let pt_data: Vec<u16> = (0..db_size)
            .map(|_| (fastrand::u64(..) % params.pt_modulus) as u16)
            .collect();

        let y_server = YServer::<u16>::new(&params, pt_data.iter().copied(), false, true);
        let mut offline_values =
            y_server.perform_offline_precomputation_simplepir(None, None, None);

        // Fixed query against the freshly-built server (reference response).
        let target_row = 7usize;
        let client = YPIRClient::from_db_sz(num_items, item_size_bits);
        let (query, _client_seed) = client.generate_query_simplepir(target_row);
        let query_bytes = query.to_bytes();
        let reference_response =
            y_server.perform_full_online_computation_simplepir(&mut offline_values, &query_bytes);

        // Dump the server and offline values via the new cache I/O API.
        let mut server_dump = Vec::new();
        y_server.dump_into(&mut server_dump).unwrap();
        let mut offline_dump = Vec::new();
        offline_values.dump_into(&mut offline_dump).unwrap();

        (
            params,
            server_dump,
            offline_dump,
            query_bytes,
            reference_response,
            target_row,
        )
    }

    #[test]
    #[ignore = "long test: builds full SP cache fixture"]
    fn test_cache_round_trip_byte_equal() {
        let (params, server_dump, offline_dump, query_bytes, reference_response, _) =
            build_cache_fixture();

        // Load both back from the dumps.
        let mut sr = std::io::Cursor::new(&server_dump);
        let server2 = YServer::<u16>::load_from(&mut sr, &params).expect("load YServer");
        let mut or = std::io::Cursor::new(&offline_dump);
        let mut offline2 = crate::serialize::OfflinePrecomputedValues::load_from(&mut or, &params)
            .expect("load OfflinePrecomputedValues");

        // Run the same query against the round-tripped server. Response bytes
        // must be identical to the reference. This is THE correctness check.
        let response2 =
            server2.perform_full_online_computation_simplepir(&mut offline2, &query_bytes);
        assert_eq!(
            reference_response, response2,
            "round-tripped server gave different response"
        );
    }

    fn truncation_lengths(dump_len: usize) -> Vec<usize> {
        let mut lengths: Vec<usize> = (0..64).collect();
        lengths.extend([
            dump_len / 4,
            dump_len / 2,
            dump_len.saturating_sub(64),
            dump_len.saturating_sub(8),
            dump_len.saturating_sub(1),
        ]);
        lengths.into_iter().filter(|&n| n < dump_len).collect()
    }

    #[test]
    #[ignore = "long test: builds full SP cache fixture"]
    fn test_yserver_load_rejects_truncation() {
        let (params, server_dump, _, _, _, _) = build_cache_fixture();
        for len in truncation_lengths(server_dump.len()) {
            let mut r = std::io::Cursor::new(&server_dump[..len]);
            let res = YServer::<u16>::load_from(&mut r, &params);
            assert!(
                res.is_err(),
                "YServer::load_from accepted truncated dump at len={len}"
            );
        }
    }

    #[test]
    #[ignore = "long test: builds full SP cache fixture"]
    fn test_offline_load_rejects_truncation() {
        let (params, _, offline_dump, _, _, _) = build_cache_fixture();
        for len in truncation_lengths(offline_dump.len()) {
            let mut r = std::io::Cursor::new(&offline_dump[..len]);
            let res = crate::serialize::OfflinePrecomputedValues::load_from(&mut r, &params);
            assert!(
                res.is_err(),
                "OfflinePrecomputedValues::load_from accepted truncated dump at len={len}"
            );
        }
    }

    /// Mutating the first byte (version) of either dump must be rejected.
    /// Mutating the second byte (flags) MAY be rejected (depends on whether
    /// the flipped bits are inside KNOWN_FLAGS or set the required bits).
    /// We pick byte 0 here because rejection is unambiguous.
    #[test]
    #[ignore = "long test: builds full SP cache fixture"]
    fn test_yserver_load_rejects_version_mutation() {
        let (params, server_dump, _, _, _, _) = build_cache_fixture();
        // Version is the first byte; flip every non-zero u8 value.
        for new_v in 1u8..=255 {
            if new_v == server_dump[0] {
                continue;
            }
            let mut mutated = server_dump.clone();
            mutated[0] = new_v;
            let mut r = std::io::Cursor::new(&mutated);
            let res = YServer::<u16>::load_from(&mut r, &params);
            assert!(
                res.is_err(),
                "YServer::load_from accepted unknown version byte 0x{new_v:02x}"
            );
        }
    }

    #[test]
    #[ignore = "long test: builds full SP cache fixture"]
    fn test_offline_load_rejects_version_mutation() {
        let (params, _, offline_dump, _, _, _) = build_cache_fixture();
        for new_v in 1u8..=255 {
            if new_v == offline_dump[0] {
                continue;
            }
            let mut mutated = offline_dump.clone();
            mutated[0] = new_v;
            let mut r = std::io::Cursor::new(&mutated);
            let res = crate::serialize::OfflinePrecomputedValues::load_from(&mut r, &params);
            assert!(
                res.is_err(),
                "OfflinePrecomputedValues::load_from accepted unknown version byte 0x{new_v:02x}"
            );
        }
    }

    /// Setting a reserved flag bit (bits 2..=7 for YServer, bits 1..=7 for
    /// OfflinePrecomputedValues) must be rejected so a future format that
    /// assigns them isn't silently mis-loaded by older code.
    #[test]
    #[ignore = "long test: builds full SP cache fixture"]
    fn test_yserver_load_rejects_unknown_flag_bits() {
        let (params, server_dump, _, _, _, _) = build_cache_fixture();
        for bit in 2..=7u8 {
            let mut mutated = server_dump.clone();
            mutated[1] |= 1 << bit;
            let mut r = std::io::Cursor::new(&mutated);
            let res = YServer::<u16>::load_from(&mut r, &params);
            assert!(
                res.is_err(),
                "YServer::load_from accepted reserved flag bit {bit}"
            );
        }
    }

    #[test]
    #[ignore = "long test: builds full SP cache fixture"]
    fn test_offline_load_rejects_unknown_flag_bits() {
        let (params, _, offline_dump, _, _, _) = build_cache_fixture();
        for bit in 1..=7u8 {
            let mut mutated = offline_dump.clone();
            mutated[1] |= 1 << bit;
            let mut r = std::io::Cursor::new(&mutated);
            let res = crate::serialize::OfflinePrecomputedValues::load_from(&mut r, &params);
            assert!(
                res.is_err(),
                "OfflinePrecomputedValues::load_from accepted reserved flag bit {bit}"
            );
        }
    }

    /// The shape validator should reject a dump whose hint_0 length is a
    /// normal-sized but wrong value (`expected + 1`). This proves the
    /// `_exact` length-checked loaders catch wrong shapes specifically,
    /// not just absurdly-large length prefixes that hit `MAX_LEN_PREFIX`.
    #[test]
    #[ignore = "long test: builds full SP cache fixture"]
    fn test_offline_load_rejects_normal_size_wrong_hint_0_length() {
        let (params, _, mut offline_dump, _, _, _) = build_cache_fixture();
        // Layout of an offline dump: payload_version (u8) + flags (u8) +
        // hint_0_len (u64 LE) + hint_0 data + ...
        // hint_0_len lives at bytes 2..10. Compute the expected value from
        // params, write `expected + 1` into the slot, and verify rejection.
        let db_cols = params.instances * params.poly_len;
        let expected = (params.poly_len * db_cols) as u64;
        let bogus = expected + 1;
        offline_dump[2..10].copy_from_slice(&bogus.to_le_bytes());
        let mut r = std::io::Cursor::new(&offline_dump);
        let err = crate::serialize::OfflinePrecomputedValues::load_from(&mut r, &params)
            .err()
            .expect("load should reject wrong hint_0 length");
        match err {
            crate::serialize::CacheError::Malformed { what, detail } => {
                assert!(
                    what == "hint_0"
                        && detail.contains(&format!("{bogus}"))
                        && detail.contains(&format!("{expected}")),
                    "expected hint_0 length-mismatch error, got {what}: {detail}"
                );
            }
            other => {
                panic!("expected CacheError::Malformed for wrong hint_0 length, got {other:?}")
            }
        }
    }

    /// Same shape: `expected - 1` (one short). The wrong-shape catch must
    /// fire on the LOW side too, not just the high side.
    #[test]
    #[ignore = "long test: builds full SP cache fixture"]
    fn test_offline_load_rejects_normal_size_short_hint_0_length() {
        let (params, _, mut offline_dump, _, _, _) = build_cache_fixture();
        let db_cols = params.instances * params.poly_len;
        let expected = (params.poly_len * db_cols) as u64;
        let bogus = expected - 1;
        offline_dump[2..10].copy_from_slice(&bogus.to_le_bytes());
        let mut r = std::io::Cursor::new(&offline_dump);
        let err = crate::serialize::OfflinePrecomputedValues::load_from(&mut r, &params)
            .err()
            .expect("load should reject short hint_0 length");
        assert!(
            matches!(
                err,
                crate::serialize::CacheError::Malformed { what: "hint_0", .. }
            ),
            "expected hint_0 length-mismatch error, got {err:?}"
        );
    }
}
