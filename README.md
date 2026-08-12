# YPIR

This is a fork of the [YPIR](https://github.com/menonsamir/ypir) implementation of the YPIR scheme for single-server private information retrieval,
introduced in ["YPIR: High-Throughput Single-Server PIR with Silent Preprocessing"](https://eprint.iacr.org/2024/270).

This fork has been **audited by [Zellic](https://zellic.io)**. The audit report is available in [`audits/zellic-audit-report.pdf`](audits/zellic-audit-report.pdf).

**Client-side code is considered frozen** in this repository. Server-side code remains open to changes. This is because these changes can only affect performance, they cannot break client privacy. A server-side change could break integrity, as could a malicious server. However, all authentication of data retrieved is not done at the cryptographic layer in YPIR, but instead is an application-layer concern. In our usages within voting and spendability, authentication is explicitly addressed (via merkle path authentication checks against a trusted merkle root, or recursive proofs post-Tachyon).

## Running

To build and run this code:
1. Ensure you are running on Ubuntu (at least 22.04), and that AVX-512 is available on the CPU (you can run `lscpu` and look for the `avx512f` flag).
Our benchmarks were collected using the AWS `r6i.16xlarge` instance type, which has all necessary CPU features.
2. Run `sudo apt-get update && sudo apt-get install -y build-essential libssl-dev pkg-config`.
3. [Install Rust using rustup](https://www.rust-lang.org/tools/install) using `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`.
  - Select `1) Proceed with installation (default)` when prompted
  - After installation, configure the current shell as instructed by running `source "$HOME/.cargo/env"`
4. Run `git clone https://github.com/valargroup/ypir.git` and `cd ypir`.
5. Run `cargo run --release -- 1073741824` to run YPIR on a random database consisting of 1073741824 bits (~134 MB).
The first time you run this command, Cargo will download and install the necessary libraries to build the code (~2 minutes);
later calls will not take as long. Stability warnings can be safely ignored. 
See below for details on how to interpret the measurements.

We have tested the above steps on a fresh AWS `r6i.16xlarge` Ubuntu 22.04 instance and confirmed they work.

### Options
To pass arguments, make sure to run `cargo run --release -- <ARGS>` (the ` -- ` is important).
Passing `--verbose` or setting the environment variable `RUST_LOG=debug`
will enable detailed logging. All PIR results are checked for correctness.
The full command-line parameters are as follows:

```
Usage: cargo run --release -- [OPTIONS] <NUM_ITEMS> [ITEM_SIZE_BITS] [NUM_CLIENTS] [TRIALS] [OUT_REPORT_JSON]

Arguments:
  <NUM_ITEMS>        Number of items in the database
  [ITEM_SIZE_BITS]   Size of each item in bits (optional, default 1), values over 8 are unsupported
  [NUM_CLIENTS]      Number of clients (optional, default 1) to perform cross-client batching over
  [TRIALS]           Number of trials (optional, default 5) to run the YPIR scheme 
                     and average performance measurements over (with one additional warmup trial excluded)
  [OUT_REPORT_JSON]  Output report file (optional) where results will be written in JSON

Options:
  -v, --verbose  Verbose mode (optional) if set, the program will print debug logs to stderr
  -h, --help     Print help
  -V, --version  Print version
```

### YPIR-SP ring dimension

YPIR-SP defaults to the audited 2048-degree parameter set. The experimental
4096-degree set can be selected with `--poly-len 4096` on the `run`, `client`,
and `server` binaries. Clients and servers must select the same degree.

Library callers can select it explicitly:

```rust
use valar_ypir::{client::YPIRClient, params::YPIRSPConfig};

let client = YPIRClient::from_db_sz_simplepir_with_config(
    num_items,
    item_size_bits,
    YPIRSPConfig::degree_4096(),
);
```

#### Why four gadget digits

The gadget base is derived from the digit count, not chosen independently
(`⌊56/t⌋ + 1` bits), so `t_exp_left` is the only dial. Doubling the ring degree
makes the packing term ~8x noisier while the decoding window (set by `p` and
`q'_1`) does not move, so the audited `t = 3` misses by a wide margin at 4096 —
a modelled failure probability around `2^-14`. Four digits shrink each digit
from 19 to 15 bits, cutting that term ~192x, which over-pays the 8x. Five
digits would buy nothing: the packing term is then already below the
modulus-switch floor, and each extra digit costs ~25% more packing-key upload.

Only `(2048, 3)` and `(4096, 4)` are accepted. `YPIRSPConfig`'s fields are
private so no other pair is constructible, and `assert_valid_ypir_sp_params`
re-checks the pair wherever a bare `&Params` enters the YPIR-SP path, since
`Params` exposes `poly_len` and `t_exp_left` publicly.

#### Inspecting the noise bound

```sh
cargo run --features cli --bin analyze-sp -- \
  <NUM_ITEMS> <ITEM_SIZE_BITS> --poly-len 4096
```

`ypir_sp_noise_report` bounds the YPIR-SP path term by term: the two
modulus-switch contributions, the SimplePIR first dimension (the only
`db_rows`-dependent term), and automorphism packing. It carries an explicit
`PACKING_TERM_SLACK`, because composing the per-automorphism bound across
`log2(poly_len)` levels is a heuristic that measurement puts ~1.85x low.
`noise_bound_dominates_measurement` in `scheme.rs` runs the real pipeline over
a range of shapes and fails if measured noise ever crosses the bound, so the
slack cannot silently rot.

For 16,384 rows of 131,072-bit items, the model reports both the tail bound for
one coefficient and a response-wide bound obtained by union-bounding over
`poly_len * instances` decoded coefficients. The response-wide value is the
one compared with the `2^-40` correctness target:

| set | per-coefficient failure | response failure | worst coefficient |
| --- | --- | --- | --- |
| 2048, t=3 | `2^-57.40` | `2^-44.08` | 20–30% of window |
| 4096, t=4 | `2^-586.88` | `2^-573.29` | 8–10% of window |

The 4096 set is the better-balanced of the two: at 2048 the packing term
dominates the modulus-switch term ~18:1, so it sits well above its own floor,
whereas 4096 is switch-limited and therefore close to the floor the wire format
allows.

Separately, aligning the model with the production gadget base (`2^19` for
three digits) changes the 2048-degree **YPIR-double** model from approximately
`2^-41.75` to `2^-26.74` total failure probability (`2^-96.70` for the
SimplePIR stage and `2^-26.74` for the double-PIR stage). That is below the
previous `2^-40` target and is tracked by a characterization test. It does not
affect the YPIR-SP bounds, and the double path is unreachable from the shipped
binaries, which require `--is-simplepir`.

### Interpreting measurements
This is an annotated version of the output
of running `RUST_LOG=debug cargo run --profile release-with-debug --bin server 8589934592 1` 
(testing on a 1 GB database),
detailing what each measurement means:
```js
{
  "offline": {
    // Bytes uploaded by the client in the offline phase
    "uploadBytes": 0,

    // Bytes downloaded by the client in the offline phase
    "downloadBytes": 0,
    
    // Server computation time, in milliseconds, in the offline phase. 
    // Includes any precomputation that must be performed on the plaintext database.
    "serverTimeMs": 3965,
    
    // Not used.
    "clientTimeMs": 0,
    
    // Time spent precomputing just the SimplePIR hint.
    "simplepirPrepTimeMs": 2539,

    // Bytes that the client *would* have to download, in the offline phase,
    // if they were performing SimplePIR (rather than YPIR) 
    // using this implementation (SimplePIR* in the paper).
    "simplepirHintBytes": 29360128,

    // Similarly, bytes that the client *would* have to download, 
    // in the offline phase DoublePIR (DoublePIR* in the paper).
    "doublepirHintBytes": 14680064
  },
  "online": {
    // Bytes uploaded by a single client in the online phase.
    "uploadBytes": 604160,
    
    // Bytes downloaded by a single client in the online phase.
    "downloadBytes": 12288,

    // Bytes that the client *would* have to download, in the online phase,
    // if they were performing SimplePIR (SimplePIR* in the paper).
    "simplepirRespBytes": 28672,

    // Bytes that the client *would* have to download, in the online phase,
    // if they were performing DoublePIR (DoublePIR* in the paper).
    "doublepirRespBytes": 12288,

    // Server computation time, in milliseconds, in the online phase.
    // This is the average time over 5 trials, after a warmup trial.
    "serverTimeMs": 402,

    // Time that the client took to generate the query.
    "clientQueryGenTimeMs": 530,

    // Time that the client took to decode the response (may round down to 0ms).
    "clientDecodeTimeMs": 0,

    // Time spent in the first pass of YPIR (the 'SimplePIR' phase)
    "firstPassTimeMs": 9,

    // Time spent in the second pass of YPIR (the 'DoublePIR' phase)
    "secondPassTimeMs": 3,

    // Time spent performing LWE-to-RLWE conversion
    "ringPackingTimeMs": 387,

    // Not used.
    "sqrtNBytes": 8192,

    // The full set of measured server computation times.
    "allServerTimesMs": [
      401,
      403,
      402,
      401,
      401
    ],
    // The standard deviation of the measured server computation times.
    "stdDevServerTimeMs": 0.8
  }
}
```

## Server & Client

You can run YPIR as a standalone HTTP server using a command like:


```sh
$ RUST_LOG=debug cargo run --profile release-with-debug --features http_server --bin server 32768 262144 --is-simplepir --inp-file ../passwords-data/hibp-passwords.bin -p 8989 --hint-file ../passwords-data/hibp-passwords-2-hint.bin
```


## Acknowledgements

YPIR is based on [DoublePIR](https://eprint.iacr.org/2022/949), and this implementation
uses matrix-vector multiplication routines based on the ones in [ahenzinger/simplepir](https://github.com/ahenzinger/simplepir).
We also use a [fork of spiral-rs](https://github.com/valargroup/spiral-rs) for Spiral to handle RLWE ciphertexts.

## Citing

Please cite the original work as:

```
@inproceedings{MW24,
  author    = {Samir Jordan Menon and David J. Wu},
  title     = {{YPIR}: High-Throughput Single-Server {PIR} with Silent Preprocessing},
  booktitle = {{USENIX} Security Symposium},
  year      = {2024}
}
```

This fork is maintained by [valargroup](https://github.com/valargroup).