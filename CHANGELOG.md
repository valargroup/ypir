# Changelog

## 0.1.4 - 2026-04-27

### Added

- Add K-generic SimplePIR server batching with `perform_online_computation_simplepir_batched`.
- Add K=5 kernel support and tests across scalar, Rayon, and explicit AVX-512 paths.
- Add K=5 SimplePIR server tests for byte-identical sequential parity and end-to-end decode correctness.
