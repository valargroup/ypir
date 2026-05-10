# Evaluation

This is a step-by-step guide on how to run YPIR to evaluate its key results.

## Getting Started
We recommend running these instructions in an Ubuntu (or similar Linux) environment.

1. Download and install [Docker](https://docs.docker.com/engine/install/ubuntu/)
2. Run `sudo docker run --security-opt seccomp:unconfined --cpus=1 ghcr.io/menonsamir/ypir 32768 131072`
  - The `seccomp:unconfined` option disables container sandboxing that Docker runs by default which can degrade performance
  - The `--cpus=1` option runs a single-threaded container (this is not crucial - the implementation will always use only 1 thread)
  - The `32768 131072` arguments indicate running YPIR-SP on 32768 items, each 131072 bits


## Building
To build from source, just run `cargo build --profile release-with-debug`. To run a basic test, run `cargo run --profile release-with-debug --bin run 32768 131072 --verbose`
