FROM rust:1.97.1-bookworm@sha256:0e2bcaef56d041a486784e54104a81aebe0da44bd03019bd70bc0401e42e4a97 AS builder
RUN apt-get update && apt-get install -y clang-14 clang++-14
WORKDIR /usr/src/ypir
COPY . .
# The repository's Rust 1.85.1 default predates stable AVX-512 intrinsics.
RUN CC=clang-14 CXX=clang++-14 cargo +1.97.1 install --locked --features explicit_avx512,cli --bin run --path .

FROM debian:bookworm-slim@sha256:7ff6ba31527b9e336162fa061746c366c4b2032f4d2cff2078bea43f78efe3f8
RUN apt-get update && apt-get install -y clang-14 clang++-14 libssl-dev pkg-config && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/cargo/bin/run /usr/local/bin/run
ENV RUST_LOG=debug
ENTRYPOINT ["run"]
