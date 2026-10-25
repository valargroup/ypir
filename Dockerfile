FROM rust:1-bookworm@sha256:48b7f435b8fc6378ce43b44f15e3a47fd8919b2825f835c39bd4bd3da50af5b8 as builder
RUN apt-get update && apt-get install -y clang-14 clang++-14
WORKDIR /usr/src/ypir
COPY . .
RUN CC=clang-14 CXX=clang++-14 cargo install --locked --features explicit_avx512 --bin run --path .

FROM debian:bookworm-slim@sha256:7ff6ba31527b9e336162fa061746c366c4b2032f4d2cff2078bea43f78efe3f8
RUN apt-get update && apt-get install -y clang-14 clang++-14 libssl-dev pkg-config && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/cargo/bin/run /usr/local/bin/run
ENV RUST_LOG=debug
ENTRYPOINT ["run"]