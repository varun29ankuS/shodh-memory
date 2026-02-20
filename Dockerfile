# Multi-stage build for minimal final image
# Stage 1: Build the Rust binary with all native dependencies
FROM rust:1-slim-bookworm AS builder

# Install build dependencies: LLVM for RocksDB bindgen, OpenSSL for TLS
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    llvm-14 \
    libclang-14-dev \
    clang-14 \
    curl \
    cmake \
    && rm -rf /var/lib/apt/lists/*

ENV LIBCLANG_PATH=/usr/lib/llvm-14/lib

WORKDIR /app

# Copy manifests first for Docker layer caching
COPY Cargo.toml Cargo.lock ./

# Copy source and all paths referenced by Cargo.toml (benches, tests)
COPY src ./src
COPY benches ./benches
COPY tests ./tests

# Download ONNX Runtime for embedding model inference
ARG ORT_VERSION=1.23.2
RUN curl -L -o ort.tgz "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
    && tar -xzf ort.tgz \
    && cp onnxruntime-linux-x64-${ORT_VERSION}/lib/libonnxruntime.so.${ORT_VERSION} /usr/local/lib/libonnxruntime.so \
    && ldconfig \
    && rm -rf ort.tgz onnxruntime-linux-x64-${ORT_VERSION}

ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so

# Build release binary
RUN cargo build --release

# Stage 2: Minimal runtime image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 shodh && \
    mkdir -p /data && \
    chown -R shodh:shodh /data

# Copy binary and ONNX Runtime from builder
COPY --from=builder /app/target/release/shodh-memory-server /usr/local/bin/shodh-memory
COPY --from=builder /usr/local/lib/libonnxruntime.so /usr/local/lib/libonnxruntime.so
RUN ldconfig

# Switch to non-root user
USER shodh

# Set working directory
WORKDIR /data

# Expose default port
EXPOSE 3030

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:3030/health || exit 1

# Set environment variables
ENV RUST_LOG=info \
    SHODH_HOST=0.0.0.0 \
    SHODH_PORT=3030 \
    SHODH_MEMORY_PATH=/data \
    LD_LIBRARY_PATH=/usr/local/lib

# Run the binary
CMD ["shodh-memory"]
