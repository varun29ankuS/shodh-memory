# Multi-stage build for minimal final image
# Stage 1: Build the Rust binary with all native dependencies
FROM rust:1-slim-bookworm AS builder

# Install build dependencies:
#   g++ / clang-14  — C++ compiler for RocksDB (cc-rs looks for "c++")
#   llvm-14         — libclang for bindgen
#   cmake / make    — native dep build systems
#   libssl-dev      — OpenSSL headers for TLS
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    llvm-14 \
    libclang-14-dev \
    clang-14 \
    g++ \
    make \
    curl \
    cmake \
    && rustup component add rustfmt \
    && rm -rf /var/lib/apt/lists/*

ENV LIBCLANG_PATH=/usr/lib/llvm-14/lib

WORKDIR /app

# Copy manifests first for Docker layer caching
COPY Cargo.toml Cargo.lock ./

# Download ONNX Runtime for embedding model inference (cacheable independent of source)
ARG ORT_VERSION=1.23.2
RUN curl -L -o ort.tgz "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
    && tar -xzf ort.tgz \
    && cp onnxruntime-linux-x64-${ORT_VERSION}/lib/libonnxruntime.so.${ORT_VERSION} /usr/local/lib/libonnxruntime.so \
    && ldconfig \
    && rm -rf ort.tgz onnxruntime-linux-x64-${ORT_VERSION}

ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so

# Download MiniLM-L6-v2 model files for semantic embeddings (cacheable independent of source)
# Pinned to commit c9745ed1 for reproducibility (~23MB quantized model + ~700KB tokenizer)
RUN mkdir -p /models/minilm-l6 \
    && curl -L -o /models/minilm-l6/model_quantized.onnx \
       "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/onnx/model_quint8_avx2.onnx" \
    && curl -L -o /models/minilm-l6/tokenizer.json \
       "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/tokenizer.json"

# Create dummy sources to build and cache dependencies
RUN mkdir -p src benches tests \
    && echo "fn main() {}" > src/main.rs \
    && echo "fn main() {}" > src/cli.rs \
    && touch src/lib.rs \
    && awk -F '"' '/name = ".*benchmarks"/ {print "echo \"fn main() {}\" > benches/" $2 ".rs"}' Cargo.toml | sh \
    && touch tests/dummy.rs \
    && cargo build --release \
    && rm -rf src benches tests

# Copy source and all paths referenced by Cargo.toml (benches, tests)
COPY src ./src
COPY benches ./benches
COPY tests ./tests

# Update modification time to ensure Cargo rebuilds the application
RUN touch src/main.rs src/cli.rs src/lib.rs

# Build release binary
RUN cargo build --release

# Stage 2: Minimal runtime image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    curl \
    wamerican \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 shodh && \
    mkdir -p /data && \
    chown -R shodh:shodh /data

# Copy binary, ONNX Runtime, and model files from builder
COPY --from=builder /app/target/release/shodh-memory-server /usr/local/bin/shodh-memory
COPY --from=builder /usr/local/lib/libonnxruntime.so /usr/local/lib/libonnxruntime.so
COPY --from=builder --chown=shodh:shodh /models/minilm-l6 /home/shodh/.cache/shodh-memory/models/minilm-l6
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
    LD_LIBRARY_PATH=/usr/local/lib \
    ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so

# Run the binary
CMD ["shodh-memory"]
