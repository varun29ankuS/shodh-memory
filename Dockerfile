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

# Pre-download embedding + NER models (cacheable independent of source code)
# MiniLM-L6-v2 quantized (~23MB) + tokenizer (~700KB) — pinned HuggingFace commit
# GLiNER bi-edge-v2 ONNX typer (~149MB model + label embeddings + tokenizer) —
#   pinned repo release `gliner-bi-edge-onnx-v1`; this is the SOLE neural typer,
#   so baking it here is what makes the image run real NER instead of the
#   rule-based fallback. All assets SHA-256 verified.
ARG GLINER_RELEASE=gliner-bi-edge-onnx-v1
ARG GLINER_BASE=https://github.com/varun29ankuS/shodh-memory/releases/download/${GLINER_RELEASE}
RUN mkdir -p /models/minilm-l6 /models/gliner-bi-edge \
    && curl -fSL -o /models/minilm-l6/model_quantized.onnx \
       "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/onnx/model_quint8_avx2.onnx" \
    && curl -fSL -o /models/minilm-l6/tokenizer.json \
       "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/tokenizer.json" \
    && curl -fSL -o /models/gliner-bi-edge/model.onnx              "${GLINER_BASE}/model.onnx" \
    && curl -fSL -o /models/gliner-bi-edge/label_embeddings.bin    "${GLINER_BASE}/label_embeddings.bin" \
    && curl -fSL -o /models/gliner-bi-edge/label_embeddings.json   "${GLINER_BASE}/label_embeddings.json" \
    && curl -fSL -o /models/gliner-bi-edge/tokenizer.json          "${GLINER_BASE}/tokenizer.json" \
    && curl -fSL -o /models/gliner-bi-edge/tokenizer_config.json   "${GLINER_BASE}/tokenizer_config.json" \
    && curl -fSL -o /models/gliner-bi-edge/special_tokens_map.json "${GLINER_BASE}/special_tokens_map.json" \
    && curl -fSL -o /models/gliner-bi-edge/gliner_config.json      "${GLINER_BASE}/gliner_config.json" \
    && echo "b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21  /models/minilm-l6/model_quantized.onnx" | sha256sum -c - \
    && echo "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037  /models/minilm-l6/tokenizer.json" | sha256sum -c - \
    && echo "209eaeb7fe6703cfa458fd7e4f084a9b078f0cbafe941ee4aae1b68c5a190d02  /models/gliner-bi-edge/model.onnx" | sha256sum -c - \
    && echo "f07cc0fecf3c8bd73a6bb4593e887a6b2ce8ed5d7022c9847407e611a0ed0a74  /models/gliner-bi-edge/label_embeddings.bin" | sha256sum -c - \
    && echo "30979ba33114adae74705a5405901f67e4b90458dc4ab581e4387b35def3b525  /models/gliner-bi-edge/label_embeddings.json" | sha256sum -c - \
    && echo "2315a8bea85452f3c4e8ce980f7853cac013820238e7776a4e48159037a5f164  /models/gliner-bi-edge/tokenizer.json" | sha256sum -c - \
    && echo "1270f8070c3ad1184d77bb700826a03bf0a99f0029b074383079203fec56116f  /models/gliner-bi-edge/tokenizer_config.json" | sha256sum -c - \
    && echo "e386620cb5e9f6570fe98481fde86167b4236cdebdcc42308652574122561619  /models/gliner-bi-edge/special_tokens_map.json" | sha256sum -c - \
    && echo "3ba491748b955c28c33ac5e78b7dc7e6a8c1968f676cbfe5776788e854e02623  /models/gliner-bi-edge/gliner_config.json" | sha256sum -c -

# Create dummy sources to build and cache dependencies
RUN mkdir -p src src/bin benches tests \
    && echo "fn main() {}" > src/main.rs \
    && echo "fn main() {}" > src/cli.rs \
    && echo "fn main() {}" > src/bin/recall_eval.rs \
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
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with home directory
RUN useradd -m -u 1000 shodh && \
    mkdir -p /data && \
    chown -R shodh:shodh /data

# Copy binary and ONNX Runtime from builder
COPY --from=builder /app/target/release/shodh-memory-server /usr/local/bin/shodh-memory
COPY --from=builder /usr/local/lib/libonnxruntime.so /usr/local/lib/libonnxruntime.so
RUN ldconfig

# Copy pre-downloaded models into the image (eliminates runtime downloads)
# Directory structure matches what the downloader expects:
#   minilm-l6/model_quantized.onnx + tokenizer.json      (embedding model)
#   gliner-bi-edge/model.onnx + label_embeddings.* + ...  (GLiNER neural typer)
COPY --from=builder --chown=shodh:shodh /models /home/shodh/.cache/shodh-memory/models

# Switch to non-root user
USER shodh

# Set working directory
WORKDIR /data

# Expose default port
EXPOSE 3030

# Health check — 120s start period allows for first-time RocksDB initialization
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -f http://localhost:3030/health || exit 1

# Set environment variables
# ORT_DYLIB_PATH: tells ort crate where libonnxruntime.so is (skips runtime download)
# SHODH_MODEL_PATH: tells embedder where the pre-baked MiniLM model is (skips download)
# SHODH_GLINER_MODEL_PATH: tells the NER stage where the pre-baked GLiNER typer is
#   (skips first-run download; without it the image would run rule-based fallback NER)
ENV RUST_LOG=info \
    SHODH_HOST=0.0.0.0 \
    SHODH_PORT=3030 \
    SHODH_MEMORY_PATH=/data \
    ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so \
    SHODH_MODEL_PATH=/home/shodh/.cache/shodh-memory/models/minilm-l6 \
    SHODH_GLINER_MODEL_PATH=/home/shodh/.cache/shodh-memory/models/gliner-bi-edge \
    LD_LIBRARY_PATH=/usr/local/lib

# Run the binary
CMD ["shodh-memory"]
