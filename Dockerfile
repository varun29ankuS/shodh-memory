# Multi-stage build for minimal final image
FROM rust:1.75-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src ./src

# Build release binary
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 shodh && \
    mkdir -p /data && \
    chown -R shodh:shodh /data

# Copy binary from builder
COPY --from=builder /app/target/release/shodh-memory /usr/local/bin/shodh-memory

# Switch to non-root user
USER shodh

# Set working directory
WORKDIR /data

# Expose default port
EXPOSE 3030

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3030/health || exit 1

# Set environment variables
ENV RUST_LOG=info \
    PORT=3030 \
    STORAGE_PATH=/data

# Run the binary
CMD ["shodh-memory"]
