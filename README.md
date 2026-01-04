# trueno-zram

SIMD and GPU-accelerated memory compression library for Linux systems. Part of the PAIML "Batuta Stack" (trueno + bashrs + aprender).

[![Tests](https://img.shields.io/badge/tests-394%20passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)]()
[![MSRV](https://img.shields.io/badge/MSRV-1.82.0-blue)]()

## Overview

trueno-zram provides userspace Rust implementations of LZ4 and ZSTD compression that leverage modern CPU vector instructions (AVX2, AVX-512, NEON) and optional CUDA GPU acceleration to replace kernel-level zram compression.

## Features

- **SIMD Acceleration**: Runtime CPU detection with optimized paths for AVX2, AVX-512, and ARM NEON
- **GPU Acceleration**: Optional CUDA support for batch compression (RTX 4090, A100, H100)
- **LZ4 Compression**: High-speed compression targeting 3+ GB/s throughput
- **ZSTD Compression**: Better ratios for compressible data
- **Adaptive Selection**: ML-driven algorithm selection based on page entropy
- **Same-Fill Detection**: 2048:1 compression for zero/repeated pages
- **Page-Based**: Optimized for 4KB memory pages
- **Kernel Compatible**: Sysfs interface compatible with kernel zram

## Installation

```bash
# Core library
cargo add trueno-zram-core

# With CUDA support (requires CUDA 12.8)
cargo add trueno-zram-core --features cuda
```

## Usage

```rust
use trueno_zram_core::{CompressorBuilder, Algorithm, PAGE_SIZE};

// Create a compressor with LZ4 algorithm
let compressor = CompressorBuilder::new()
    .algorithm(Algorithm::Lz4)
    .build()?;

// Compress a page
let page = [0u8; PAGE_SIZE];
let compressed = compressor.compress(&page)?;

// Decompress
let decompressed = compressor.decompress(&compressed)?;
assert_eq!(page, decompressed);
```

### GPU Batch Compression

```rust
use trueno_zram_core::gpu::{GpuCompressor, BatchCompressionRequest, gpu_available};
use trueno_zram_core::{Algorithm, PAGE_SIZE};

if gpu_available() {
    let compressor = GpuCompressor::new(0, Algorithm::Lz4)?;

    let pages: Vec<[u8; PAGE_SIZE]> = vec![[0u8; PAGE_SIZE]; 1000];
    let request = BatchCompressionRequest {
        pages,
        algorithm: Algorithm::Lz4,
    };

    let result = compressor.compress_batch(&request)?;
    println!("Compressed {} pages in {} ms",
             result.compressed.len(),
             result.total_time_ns / 1_000_000);
}
```

## Examples

```bash
# GPU information and backend selection
cargo run --example gpu_info
cargo run --example gpu_info --features cuda  # With CUDA

# Compression benchmarks
cargo run --example compress_benchmark --release
```

## Crate Structure

| Crate | Description |
|-------|-------------|
| **trueno-zram-core** | SIMD/GPU-vectorized LZ4/ZSTD compression engines |
| **trueno-zram-adaptive** | ML-driven compression algorithm selection |
| **trueno-zram-generator** | systemd integration for zram device configuration |
| **trueno-zram-cli** | Rust-native zramctl replacement |

### Core Modules

- `lz4/` - LZ4 compression with AVX2/AVX-512/NEON acceleration
- `zstd/` - Zstandard compression with SIMD optimization
- `gpu/` - CUDA batch compression with PCIe 5x rule
- `samefill` - Same-fill page detection (2048:1 for zero pages)
- `compat` - Kernel zram sysfs compatibility layer
- `benchmark` - Performance benchmarking utilities

## Building

```bash
# Standard build
cargo build --release --all-features

# With CUDA support (requires CUDA 12.8)
cargo build --release --features cuda
```

## Testing

```bash
# All tests
cargo test --workspace

# With CUDA
cargo test --workspace --features cuda

# Coverage report
cargo llvm-cov --workspace
```

## Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| LZ4 Compression | >= 3 GB/s | 3.2 GB/s (AVX2) |
| LZ4 Decompression | >= 5 GB/s | 5.1 GB/s (AVX2) |
| SIMD vs Scalar | >= 40% improvement | 45% faster |
| P99 Latency | < 100us | 85us |
| Same-fill ratio | 2048:1 | 2048:1 |

### Lambda Lab Tiers

Automatic hardware detection for optimal configuration:

| Tier | GPU | CPU | Batch Size |
|------|-----|-----|------------|
| Full | H100/A100 | AVX-512 | 10,000 pages |
| High | RTX 4090 | AVX-512 | 5,000 pages |
| Medium | Consumer | AVX2 | 1,000 pages |
| Minimal | None | SSE4.2 | 100 pages |

## Requirements

- Linux Kernel >= 5.10 LTS with zram module
- x86_64 (AVX2/AVX-512) or AArch64 (NEON)
- Rust >= 1.82.0
- Optional: CUDA 12.8+ for GPU acceleration

## License

MIT OR Apache-2.0
