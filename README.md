# trueno-zram

```
  ████████╗██████╗ ██╗   ██╗███████╗███╗   ██╗ ██████╗       ███████╗██████╗  █████╗ ███╗   ███╗
  ╚══██╔══╝██╔══██╗██║   ██║██╔════╝████╗  ██║██╔═══██╗      ╚══███╔╝██╔══██╗██╔══██╗████╗ ████║
     ██║   ██████╔╝██║   ██║█████╗  ██╔██╗ ██║██║   ██║  █████╗ ███╔╝ ██████╔╝███████║██╔████╔██║
     ██║   ██╔══██╗██║   ██║██╔══╝  ██║╚██╗██║██║   ██║  ╚════╝███╔╝  ██╔══██╗██╔══██║██║╚██╔╝██║
     ██║   ██║  ██║╚██████╔╝███████╗██║ ╚████║╚██████╔╝       ███████╗██║  ██║██║  ██║██║ ╚═╝ ██║
     ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝        ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝

                    ⚡ SIMD-Accelerated Memory Compression ⚡
                       3.87x Ratio | 20-30 GB/s Compress | P99 < 20µs
```

[![Crates.io](https://img.shields.io/crates/v/trueno-zram-core.svg)](https://crates.io/crates/trueno-zram-core)
[![Tests](https://img.shields.io/badge/tests-461%20passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-94%25-brightgreen)]()
[![MSRV](https://img.shields.io/badge/MSRV-1.82.0-blue)]()
[![Status](https://img.shields.io/badge/status-PRODUCTION-success)]()

SIMD and GPU-accelerated memory compression library for Linux systems. Part of the PAIML "Batuta Stack" (trueno + bashrs + aprender).

## Production Status

**MILESTONE ACHIEVED (2026-01-06):** trueno-zram is running as system swap!

| Status | Description |
|--------|-------------|
| **DT-005** | 8GB trueno-zram device active as primary swap (priority 150) |
| **DT-007** | Swap deadlock FIXED via mlock() - 211 MB daemon memory pinned |
| **Compression** | 3.87x ratio (55% better than kernel ZRAM's 2.5x) |
| **Latency** | P99 < 20µs |

### Current Architecture: Userspace ublk

```
Production Pipeline:
  Compress: CPU SIMD (AVX-512) @ 20-30 GB/s, 3.87x ratio
  Decompress: CPU parallel @ 48 GB/s
  I/O: ublk userspace block device
```

**Important:** Kernel ZRAM has higher raw I/O throughput (operates entirely in kernel space). trueno-zram's advantage is **compression efficiency** (3.87x vs 2.5x) and **userspace flexibility**.

## Overview

trueno-zram provides userspace Rust implementations of LZ4 and ZSTD compression that leverage modern CPU vector instructions (AVX2, AVX-512, NEON) and optional CUDA GPU acceleration to replace kernel-level zram compression.

## Features

- **SIMD Acceleration**: Runtime CPU detection with optimized paths for AVX2, AVX-512, and ARM NEON
- **GPU Acceleration**: Optional CUDA support for batch compression (RTX 4090, A100, H100)
- **LZ4 Compression**: High-speed compression targeting 3+ GB/s throughput
- **ZSTD Compression**: Better ratios for compressible data (up to 13 GB/s with AVX-512)
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
use trueno_zram_core::gpu::{GpuBatchCompressor, GpuBatchConfig, gpu_available};
use trueno_zram_core::{Algorithm, PAGE_SIZE};

if gpu_available() {
    let config = GpuBatchConfig {
        device_index: 0,
        algorithm: Algorithm::Lz4,
        batch_size: 1000,
        async_dma: true,
        ring_buffer_slots: 4,
    };

    let mut compressor = GpuBatchCompressor::new(config)?;

    let pages: Vec<[u8; PAGE_SIZE]> = vec![[0u8; PAGE_SIZE]; 1000];
    let result = compressor.compress_batch(&pages)?;

    println!("Compressed {} pages", result.pages.len());
    println!("Compression ratio: {:.2}x", result.compression_ratio());
    println!("PCIe 5x rule satisfied: {}", result.pcie_rule_satisfied());
}
```

## Examples

```bash
# GPU information and backend selection
cargo run -p trueno-zram-core --example gpu_info
cargo run -p trueno-zram-core --example gpu_info --features cuda

# Compression benchmarks (use --release for accurate numbers)
cargo run -p trueno-zram-core --example compress_benchmark --release
cargo run -p trueno-zram-core --example compress_benchmark --release --features cuda
```

### Example Output

```
trueno-zram Compression Benchmark
=================================

Pattern: Zeros (compressible)
----------------------------------------------------------------------
   Pages  Algorithm     Compress   Decompress      Ratio    Backend
    1000        Lz4      22.01 GB/s      46.02 GB/s   2048.00x Avx512
    1000       Zstd      10.33 GB/s      29.70 GB/s   2048.00x Avx512

Pattern: Text (compressible)
----------------------------------------------------------------------
    1000        Lz4       4.44 GB/s       5.37 GB/s      3.21x Avx512
    1000       Zstd      11.27 GB/s      16.23 GB/s      4.52x Avx512

Pattern: Random (incompressible)
----------------------------------------------------------------------
    1000        Lz4       1.61 GB/s      31.64 GB/s      1.00x Avx512
    1000       Zstd       8.72 GB/s      46.49 GB/s      1.00x Avx512
```

## Crate Structure

| Crate | Description | crates.io |
|-------|-------------|-----------|
| **trueno-zram-core** | SIMD/GPU-vectorized LZ4/ZSTD compression engines | [![](https://img.shields.io/crates/v/trueno-zram-core.svg)](https://crates.io/crates/trueno-zram-core) |
| **trueno-zram-adaptive** | ML-driven compression algorithm selection | - |
| **trueno-zram-generator** | systemd integration for zram device configuration | - |
| **trueno-zram-cli** | Rust-native zramctl replacement | - |

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

# Build all workspace crates
cargo build --workspace --all-features
```

## Testing

```bash
# All tests
cargo test --workspace --all-features

# Core library tests only
cargo test -p trueno-zram-core --all-features

# With CUDA
cargo test --workspace --features cuda

# Coverage report
cargo llvm-cov --workspace --all-features
```

## Performance

### Validated Claims (QA-FALSIFY-001, 2026-01-06)

| Claim | Result | Status |
|-------|--------|--------|
| Compression ratio | **3.87x** vs kernel's 2.5x | **PASS** |
| SIMD compression | **20-30 GB/s** parallel | **PASS** |
| SIMD decompression | **48 GB/s** | **PASS** |
| P99 latency | **16.5 µs** (< 100µs threshold) | **PASS** |
| mlock (DT-007) | **211 MB** locked | **PASS** |

### Falsified Claims (Corrected)

| Original Claim | Actual Result | Notes |
|----------------|---------------|-------|
| ~~1.8x vs kernel ZRAM I/O~~ | Kernel 3-13x faster | ublk userspace overhead |
| ~~228K IOPS~~ | 123K IOPS | Still good, but overstated |
| ~~2048:1 same-fill~~ | 157:1 zeros | Implementation difference |

### Why Kernel ZRAM Has Higher I/O

Kernel ZRAM operates entirely in kernel space with zero syscall overhead. trueno-zram uses ublk which requires a userspace hop. Even with io_uring shared memory, context switching adds latency.

**The true value proposition of trueno-zram is NOT raw I/O speed, but:**

1. **Better compression ratio:** 3.87x vs kernel's ~2.5x (**55% more space efficient**)
2. **SIMD-accelerated compression:** 20-30 GB/s throughput
3. **Low latency:** P99 < 20µs
4. **Userspace flexibility:** debugging, monitoring, custom algorithms, GPU offload potential

### Component Performance (Validated)

| Metric | Target | Achieved |
|--------|--------|----------|
| LZ4 Compression | >= 3 GB/s | 4.4 GB/s (AVX-512) |
| LZ4 Decompression | >= 5 GB/s | 5.4 GB/s (AVX-512) |
| ZSTD Compression | >= 8 GB/s | 11.2 GB/s (AVX-512) |
| ZSTD Decompression | >= 20 GB/s | 46 GB/s (AVX-512) |
| CPU Parallel Decompress | >= 30 GB/s | 48 GB/s (Threadripper) |
| P99 Latency | < 100µs | 16.5µs |
| Compression Ratio | >= 3x | 3.87x |

### Lambda Lab Tiers

Automatic hardware detection for optimal configuration:

| Tier | GPU | CPU | Batch Size |
|------|-----|-----|------------|
| Full | H100/A100 | AVX-512 | 10,000 pages |
| High | RTX 4090 | AVX-512 | 5,000 pages |
| Medium | Consumer | AVX2 | 1,000 pages |
| Minimal | None | SSE4.2 | 100 pages |

## GPU Support

GPU decompression is available but CPU parallel path is faster for most workloads due to PCIe transfer overhead.

| Path | Throughput | Notes |
|------|------------|-------|
| CPU Parallel Decompress | 50+ GB/s | Primary path, pre-allocated buffers |
| GPU End-to-End | ~6 GB/s | PCIe 4.0 transfer bottleneck |
| GPU Kernel-only | ~9 GB/s | Without H2D/D2H transfers |

> **Recommendation:** Use CPU parallel path for best performance. GPU useful for future PCIe 5.0+ systems.

## Requirements

- Linux Kernel >= 5.10 LTS with zram module
- x86_64 (AVX2/AVX-512) or AArch64 (NEON)
- Rust >= 1.82.0
- Optional: CUDA 12.8+ for GPU acceleration

## Related Crates

Part of the PAIML ecosystem:

- [trueno](https://crates.io/crates/trueno) - High-performance SIMD compute library
- [trueno-gpu](https://crates.io/crates/trueno-gpu) - Pure Rust PTX generation for CUDA
- [aprender](https://crates.io/crates/aprender) - Machine learning in pure Rust
- [certeza](https://crates.io/crates/certeza) - Asymptotic test effectiveness framework

## License

MIT OR Apache-2.0
