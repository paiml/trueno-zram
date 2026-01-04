# Introduction

**trueno-zram** is a SIMD and GPU-accelerated memory compression library for Linux systems. It provides userspace Rust implementations of LZ4 and ZSTD compression that leverage modern CPU vector instructions (AVX2, AVX-512, NEON) and optional CUDA GPU acceleration.

## Why trueno-zram?

Linux's kernel zram module provides transparent memory compression, but it has limitations:

1. **Fixed algorithms**: Kernel zram supports limited compression algorithms
2. **No SIMD optimization**: Kernel implementations don't fully utilize modern CPU features
3. **No GPU offload**: Large batch compression can't leverage GPU acceleration
4. **Limited tunability**: Hard to optimize for specific workloads

trueno-zram addresses these limitations by providing:

- **Runtime SIMD dispatch**: Automatically selects AVX-512, AVX2, or NEON based on CPU
- **GPU batch compression**: Offloads large batches to CUDA GPUs when beneficial
- **Adaptive algorithm selection**: ML-driven selection based on page entropy
- **Same-fill optimization**: 2048:1 compression for zero/repeated pages
- **Kernel compatibility**: Drop-in replacement via sysfs interface

## Performance Highlights

| Metric | Achieved |
|--------|----------|
| LZ4 Compression | 4.4 GB/s (AVX-512) |
| LZ4 Decompression | 5.4 GB/s (AVX-512) |
| ZSTD Compression | 11.2 GB/s (AVX-512) |
| ZSTD Decompression | 46 GB/s (AVX-512) |
| Same-fill ratio | 2048:1 |

## Part of the PAIML Ecosystem

trueno-zram is part of the "Batuta Stack":

- [trueno](https://crates.io/crates/trueno) - High-performance SIMD compute library
- [trueno-gpu](https://crates.io/crates/trueno-gpu) - Pure Rust PTX generation for CUDA
- [aprender](https://crates.io/crates/aprender) - Machine learning in pure Rust
- [certeza](https://crates.io/crates/certeza) - Asymptotic test effectiveness framework

## License

trueno-zram is dual-licensed under MIT and Apache-2.0.
