# Introduction

**trueno-zram** is a SIMD and GPU-accelerated memory compression library for Linux systems. It provides userspace Rust implementations of LZ4 and ZSTD compression that leverage modern CPU vector instructions (AVX2, AVX-512, NEON) and optional CUDA GPU acceleration.

## Production Status (2026-01-06)

**MILESTONE DT-005 ACHIEVED:** trueno-zram is running as system swap!

- 8GB device active as primary swap (priority 150)
- Validated under memory pressure with ~185MB swap utilization
- Hybrid architecture: CPU SIMD compress (20-24 GB/s) + GPU decompress (137 GB/s)

**Known Limitations:**
- Swap deadlock under extreme memory pressure (fix pending via DT-007 mlock integration)
- GPU compression blocked by NVIDIA F081 bug (using CPU SIMD instead)

## Why trueno-zram?

Linux's kernel zram module provides transparent memory compression, but it has limitations:

1. **Fixed algorithms**: Kernel zram supports limited compression algorithms
2. **No SIMD optimization**: Kernel implementations don't fully utilize modern CPU features
3. **No GPU offload**: Large batch compression can't leverage GPU acceleration
4. **Limited tunability**: Hard to optimize for specific workloads

trueno-zram addresses these limitations by providing:

- **Runtime SIMD dispatch**: Automatically selects AVX-512, AVX2, or NEON based on CPU
- **GPU batch decompression**: Offloads large batches to CUDA GPUs (137 GB/s on RTX 4090)
- **Adaptive algorithm selection**: ML-driven selection based on page entropy
- **Same-fill optimization**: 2048:1 compression for zero/repeated pages
- **Kernel compatibility**: Drop-in replacement via sysfs interface

## Performance Highlights

| Metric | Achieved | vs Linux Kernel |
|--------|----------|-----------------|
| LZ4 Compression (sequential) | 3.7 GB/s | **6.9x faster** |
| LZ4 Compression (parallel) | 19-24 GB/s | **35-45x faster** |
| LZ4 Decompression | 5.4 GB/s | +54% |
| GPU Decompression | 137 GB/s | **22.8x speedup** |
| ZSTD Compression | 11.2 GB/s | N/A |
| Same-fill detection | 22 GB/s | **2.75x faster** |
| 10GB scale validated | 19-24 GB/s | **Production ready** |

## Part of the PAIML Ecosystem

trueno-zram is part of the "Batuta Stack":

- [trueno](https://crates.io/crates/trueno) - High-performance SIMD compute library
- [trueno-gpu](https://crates.io/crates/trueno-gpu) - Pure Rust PTX generation for CUDA
- [aprender](https://crates.io/crates/aprender) - Machine learning in pure Rust
- [certeza](https://crates.io/crates/certeza) - Asymptotic test effectiveness framework

## License

trueno-zram is dual-licensed under MIT and Apache-2.0.
