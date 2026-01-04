# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-04

### Added

- **Core compression library** (`trueno-zram-core`)
  - LZ4 compression with AVX2/AVX-512/NEON acceleration
  - ZSTD compression with SIMD optimization
  - Runtime CPU feature detection and dispatch
  - Same-fill detection for 2048:1 zero page compression

- **GPU batch compression**
  - CUDA support via cudarc
  - Pure Rust PTX generation via trueno-gpu
  - Warp-cooperative LZ4 kernel (4 warps/block)
  - PCIe 5x rule evaluation
  - Async DMA ring buffer support

- **Kernel compatibility**
  - Sysfs interface compatible with kernel zram
  - Algorithm compatibility layer
  - Statistics matching kernel format

- **Benchmarking utilities**
  - Criterion benchmarks
  - Example programs for testing
  - Performance measurement infrastructure

### Performance

- LZ4 compression: 4.4 GB/s (AVX-512)
- LZ4 decompression: 5.4 GB/s (AVX-512)
- ZSTD compression: 11.2 GB/s (AVX-512)
- ZSTD decompression: 46 GB/s (AVX-512)
- Same-fill detection: 22 GB/s

### Infrastructure

- Published to crates.io as `trueno-zram-core`
- 461 tests passing
- 94% test coverage
- Full documentation with mdBook

## [Unreleased]

### Planned

- trueno-zram-adaptive: ML-driven algorithm selection
- trueno-zram-generator: systemd integration
- trueno-zram-cli: zramctl replacement
- trueno-ublk: ublk daemon for kernel bypass

---

[0.1.0]: https://github.com/paiml/trueno-zram/releases/tag/v0.1.0
[Unreleased]: https://github.com/paiml/trueno-zram/compare/v0.1.0...HEAD
