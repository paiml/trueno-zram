# trueno-zram

## Project Specification v1.0.0

**SIMD-Accelerated Memory Compression for the Sovereign AI Stack**

---

## Document Control

| Field | Value |
|-------|-------|
| **Project** | trueno-zram |
| **Version** | 1.0.0 |
| **Author** | Noah Gift <noah@paiml.com> |
| **Organization** | Pragmatic AI Labs (PAIML) |
| **License** | MIT OR Apache-2.0 |
| **Repository** | https://github.com/paiml/trueno-zram |
| **PMAT Level** | 4 (Managed & Measured) |
| **Last Updated** | 2025-12-28 |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Scientific Foundation](#2-scientific-foundation)
3. [Architecture](#3-architecture)
4. [Toyota Way Principles](#4-toyota-way-principles)
5. [Extreme TDD Methodology](#5-extreme-tdd-methodology)
6. [PMAT Quality Framework](#6-pmat-quality-framework)
7. [Project Enforcement](#7-project-enforcement)
8. [100-Point Popperian Falsification Checklist](#8-100-point-popperian-falsification-checklist)
9. [Implementation Roadmap](#9-implementation-roadmap)
10. [References](#10-references)

---

## 1. Executive Summary

### 1.1 Vision

trueno-zram delivers SIMD-accelerated memory compression for Linux systems, showcasing the complete PAIML "Batuta Stack" (trueno + bashrs + aprender). It replaces kernel-level compression with userspace Rust implementations that leverage modern CPU vector instructions.

### 1.2 Problem Statement

Memory compression in the Linux kernel (zram/zswap) uses scalar C implementations of LZ4 and ZSTD. Modern CPUs provide SIMD instruction sets (AVX2, AVX-512, NEON) that can accelerate compression by 40-110% [1]. Current implementations do not fully exploit this potential.

### 1.3 Solution

trueno-zram provides:

1. **trueno-core**: SIMD-vectorized LZ4/ZSTD compression engines
2. **trueno-generator**: systemd integration for zram device configuration
3. **trueno-cli**: Rust-native zramctl replacement
4. **trueno-adaptive**: ML-driven compression algorithm selection

### 1.4 Falsifiable Hypothesis (Primary)

> **H₀**: trueno-zram SIMD compression achieves ≥40% throughput improvement over kernel scalar LZ4 on the Silesia corpus benchmark, measured at p<0.05 significance level.

If this hypothesis is falsified through rigorous benchmarking, the project will pivot or terminate.

### 1.5 System Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Linux Kernel ≥ 5.10 LTS (`zram` module enabled) |
| **CPU** | x86_64 (AVX2/AVX-512) or AArch64 (NEON) |
| **Memory** | Minimum 512MB system RAM |
| **Privileges** | `CAP_SYS_ADMIN` required for device configuration |
| **Rust** | Stable toolchain ≥ 1.70.0 |

---

## 2. Scientific Foundation

### 2.1 Peer-Reviewed Citations

#### 2.1.1 Compression Algorithms

**[1] Ziv, J. and Lempel, A. (1977).** "A universal algorithm for sequential data compression." *IEEE Transactions on Information Theory*, 23(3), 337-343.
- Foundation of LZ77 family algorithms
- Establishes theoretical basis for dictionary compression

**[2] Ziv, J. and Lempel, A. (1978).** "Compression of individual sequences via variable-rate coding." *IEEE Transactions on Information Theory*, 24(5), 530-536.
- LZ78 algorithm foundation
- Theoretical compression limits

**[3] Liu, W., Mei, F., Wang, C., et al. (2018).** "Data Compression Device based on Modified LZ4 Algorithm." *IEEE Transactions on Consumer Electronics*, 64(1), 110-117.
- Hardware LZ4 implementation achieving 1.92 Gbps throughput
- Modified algorithm for real-time processing
- Compression ratio up to 2.05x

**[4] Bartik, M., Ubik, S., and Kubalik, P. (2015).** "LZ4 compression algorithm on FPGA." *IEEE International Conference on Electronics, Circuits, and Systems*, Cairo, Egypt, 179-182.
- LZ4 hardware analysis and bottleneck identification
- Suitability for parallel implementation

**[5] Collet, Y. and Kucherawy, M. (2018).** "Zstandard Compression and the application/zstd Media Type." *RFC 8478*, IETF.
- Official Zstandard specification
- Entropy coding via FSE and Huffman

**[6] Maulidina, A.P., Wijaya, R.A., et al. (2024).** "Comparative Study of Data Compression Algorithms: Zstandard, zlib & LZ4." *Communications in Computer and Information Science*, vol. 2198, Springer.
- Benchmark methodology on Silesia corpus
- Compression ratio vs speed tradeoffs

#### 2.1.2 SIMD Optimization

**[7] Schlegel, B., Gemulla, R., and Lehner, W. (2010).** "Fast integer compression using SIMD instructions." *Proceedings of the Sixth International Workshop on Data Management on New Hardware*, 34-40.
- SIMD decompression speedup of 1.5x-6.7x
- Vectorized null suppression and Elias gamma encoding

**[8] Zhang, J., Long, X., and Suel, T. (2016).** "A General SIMD-Based Approach to Accelerating Compression Algorithms." *ACM Transactions on Information Systems*, 34(3), Article 15.
- **Key finding: SIMD algorithms outperform non-SIMD by 40-110%**
- Group-Simple, Group-Scheme, Group-AFOR, Group-PFD algorithms
- Evaluated on TREC, Wikipedia, and Twitter datasets

**[9] Lemire, D. and Boytsov, L. (2015).** "Decoding billions of integers per second through vectorization." *Software: Practice and Experience*, 45(1), 1-29.
- SIMD-BP128 scheme
- Nearly twice as fast as varint-G8IU and PFOR

**[10] Dube, G., et al. (2022).** "SIMD Lossy Compression for Scientific Data." *arXiv:2201.04614*.
- 15x speedup over SZ-1.4
- Prediction/quantization bandwidth >3.4 GB/s

#### 2.1.3 Memory Compression Systems

**[11] Jennings, S. (2013).** "zswap: compressed swap caching." *Linux Kernel Documentation*.
- Linux kernel zswap architecture
- Frontswap API integration

**[12] Gupta, N. (2014).** "zram: Compressed RAM based block devices." *Linux Kernel Documentation*, v3.14.
- zram module design
- Compression algorithm selection interface

#### 2.1.4 Rust and Systems Programming

**[13] Matsakis, N.D. and Klock, F.S. (2014).** "The Rust Language." *ACM SIGAda Ada Letters*, 34(3), 103-104.
- Memory safety without garbage collection
- Zero-cost abstractions

**[14] Jung, R., et al. (2017).** "RustBelt: Securing the Foundations of the Rust Programming Language." *Proceedings of the ACM on Programming Languages*, 2(POPL), Article 66.
- Formal verification of Rust's type system
- Safety guarantees for unsafe code

### 2.2 Theoretical Framework

#### 2.2.1 Compression Bound (Shannon Entropy)

For a source X with probability distribution P(x):

```
H(X) = -Σ P(x) log₂ P(x)
```

No lossless compression algorithm can achieve better than H(X) bits per symbol on average.

#### 2.2.2 SIMD Parallelism Factor

Theoretical speedup from k-way SIMD:

```
Speedup_theoretical = k
Speedup_actual = k / (1 + overhead_factor)
```

Where overhead_factor accounts for:
- Data alignment costs
- Shuffle/permute instructions
- Horizontal operations

**Expected range**: 2x-8x for 256-bit AVX2, 4x-16x for 512-bit AVX-512

#### 2.2.3 Compression Ratio vs Throughput Pareto Frontier

```
             Compression Ratio
                    ^
                    |     * lzma
                    |   * zstd-19
                    |  * zstd-9
                    | * zstd-3
                    |* lz4-hc
         Pareto    *| lz4
         Frontier   |
                    +-------------------> Throughput (GB/s)
```

trueno-zram targets the LZ4 and ZSTD-3 region with SIMD acceleration.

---

## 3. Architecture

### 3.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         trueno-zram                                 │
│                  "Sovereign Memory Compression"                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────────────┐ │
│  │ trueno-core   │   │trueno-adaptive│   │    trueno-generator   │ │
│  │               │   │               │   │                       │ │
│  │ ┌───────────┐ │   │ ┌───────────┐ │   │ ┌───────────────────┐ │ │
│  │ │ LZ4 SIMD  │ │   │ │ Entropy   │ │   │ │ systemd units     │ │ │
│  │ │ AVX2/512  │ │◄──┤ │ Analyzer  │ │   │ │ configuration     │ │ │
│  │ └───────────┘ │   │ └───────────┘ │   │ └───────────────────┘ │ │
│  │ ┌───────────┐ │   │ ┌───────────┐ │   │ ┌───────────────────┐ │ │
│  │ │ ZSTD SIMD │ │   │ │ aprender  │ │   │ │ fstab generation  │ │ │
│  │ │ AVX2/512  │ │◄──┤ │ ML model  │ │   │ │                   │ │ │
│  │ └───────────┘ │   │ └───────────┘ │   │ └───────────────────┘ │ │
│  └───────────────┘   └───────────────┘   └───────────────────────┘ │
│          │                   │                       │             │
│          └───────────────────┴───────────────────────┘             │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                      trueno-cli                              │  │
│  │         Rust-native zramctl replacement                      │  │
│  │    • Device creation    • Statistics    • Benchmarking       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Linux Kernel                              │  │
│  │         /dev/zram0  ◄──  zram module  ◄──  mm subsystem      │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Crate Structure

```
trueno-zram/
├── Cargo.toml                      # Workspace root
├── README.md
├── SPECIFICATION.md                # This document
├── CHANGELOG.md
├── LICENSE-MIT
├── LICENSE-APACHE
│
├── crates/
│   ├── trueno-zram-core/           # SIMD compression engines
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── lz4/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── compress.rs     # SIMD LZ4 compression
│   │   │   │   ├── decompress.rs   # SIMD LZ4 decompression
│   │   │   │   ├── avx2.rs         # AVX2 implementation
│   │   │   │   ├── avx512.rs       # AVX-512 implementation
│   │   │   │   └── neon.rs         # ARM NEON implementation
│   │   │   ├── zstd/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── compress.rs
│   │   │   │   ├── decompress.rs
│   │   │   │   ├── fse.rs          # Finite State Entropy
│   │   │   │   └── huffman.rs      # SIMD Huffman decoder
│   │   │   ├── simd/
│   │   │   │   ├── mod.rs
│   │   │   │   ├── detect.rs       # CPU feature detection
│   │   │   │   └── dispatch.rs     # Runtime dispatch
│   │   │   └── page.rs             # 4KB page compression
│   │   ├── benches/
│   │   │   ├── compression.rs      # Criterion benchmarks
│   │   │   ├── silesia.rs          # Silesia corpus benchmark
│   │   │   └── vs_kernel.rs        # Comparison with kernel impl
│   │   └── tests/
│   │       ├── roundtrip.rs        # Compression/decompression
│   │       ├── fuzz.rs             # Fuzz testing harness
│   │       └── property.rs         # Property-based tests
│   │
│   ├── trueno-zram-adaptive/       # ML-driven algorithm selection
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── entropy.rs          # Shannon entropy calculation
│   │   │   ├── classifier.rs       # Page classification
│   │   │   └── model.rs            # aprender integration
│   │   └── tests/
│   │       └── accuracy.rs
│   │
│   ├── trueno-zram-generator/      # systemd integration
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── main.rs
│   │   │   ├── config.rs           # Configuration parsing
│   │   │   ├── unit.rs             # Unit file generation
│   │   │   └── fstab.rs            # fstab entry generation
│   │   └── tests/
│   │       └── generation.rs
│   │
│   └── trueno-zram-cli/            # Management CLI
│       ├── Cargo.toml
│       ├── src/
│       │   ├── main.rs
│       │   ├── commands/
│       │   │   ├── mod.rs
│       │   │   ├── create.rs
│       │   │   ├── remove.rs
│       │   │   ├── status.rs
│       │   │   └── benchmark.rs
│       │   └── output.rs           # Formatting
│       └── tests/
│           └── cli.rs
│
├── scripts/                        # bashrs-quality scripts
│   ├── setup.sh                    # Full system setup
│   ├── benchmark.sh                # Run benchmarks
│   ├── install.sh                  # Install binaries
│   └── ci/
│       ├── test.sh
│       └── lint.sh
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── BENCHMARKS.md
│   ├── CONTRIBUTING.md
│   └── API.md
│
├── config/
│   └── trueno-zram.conf.example
│
└── .github/
    └── workflows/
        ├── ci.yml
        ├── benchmark.yml
        └── release.yml
```

### 3.3 Core API Design

```rust
//! trueno-zram-core/src/lib.rs

/// Compression algorithm selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    /// LZ4 fast compression
    Lz4,
    /// LZ4-HC high compression
    Lz4Hc,
    /// Zstandard with configurable level
    Zstd { level: i32 },
    /// Adaptive selection based on entropy
    Adaptive,
}

/// SIMD implementation backend
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdBackend {
    /// Scalar fallback (no SIMD)
    Scalar,
    /// SSE4.2 (128-bit)
    Sse42,
    /// AVX2 (256-bit)
    Avx2,
    /// AVX-512 (512-bit)
    Avx512,
    /// ARM NEON (128-bit)
    Neon,
}

/// Page compression result
#[derive(Debug)]
pub struct CompressedPage {
    /// Compressed data
    pub data: Vec<u8>,
    /// Original size (always 4096 for pages)
    pub original_size: usize,
    /// Algorithm used
    pub algorithm: Algorithm,
    /// Compression ratio (original / compressed)
    pub ratio: f64,
}

/// Main compression interface
pub trait PageCompressor: Send + Sync {
    /// Compress a 4KB page
    fn compress(&self, page: &[u8; 4096]) -> Result<CompressedPage, Error>;
    
    /// Decompress to a 4KB page
    fn decompress(&self, compressed: &CompressedPage) -> Result<[u8; 4096], Error>;
    
    /// Get the SIMD backend in use
    fn backend(&self) -> SimdBackend;
    
    /// Get compression statistics
    fn stats(&self) -> CompressionStats;
}

/// Builder for configuring compressor
pub struct CompressorBuilder {
    algorithm: Algorithm,
    preferred_backend: Option<SimdBackend>,
    // ...
}

impl CompressorBuilder {
    pub fn new() -> Self { /* ... */ }
    pub fn algorithm(mut self, algo: Algorithm) -> Self { /* ... */ }
    pub fn prefer_backend(mut self, backend: SimdBackend) -> Self { /* ... */ }
    pub fn build(self) -> Result<Box<dyn PageCompressor>, Error> { /* ... */ }
}
```

### 3.4 SIMD LZ4 Implementation Strategy

Based on [7] and [8], the SIMD implementation follows:

```rust
//! trueno-zram-core/src/lz4/avx2.rs

use std::arch::x86_64::*;

/// AVX2-accelerated LZ4 decompression
/// 
/// Strategy based on Zhang et al. [8]:
/// - 4-way vertical data layout
/// - Pre-generated SIMD instruction sequences
/// - Lookup tables for pattern decoding
#[target_feature(enable = "avx2")]
pub unsafe fn decompress_avx2(
    src: &[u8],
    dst: &mut [u8; 4096],
) -> Result<usize, Error> {
    // Match copy using 256-bit operations
    // Process 32 bytes per iteration
    let mut src_ptr = src.as_ptr();
    let mut dst_ptr = dst.as_mut_ptr();
    
    // Token parsing with SIMD gather
    // ...
    
    // Literal copy with aligned stores
    // ...
    
    // Match copy with overlapping handling
    // ...
    
    Ok(decompressed_size)
}
```

### 3.5 Privilege Model

`trueno-zram` operates across userspace and kernel space boundaries, requiring specific security considerations:

1.  **Configuration (`trueno-cli`)**:
    *   Requires `CAP_SYS_ADMIN` or `root` to interact with `/sys/class/zram-control` and `/sys/block/zram*`.
    *   Follows the Principle of Least Privilege: drops capabilities immediately after opening device handles if possible.

2.  **Compression Engine (`trueno-core`)**:
    *   Pure userspace library.
    *   Memory-safe (except specific SIMD `unsafe` blocks).
    *   No elevated privileges required for compression/decompression logic.

3.  **System Integration (`trueno-generator`)**:
    *   Runs as root during early boot (systemd generator phase).
    *   Must be panic-free to prevent boot failures.

---

## 4. Toyota Way Principles

trueno-zram adopts the 14 principles of the Toyota Production System (TPS) for software development:

### 4.1 Philosophy (Long-term Thinking)

| Principle | Application |
|-----------|-------------|
| **1. Base decisions on long-term philosophy** | Build for the Sovereign AI Stack ecosystem, not short-term gains. Prioritize correctness and safety over premature optimization. |

### 4.2 Process (Eliminate Waste)

| Principle | Application |
|-----------|-------------|
| **2. Create continuous process flow** | CI/CD pipeline runs on every commit. No manual gates. |
| **3. Use pull systems** | Feature development driven by benchmarked performance gaps. |
| **4. Level the workload (heijunka)** | Sprint planning balances features, tests, and documentation. |
| **5. Build culture of stopping to fix problems** | Failing tests block merge. No "fix later" comments. |
| **6. Standardized tasks are foundation** | All code follows rustfmt + clippy + custom lints. |
| **7. Use visual control** | Dashboard shows benchmark trends, coverage, mutation score. |
| **8. Use only reliable, tested technology** | Dependencies must have >1.0 version, active maintenance. |

### 4.3 People (Respect and Challenge)

| Principle | Application |
|-----------|-------------|
| **9. Grow leaders who understand work** | Contributors must run benchmarks locally before PR. |
| **10. Develop exceptional people and teams** | Pair programming for complex SIMD code. |
| **11. Respect extended network** | Upstream contributions to lz4, zstd, Rust ecosystem. |

### 4.4 Problem Solving (Continuous Improvement)

| Principle | Application |
|-----------|-------------|
| **12. Go see for yourself (genchi genbutsu)** | Profile real workloads, not synthetic benchmarks only. |
| **13. Make decisions slowly, implement rapidly** | RFC process for API changes; fast iteration on implementation. |
| **14. Become learning organization (hansei/kaizen)** | Post-mortems for performance regressions; monthly architecture reviews. |

### 4.5 Jidoka (Automation with Human Touch)

```
┌─────────────────────────────────────────────────────────────────┐
│                     Jidoka Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │
│  │  Code   │───►│  Test   │───►│ Bench   │───►│ Deploy  │     │
│  │  Push   │    │  Gate   │    │  Gate   │    │  Gate   │     │
│  └─────────┘    └────┬────┘    └────┬────┘    └────┬────┘     │
│                      │              │              │           │
│                      ▼              ▼              ▼           │
│                 ┌─────────┐    ┌─────────┐    ┌─────────┐     │
│                 │ STOP if │    │ STOP if │    │ STOP if │     │
│                 │ failing │    │ regress │    │ unsafe  │     │
│                 │ tests   │    │ >5%     │    │ code    │     │
│                 └─────────┘    └─────────┘    └─────────┘     │
│                      │              │              │           │
│                 Human review   Human review   Human review     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.6 Andon (Signal System)

CI status indicators:

| Status | Meaning | Action |
|--------|---------|--------|
| 🟢 Green | All checks pass | Proceed |
| 🟡 Yellow | Warnings present | Review before merge |
| 🔴 Red | Tests/benchmarks fail | Stop and fix |
| ⚪ Gray | Infrastructure issue | Investigate CI |

---

## 5. Extreme TDD Methodology

### 5.1 Test-First Development Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│                    Extreme TDD Cycle                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│     ┌──────────────┐                                           │
│     │ 1. Write     │                                           │
│     │ Failing Test │                                           │
│     └──────┬───────┘                                           │
│            │                                                    │
│            ▼                                                    │
│     ┌──────────────┐      ┌──────────────┐                     │
│     │ 2. Run Test  │─────►│ 3. Verify    │                     │
│     │ (MUST FAIL)  │      │ Red          │                     │
│     └──────────────┘      └──────┬───────┘                     │
│                                  │                              │
│            ┌─────────────────────┘                              │
│            ▼                                                    │
│     ┌──────────────┐                                           │
│     │ 4. Write     │                                           │
│     │ Minimal Code │                                           │
│     └──────┬───────┘                                           │
│            │                                                    │
│            ▼                                                    │
│     ┌──────────────┐      ┌──────────────┐                     │
│     │ 5. Run Test  │─────►│ 6. Verify    │                     │
│     │ (MUST PASS)  │      │ Green        │                     │
│     └──────────────┘      └──────┬───────┘                     │
│                                  │                              │
│            ┌─────────────────────┘                              │
│            ▼                                                    │
│     ┌──────────────┐      ┌──────────────┐                     │
│     │ 7. Refactor  │─────►│ 8. Run ALL   │───┐                 │
│     │              │      │ Tests        │   │                 │
│     └──────────────┘      └──────────────┘   │                 │
│            ▲                                  │                 │
│            └─────────────────────────────────┘                 │
│                         Still Green                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Test Categories

#### 5.2.1 Unit Tests (Coverage Target: 95%)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // Basic functionality
    #[test]
    fn test_compress_empty_page() {
        let page = [0u8; 4096];
        let compressor = CompressorBuilder::new()
            .algorithm(Algorithm::Lz4)
            .build()
            .unwrap();
        
        let result = compressor.compress(&page).unwrap();
        assert!(result.ratio > 10.0); // High compression for zeros
    }

    // Property-based testing
    proptest! {
        #[test]
        fn roundtrip_preserves_data(data: [u8; 4096]) {
            let compressor = CompressorBuilder::new()
                .algorithm(Algorithm::Lz4)
                .build()
                .unwrap();
            
            let compressed = compressor.compress(&data).unwrap();
            let decompressed = compressor.decompress(&compressed).unwrap();
            
            prop_assert_eq!(data, decompressed);
        }

        #[test]
        fn compression_reduces_size_for_compressible_data(
            pattern: u8,
            repeat: usize in 100..4096
        ) {
            let mut page = [0u8; 4096];
            for i in 0..repeat {
                page[i] = pattern;
            }
            
            let compressor = CompressorBuilder::new()
                .algorithm(Algorithm::Lz4)
                .build()
                .unwrap();
            
            let compressed = compressor.compress(&page).unwrap();
            prop_assert!(compressed.data.len() < 4096);
        }
    }
}
```

#### 5.2.2 Integration Tests

```rust
// tests/integration/kernel_comparison.rs

#[test]
fn output_matches_kernel_lz4() {
    // Load kernel LZ4 implementation via FFI
    let kernel_lz4 = unsafe { load_kernel_lz4() };
    
    let test_data = load_silesia_corpus();
    
    for chunk in test_data.chunks(4096) {
        let mut page = [0u8; 4096];
        page[..chunk.len()].copy_from_slice(chunk);
        
        let kernel_compressed = kernel_lz4.compress(&page);
        let trueno_compressed = trueno_compress(&page);
        
        // Decompression must produce identical output
        // (Compressed bytes may differ due to implementation choices)
        let kernel_decompressed = kernel_lz4.decompress(&trueno_compressed);
        let trueno_decompressed = trueno_decompress(&kernel_compressed);
        
        assert_eq!(page, kernel_decompressed);
        assert_eq!(page, trueno_decompressed);
    }
}
```

#### 5.2.3 Fuzz Tests

```rust
// tests/fuzz/fuzz_targets/compress.rs

#![no_main]
use libfuzzer_sys::fuzz_target;
use trueno_zram_core::*;

fuzz_target!(|data: &[u8]| {
    if data.len() != 4096 {
        return;
    }
    
    let page: [u8; 4096] = data.try_into().unwrap();
    let compressor = CompressorBuilder::new()
        .algorithm(Algorithm::Lz4)
        .build()
        .unwrap();
    
    if let Ok(compressed) = compressor.compress(&page) {
        // Must not panic
        let _ = compressor.decompress(&compressed);
    }
});
```

#### 5.2.4 Mutation Tests (Score Target: 80%)

Using `cargo-mutants`:

```bash
# Run mutation testing
cargo mutants --package trueno-zram-core

# Expected output:
# Mutations: 150
# Killed: 120 (80%)
# Survived: 25 (17%)
# Timeout: 5 (3%)
```

#### 5.2.5 Benchmark Tests

```rust
// benches/compression.rs

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

fn benchmark_lz4_compression(c: &mut Criterion) {
    let silesia = load_silesia_corpus();
    
    let mut group = c.benchmark_group("LZ4 Compression");
    group.throughput(Throughput::Bytes(silesia.len() as u64));
    
    // Kernel baseline
    group.bench_function("kernel_lz4", |b| {
        b.iter(|| {
            for chunk in silesia.chunks(4096) {
                kernel_lz4_compress(chunk);
            }
        })
    });
    
    // trueno scalar
    group.bench_function("trueno_scalar", |b| {
        let compressor = CompressorBuilder::new()
            .algorithm(Algorithm::Lz4)
            .prefer_backend(SimdBackend::Scalar)
            .build()
            .unwrap();
        
        b.iter(|| {
            for chunk in silesia.chunks(4096) {
                compressor.compress(chunk);
            }
        })
    });
    
    // trueno AVX2
    group.bench_function("trueno_avx2", |b| {
        let compressor = CompressorBuilder::new()
            .algorithm(Algorithm::Lz4)
            .prefer_backend(SimdBackend::Avx2)
            .build()
            .unwrap();
        
        b.iter(|| {
            for chunk in silesia.chunks(4096) {
                compressor.compress(chunk);
            }
        })
    });
    
    group.finish();
}

criterion_group!(benches, benchmark_lz4_compression);
criterion_main!(benches);
```

### 5.3 Test Pyramid

```
                    ╱╲
                   ╱  ╲
                  ╱ E2E╲         ← 5% (System tests)
                 ╱──────╲
                ╱        ╲
               ╱Integration╲     ← 15% (Cross-crate tests)
              ╱────────────╲
             ╱              ╲
            ╱   Unit Tests   ╲   ← 80% (Function-level)
           ╱──────────────────╲
```

---

## 6. PMAT Quality Framework

### 6.1 Process Maturity Assessment

trueno-zram targets **PMAT Level 4: Managed & Measured**.

| Level | Name | Characteristics | Status |
|-------|------|-----------------|--------|
| 1 | Initial | Ad-hoc, chaotic | ✗ |
| 2 | Repeatable | Basic project management | ✗ |
| 3 | Defined | Documented standards | ✗ |
| **4** | **Managed** | **Quantitative measurement** | **Target** |
| 5 | Optimizing | Continuous improvement | Future |

### 6.2 Key Process Areas (KPAs)

#### 6.2.1 Requirements Management

| Metric | Target | Measurement |
|--------|--------|-------------|
| Requirements traceability | 100% | Each test links to requirement |
| Change request turnaround | <48h | Time from request to decision |
| Requirement volatility | <10% | Changes per sprint |

#### 6.2.2 Project Planning

| Metric | Target | Measurement |
|--------|--------|-------------|
| Estimation accuracy | ±20% | Planned vs actual effort |
| Milestone hit rate | >90% | On-time deliveries |
| Risk identification | >80% | Risks identified before impact |

#### 6.2.3 Quality Assurance

| Metric | Target | Measurement |
|--------|--------|-------------|
| Code coverage | >95% | lcov report |
| Mutation score | >80% | cargo-mutants |
| Defect density | <0.5/KLOC | Bugs per 1000 lines |
| MTTR | <4h | Mean time to resolve |

#### 6.2.4 Configuration Management

| Metric | Target | Measurement |
|--------|--------|-------------|
| Build reproducibility | 100% | cargo build --locked |
| Dependency freshness | <30 days | Time since last audit |
| Security vulnerabilities | 0 critical | cargo audit |

### 6.3 Quality Gates

```
┌─────────────────────────────────────────────────────────────────┐
│                      Quality Gate Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Gate 1: Commit         Gate 2: PR            Gate 3: Release  │
│  ┌─────────────────┐    ┌─────────────────┐   ┌──────────────┐ │
│  │ □ cargo fmt     │    │ □ All Gate 1    │   │ □ All Gate 2 │ │
│  │ □ cargo clippy  │    │ □ Coverage >95% │   │ □ Benchmarks │ │
│  │ □ cargo test    │    │ □ Mutation >80% │   │ □ Changelog  │ │
│  │ □ cargo doc     │    │ □ No warnings   │   │ □ Version    │ │
│  │                 │    │ □ 2 approvals   │   │ □ Tag signed │ │
│  └─────────────────┘    └─────────────────┘   └──────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Project Enforcement

### 7.1 Automated Enforcement

#### 7.1.1 Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: cargo-fmt
        name: cargo fmt
        entry: cargo fmt --all -- --check
        language: system
        types: [rust]
        pass_filenames: false
        
      - id: cargo-clippy
        name: cargo clippy
        entry: cargo clippy --all-targets --all-features -- -D warnings
        language: system
        types: [rust]
        pass_filenames: false
        
      - id: cargo-test
        name: cargo test
        entry: cargo test --all-features
        language: system
        types: [rust]
        pass_filenames: false
        
      - id: shellcheck
        name: shellcheck
        entry: shellcheck
        language: system
        types: [shell]
```

#### 7.1.2 CI Pipeline

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

env:
  CARGO_TERM_COLOR: always
  RUSTFLAGS: "-D warnings"

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy
      
      - name: Format check
        run: cargo fmt --all -- --check
      
      - name: Clippy
        run: cargo clippy --all-targets --all-features
      
      - name: Build
        run: cargo build --all-features
      
      - name: Test
        run: cargo test --all-features
      
      - name: Doc
        run: cargo doc --no-deps

  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@nightly
        with:
          components: llvm-tools-preview
      
      - name: Install cargo-llvm-cov
        uses: taiki-e/install-action@cargo-llvm-cov
      
      - name: Coverage
        run: cargo llvm-cov --all-features --lcov --output-path lcov.info
      
      - name: Check coverage threshold
        run: |
          COVERAGE=$(cargo llvm-cov --all-features --json | jq '.data[0].totals.lines.percent')
          if (( $(echo "$COVERAGE < 95" | bc -l) )); then
            echo "Coverage $COVERAGE% is below 95% threshold"
            exit 1
          fi

  mutation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      
      - name: Install cargo-mutants
        run: cargo install cargo-mutants
      
      - name: Mutation testing
        run: |
          cargo mutants --package trueno-zram-core --json > mutants.json
          SCORE=$(jq '.summary.mutation_score' mutants.json)
          if (( $(echo "$SCORE < 0.80" | bc -l) )); then
            echo "Mutation score $SCORE is below 80% threshold"
            exit 1
          fi

  benchmark:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      
      - name: Run benchmarks
        run: cargo bench --all-features -- --save-baseline pr
      
      - name: Compare to main
        run: |
          git fetch origin main
          git checkout origin/main
          cargo bench --all-features -- --save-baseline main
          git checkout -
          cargo bench --all-features -- --baseline main --compare pr
```

### 7.2 Manual Enforcement

#### 7.2.1 Code Review Checklist

```markdown
## PR Review Checklist

### Correctness
- [ ] Tests pass locally
- [ ] New code has tests
- [ ] Edge cases handled
- [ ] Error handling appropriate

### Performance
- [ ] No unnecessary allocations
- [ ] SIMD paths tested
- [ ] Benchmarks included for perf changes

### Safety
- [ ] `unsafe` blocks documented
- [ ] `unsafe` blocks minimal
- [ ] No undefined behavior
- [ ] Panic-free in library code

### Style
- [ ] Follows Rust API guidelines
- [ ] Documentation complete
- [ ] No TODO/FIXME without issue link
- [ ] Meaningful commit messages
```

#### 7.2.2 Release Checklist

```markdown
## Release Checklist

### Pre-release
- [ ] All tests pass on CI
- [ ] Benchmarks show no regression
- [ ] CHANGELOG.md updated
- [ ] Version bumped in Cargo.toml
- [ ] Documentation built successfully
- [ ] Security audit clean (`cargo audit`)

### Release
- [ ] Tag created and signed
- [ ] GitHub release created
- [ ] Crates published to crates.io
- [ ] Announcement drafted

### Post-release
- [ ] Documentation deployed
- [ ] Benchmark results archived
- [ ] Downstream dependencies notified
```

---

## 8. 100-Point Popperian Falsification Checklist

### 8.1 Methodology

Following Karl Popper's philosophy of science, each checklist item represents a **falsifiable hypothesis**. The project is considered complete only if all hypotheses survive attempted falsification.

**Scoring:**
- ✅ Passed (hypothesis survived falsification attempt)
- ❌ Failed (hypothesis falsified)
- ⏳ Pending (not yet tested)

**Completion criteria:** 100/100 items must pass.

### 8.2 Core Compression (25 points)

#### LZ4 Implementation

| # | Hypothesis | Falsification Method | Status |
|---|------------|---------------------|--------|
| 1 | trueno LZ4 produces valid LZ4 frames | Decompress with reference lz4 CLI | ⏳ |
| 2 | trueno LZ4 decompresses reference LZ4 frames | Compress with lz4 CLI, decompress with trueno | ⏳ |
| 3 | LZ4 roundtrip preserves all 2^32 possible 4-byte sequences | Property test with exhaustive 4-byte patterns | ⏳ |
| 4 | LZ4 roundtrip preserves random 4KB pages | Property test with 10M random pages | ⏳ |
| 5 | LZ4 compression ratio ≥2.0x on Silesia corpus | Benchmark on standard corpus | ⏳ |
| 6 | LZ4 AVX2 produces identical output to scalar | Differential test on 10M pages | ⏳ |
| 7 | LZ4 AVX-512 produces identical output to scalar | Differential test on 10M pages | ⏳ |
| 8 | LZ4 handles incompressible data without expansion >1% | Test with random bytes | ⏳ |
| 9 | LZ4 handles all-zero pages with ratio >100x | Test with zero-filled pages | ⏳ |
| 10 | LZ4 handles pathological repeating patterns | Test with {0xAA}*4096, {0x00,0xFF}*2048 | ⏳ |

#### ZSTD Implementation

| # | Hypothesis | Falsification Method | Status |
|---|------------|---------------------|--------|
| 11 | trueno ZSTD produces valid zstd frames | Decompress with reference zstd CLI | ⏳ |
| 12 | trueno ZSTD decompresses reference zstd frames | Compress with zstd CLI, decompress with trueno | ⏳ |
| 13 | ZSTD roundtrip preserves random 4KB pages | Property test with 10M random pages | ⏳ |
| 14 | ZSTD level 3 compression ratio ≥2.5x on Silesia | Benchmark on standard corpus | ⏳ |
| 15 | ZSTD FSE decoder produces correct output | Compare with reference FSE implementation | ⏳ |
| 16 | ZSTD Huffman decoder produces correct output | Compare with reference Huffman implementation | ⏳ |
| 17 | ZSTD dictionary mode improves small page compression | Test with correlated page sequences | ⏳ |

#### Adaptive Selection

| # | Hypothesis | Falsification Method | Status |
|---|------------|---------------------|--------|
| 18 | Entropy calculator matches reference implementation | Compare with scipy.stats.entropy | ⏳ |
| 19 | High-entropy pages classified as incompressible | Test with random bytes | ⏳ |
| 20 | Low-entropy pages classified as highly compressible | Test with repeated patterns | ⏳ |
| 21 | Adaptive selection improves overall throughput | Benchmark mixed workload vs fixed algorithm | ⏳ |
| 22 | Adaptive selection maintains compression ratio | Compare ratio on mixed workload | ⏳ |
| 23 | Classification overhead <5% of compression time | Profile classification cost | ⏳ |
| 24 | Model predictions have >90% accuracy | Test against labeled dataset | ⏳ |
| 25 | aprender integration does not introduce latency spikes | Measure P99 latency | ⏳ |

### 8.3 Performance (25 points)

#### SIMD Speedup

| # | Hypothesis | Falsification Method | Status |
|---|------------|---------------------|--------|
| 26 | AVX2 LZ4 compression ≥40% faster than scalar | Criterion benchmark | ⏳ |
| 27 | AVX2 LZ4 decompression ≥40% faster than scalar | Criterion benchmark | ⏳ |
| 28 | AVX-512 LZ4 compression ≥60% faster than scalar | Criterion benchmark | ⏳ |
| 29 | AVX-512 LZ4 decompression ≥60% faster than scalar | Criterion benchmark | ⏳ |
| 30 | AVX2 ZSTD decompression ≥30% faster than scalar | Criterion benchmark | ⏳ |
| 31 | SIMD paths do not regress on small pages (<512 bytes) | Benchmark small pages | ⏳ |
| 32 | SIMD paths do not regress on unaligned data | Benchmark unaligned buffers | ⏳ |

#### Throughput Targets

| # | Hypothesis | Falsification Method | Status |
|---|------------|---------------------|--------|
| 33 | LZ4 compression throughput ≥3 GB/s (AVX2) | Benchmark on Silesia corpus | ⏳ |
| 34 | LZ4 decompression throughput ≥5 GB/s (AVX2) | Benchmark on Silesia corpus | ⏳ |
| 35 | ZSTD-3 compression throughput ≥500 MB/s | Benchmark on Silesia corpus | ⏳ |
| 36 | ZSTD decompression throughput ≥1.5 GB/s | Benchmark on Silesia corpus | ⏳ |
| 37 | Page compression latency P99 <100μs | Latency distribution benchmark | ⏳ |
| 38 | No throughput degradation under memory pressure | Benchmark with constrained memory | ⏳ |

#### vs Kernel Baseline

| # | Hypothesis | Falsification Method | Status |
|---|------------|---------------------|--------|
| 39 | trueno LZ4 ≥ kernel LZ4 compression throughput | Compare with /lib/modules lz4 | ⏳ |
| 40 | trueno LZ4 ≥ kernel LZ4 decompression throughput | Compare with /lib/modules lz4 | ⏳ |
| 41 | trueno ZSTD ≥ kernel ZSTD compression throughput | Compare with /lib/modules zstd | ⏳ |
| 42 | trueno ZSTD ≥ kernel ZSTD decompression throughput | Compare with /lib/modules zstd | ⏳ |
| 43 | trueno maintains advantage under concurrent load | Multi-threaded benchmark | ⏳ |

#### Memory Efficiency

| # | Hypothesis | Falsification Method | Status |
|---|------------|---------------------|--------|
| 44 | Compression uses ≤64KB working memory | Memory profiling | ⏳ |
| 45 | No memory leaks in 24-hour stress test | Valgrind/AddressSanitizer | ⏳ |
| 46 | Peak memory <2x working set | Memory profiling | ⏳ |
| 47 | No allocation in hot path after warmup | Allocation profiling | ⏳ |
| 48 | Stack usage <32KB per compression call | Stack profiling | ⏳ |
| 49 | Decompression is allocation-free | Static analysis + runtime check | ⏳ |
| 50 | Thread-local state does not leak between calls | Concurrent test with different data | ⏳ |

### 8.4 Safety & Correctness (25 points)

#### Memory Safety

| # | Hypothesis | Falsification Method | Status |
|---|------------|---------------------|--------|
| 51 | No buffer overflows in compression | Fuzz testing (1B iterations) | ⏳ |
| 52 | No buffer overflows in decompression | Fuzz testing (1B iterations) | ⏳ |
| 53 | No use-after-free | AddressSanitizer | ⏳ |
| 54 | No double-free | AddressSanitizer | ⏳ |
| 55 | No data races | ThreadSanitizer | ⏳ |
| 56 | No undefined behavior in unsafe blocks | Miri | ⏳ |
| 57 | All unsafe blocks have safety comments | Static analysis | ⏳ |
| 58 | Unsafe code minimized (<5% of codebase) | Line count analysis | ⏳ |

#### Error Handling

| # | Hypothesis | Falsification Method | Status |
|---|------------|---------------------|--------|
| 59 | Corrupted compressed data returns Error, not panic | Fuzz with invalid frames | ⏳ |
| 60 | Truncated frames return Error | Test with truncated data | ⏳ |
| 61 | Invalid magic bytes return Error | Test with wrong header | ⏳ |
| 62 | Excessive output size returns Error | Test with decompression bomb | ⏳ |
| 63 | All error types implement std::error::Error | Static analysis | ⏳ |
| 64 | Error messages are actionable | Manual review | ⏳ |
| 65 | No panics in library code | #![deny(clippy::panic)] | ⏳ |

#### API Correctness

| # | Hypothesis | Falsification Method | Status |
|---|------------|---------------------|--------|
| 66 | Public API is #[deny(missing_docs)] compliant | Cargo doc | ⏳ |
| 67 | All public types implement Debug | Static analysis | ⏳ |
| 68 | All public types implement Clone where sensible | API review | ⏳ |
| 69 | Builder pattern validates inputs | Test invalid configurations | ⏳ |
| 70 | Thread-safe types are Send + Sync | Compile-time check | ⏳ |
| 71 | API follows Rust API guidelines | Manual review | ⏳ |
| 72 | Semver compatibility maintained | cargo-semver-checks | ⏳ |
| 73 | MSRV (1.70) is documented and tested | CI on MSRV | ⏳ |
| 74 | All dependencies are audited | cargo audit | ⏳ |
| 75 | No yanked dependencies | cargo deny | ⏳ |

### 8.5 Integration (15 points)

#### systemd Integration

| # | Hypothesis | Falsification Method | Status |
|---|------------|---------------------|--------|
| 76 | Generator creates valid systemd units | systemd-analyze verify | ⏳ |
| 77 | Units start successfully on boot | Integration test in VM | ⏳ |
| 78 | Configuration changes apply without reboot | Test config reload | ⏳ |
| 79 | Invalid config produces helpful error | Test malformed config | ⏳ |
| 80 | Generator idempotent on repeated runs | Run 3x, compare output | ⏳ |

#### CLI Compatibility

| # | Hypothesis | Falsification Method | Status |
|---|------------|---------------------|--------|
| 81 | CLI parses all zramctl options | Test all documented flags | ⏳ |
| 82 | CLI output format matches zramctl | Diff output | ⏳ |
| 83 | CLI returns correct exit codes | Test success/failure cases | ⏳ |
| 84 | CLI handles missing zram module gracefully | Test without module | ⏳ |
| 85 | CLI respects $NO_COLOR | Test with env var | ⏳ |

#### bashrs Integration

| # | Hypothesis | Falsification Method | Status |
|---|------------|---------------------|--------|
| 86 | setup.sh passes shellcheck | shellcheck --severity=warning | ⏳ |
| 87 | setup.sh is idempotent | Run 3x, verify state | ⏳ |
| 88 | setup.sh works on Ubuntu 22.04 | Integration test | ⏳ |
| 89 | setup.sh works on Ubuntu 24.04 | Integration test | ⏳ |
| 90 | setup.sh has --dry-run mode | Test dry run | ⏳ |

### 8.6 Documentation & Process (10 points)

#### Documentation

| # | Hypothesis | Falsification Method | Status |
|---|------------|---------------------|--------|
| 91 | README has working quick-start example | Follow instructions | ⏳ |
| 92 | API docs have examples for all public items | cargo doc coverage | ⏳ |
| 93 | Architecture doc matches implementation | Manual review | ⏳ |
| 94 | Benchmark methodology is reproducible | Independent reproduction | ⏳ |
| 95 | CHANGELOG follows Keep a Changelog format | Lint check | ⏳ |

#### Process

| # | Hypothesis | Falsification Method | Status |
|---|------------|---------------------|--------|
| 96 | CI passes on all PRs | GitHub Actions history | ⏳ |
| 97 | Coverage does not decrease on PRs | Coverage diff check | ⏳ |
| 98 | Benchmarks run on all PRs | CI logs | ⏳ |
| 99 | Release process automated | GitHub Actions release | ⏳ |
| 100 | All issues have response <48h | Issue tracker audit | ⏳ |

### 8.3 Scoring Summary

```
┌─────────────────────────────────────────────────────────────────┐
│              Popperian Falsification Score                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Category                  Points    Passed    Score            │
│  ─────────────────────────────────────────────────────          │
│  Core Compression            25        ⏳        /25            │
│  Performance                 25        ⏳        /25            │
│  Safety & Correctness        25        ⏳        /25            │
│  Integration                 15        ⏳        /15            │
│  Documentation & Process     10        ⏳        /10            │
│  ─────────────────────────────────────────────────────          │
│  TOTAL                      100        ⏳        /100           │
│                                                                 │
│  Status: IN PROGRESS                                            │
│  Completion: Requires 100/100 for v1.0.0 release                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Implementation Roadmap

### 9.1 Phase 1: Foundation (Weeks 1-4)

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: Foundation                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Week 1: Project Setup                                           │
│ ├── Repository structure                                        │
│ ├── CI/CD pipeline                                             │
│ ├── Pre-commit hooks                                           │
│ └── Benchmark infrastructure                                    │
│                                                                 │
│ Week 2: Scalar LZ4                                              │
│ ├── LZ4 compression (scalar)                                   │
│ ├── LZ4 decompression (scalar)                                 │
│ ├── Property tests                                             │
│ └── Fuzz testing harness                                       │
│                                                                 │
│ Week 3: Scalar ZSTD                                             │
│ ├── ZSTD frame parsing                                         │
│ ├── FSE decoder (scalar)                                       │
│ ├── Huffman decoder (scalar)                                   │
│ └── Integration tests                                          │
│                                                                 │
│ Week 4: Baseline Benchmarks                                     │
│ ├── Silesia corpus integration                                 │
│ ├── Kernel comparison tests                                    │
│ ├── Criterion benchmarks                                       │
│ └── Performance baseline documentation                         │
│                                                                 │
│ Deliverables:                                                   │
│ □ Working scalar compression                                    │
│ □ Test coverage >80%                                           │
│ □ Benchmark baseline established                               │
│ □ Checklist items 1-17 passing                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 Phase 2: SIMD Optimization (Weeks 5-8)

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: SIMD Optimization                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Week 5: AVX2 LZ4                                                │
│ ├── CPU feature detection                                      │
│ ├── Runtime dispatch                                           │
│ ├── LZ4 compression (AVX2)                                     │
│ └── LZ4 decompression (AVX2)                                   │
│                                                                 │
│ Week 6: AVX2 ZSTD                                               │
│ ├── SIMD FSE decoder                                           │
│ ├── SIMD Huffman decoder                                       │
│ └── Match copy optimization                                    │
│                                                                 │
│ Week 7: AVX-512 & NEON                                          │
│ ├── AVX-512 implementations                                    │
│ ├── ARM NEON implementations                                   │
│ └── Cross-platform testing                                     │
│                                                                 │
│ Week 8: Performance Validation                                  │
│ ├── Benchmark all SIMD paths                                   │
│ ├── Verify 40%+ speedup hypothesis                             │
│ ├── Latency distribution analysis                              │
│ └── Memory profiling                                           │
│                                                                 │
│ Deliverables:                                                   │
│ □ SIMD implementations complete                                 │
│ □ 40%+ speedup verified                                        │
│ □ Test coverage >90%                                           │
│ □ Checklist items 26-50 passing                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.3 Phase 3: Integration (Weeks 9-12)

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 3: Integration                                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Week 9: Adaptive Selection                                      │
│ ├── Entropy calculator                                         │
│ ├── Page classifier                                            │
│ ├── aprender model integration                                 │
│ └── Adaptive throughput validation                             │
│                                                                 │
│ Week 10: systemd Generator                                      │
│ ├── Configuration parser                                       │
│ ├── Unit file generation                                       │
│ ├── Boot integration testing                                   │
│ └── Error handling                                             │
│                                                                 │
│ Week 11: CLI Tool                                               │
│ ├── Command structure                                          │
│ ├── zramctl compatibility                                      │
│ ├── Status/monitoring commands                                 │
│ └── Benchmark command                                          │
│                                                                 │
│ Week 12: bashrs Scripts                                         │
│ ├── setup.sh implementation                                    │
│ ├── Cross-distro testing                                       │
│ ├── Idempotency verification                                   │
│ └── Documentation                                              │
│                                                                 │
│ Deliverables:                                                   │
│ □ Full system integration                                       │
│ □ CLI feature complete                                         │
│ □ Setup automation working                                     │
│ □ Checklist items 76-90 passing                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.4 Phase 4: Release (Weeks 13-14)

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 4: Release                                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Week 13: Hardening                                              │
│ ├── Extended fuzz testing (1B iterations)                      │
│ ├── Miri undefined behavior check                              │
│ ├── Security audit                                             │
│ ├── Mutation testing >80%                                      │
│ └── All remaining checklist items                              │
│                                                                 │
│ Week 14: Release                                                │
│ ├── CHANGELOG finalization                                     │
│ ├── Version tagging                                            │
│ ├── crates.io publication                                      │
│ ├── Documentation deployment                                   │
│ └── Announcement                                               │
│                                                                 │
│ Deliverables:                                                   │
│ □ 100/100 checklist items passing                              │
│ □ v1.0.0 released                                              │
│ □ Documentation live                                           │
│ □ Benchmarks published                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. References

### 10.1 Academic Papers

1. Ziv, J. and Lempel, A. (1977). "A universal algorithm for sequential data compression." *IEEE Transactions on Information Theory*, 23(3), 337-343.

2. Ziv, J. and Lempel, A. (1978). "Compression of individual sequences via variable-rate coding." *IEEE Transactions on Information Theory*, 24(5), 530-536.

3. Liu, W., Mei, F., Wang, C., O'Neill, M., and Swartzlander, E.E. (2018). "Data Compression Device based on Modified LZ4 Algorithm." *IEEE Transactions on Consumer Electronics*, 64(1), 110-117.

4. Bartik, M., Ubik, S., and Kubalik, P. (2015). "LZ4 compression algorithm on FPGA." *IEEE International Conference on Electronics, Circuits, and Systems*, 179-182.

5. Zhang, J., Long, X., and Suel, T. (2016). "A General SIMD-Based Approach to Accelerating Compression Algorithms." *ACM Transactions on Information Systems*, 34(3), Article 15.

6. Lemire, D. and Boytsov, L. (2015). "Decoding billions of integers per second through vectorization." *Software: Practice and Experience*, 45(1), 1-29.

7. Schlegel, B., Gemulla, R., and Lehner, W. (2010). "Fast integer compression using SIMD instructions." *Proceedings of the Sixth International Workshop on Data Management on New Hardware*, 34-40.

8. Dube, G., et al. (2022). "SIMD Lossy Compression for Scientific Data." *arXiv:2201.04614*.

9. Matsakis, N.D. and Klock, F.S. (2014). "The Rust Language." *ACM SIGAda Ada Letters*, 34(3), 103-104.

10. Jung, R., et al. (2017). "RustBelt: Securing the Foundations of the Rust Programming Language." *Proceedings of the ACM on Programming Languages*, 2(POPL), Article 66.

### 10.2 Standards

11. Collet, Y. and Kucherawy, M. (2018). "Zstandard Compression and the application/zstd Media Type." *RFC 8478*, IETF.

12. Collet, Y. and Kucherawy, M. (2021). "Zstandard Compression and the 'application/zstd' Media Type." *RFC 8878*, IETF.

### 10.3 Technical Documentation

13. LZ4 Block Format Specification. https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md

14. LZ4 Frame Format Specification. https://github.com/lz4/lz4/blob/dev/doc/lz4_Frame_format.md

15. Linux Kernel zram Documentation. https://docs.kernel.org/admin-guide/blockdev/zram.html

16. Linux Kernel zswap Documentation. https://docs.kernel.org/vm/zswap.html

### 10.4 Software References

17. LZ4 Reference Implementation. https://github.com/lz4/lz4

18. Zstandard Reference Implementation. https://github.com/facebook/zstd

19. zram-generator (Rust). https://github.com/systemd/zram-generator

20. Silesia Compression Corpus. https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **ANS** | Asymmetric Numeral Systems - entropy coding method |
| **AVX2** | Advanced Vector Extensions 2 - 256-bit SIMD |
| **AVX-512** | Advanced Vector Extensions 512 - 512-bit SIMD |
| **Batuta Stack** | PAIML's Sovereign AI Stack (trueno + bashrs + aprender) |
| **FSE** | Finite State Entropy - tabled ANS implementation |
| **LZ4** | Lempel-Ziv 4 - fast compression algorithm |
| **LZ77** | Lempel-Ziv 1977 - dictionary compression family |
| **NEON** | ARM SIMD instruction set |
| **PMAT** | Process Maturity Assessment Tool |
| **SIMD** | Single Instruction Multiple Data |
| **Sovereign AI** | AI systems fully controlled by the user, independent of cloud providers |
| **TDD** | Test-Driven Development |
| **TPS** | Toyota Production System |
| **zram** | Linux compressed RAM block device |
| **zstd** | Zstandard compression algorithm |
| **zswap** | Linux compressed swap cache |

---

## Appendix B: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-28 | Noah Gift | Initial specification |

---

*End of Specification*
