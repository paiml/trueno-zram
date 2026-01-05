# GPU LZ4 Compression Kernel Specification

**Version**: 1.0
**Date**: 2026-01-05
**Status**: SPECIFICATION - Requires Implementation
**Priority**: P0 - Critical for 5X Speedup Target
**Crate**: `trueno-gpu` (kernel), `trueno-zram-core` (integration)
**Philosophy**: Pure Rust PTX - Full LZ4 Compression on GPU

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-05 | Batuta Team | Initial specification - defines GPU LZ4 compression requirements |

---

## Executive Summary

**CRITICAL ISSUE**: The current `Lz4WarpCompressKernel` in trueno-gpu performs **ONLY zero-page detection**, not actual LZ4 compression. This means all compression work falls back to CPU, negating GPU benefits and failing to achieve the 5X speedup target.

### Current State (BROKEN)

```
Current GPU "Compression" Pipeline:
┌─────────────┐    ┌─────────────────────┐    ┌─────────────────┐
│ Host Pages  │───▶│ GPU Zero Detection  │───▶│ CPU LZ4 Compress│
│ (H2D xfer)  │    │ (trivial kernel)    │    │ (sequential)    │
└─────────────┘    └─────────────────────┘    └─────────────────┘
                          │
                          ▼
                   Performance: ~0.08-0.16 GB/s
                   Speedup: 0.02x (SLOWER than CPU!)
```

### Target State (5X SPEEDUP)

```
Target GPU Compression Pipeline:
┌─────────────┐    ┌─────────────────────┐    ┌─────────────────┐
│ Host Pages  │───▶│ GPU LZ4 Compress    │───▶│ Host Results    │
│ (H2D xfer)  │    │ (full kernel)       │    │ (D2H xfer)      │
└─────────────┘    └─────────────────────┘    └─────────────────┘
                          │
                          ▼
                   Performance: ≥30 GB/s
                   Speedup: ≥5x over kernel ZRAM (6 GB/s)
```

### Core Thesis

> **Requirement**: The pure-Rust PTX kernel MUST implement full LZ4 block compression on GPU. Zero-page detection is a trivial optimization, not compression. Without GPU-native LZ4, the 5X speedup target is **impossible**.

---

## 1. Problem Analysis

### 1.1 Current Kernel Implementation

The existing `Lz4WarpCompressKernel` in `/home/noah/src/trueno/trueno-gpu/src/kernels/lz4.rs`:

```rust
// CURRENT STATE: Only detects zero pages, does NOT compress!
impl Lz4WarpCompressKernel {
    pub fn emit_ptx(&self) -> String {
        // ... PTX generation ...
        // Kernel does:
        // 1. Load page from global memory
        // 2. Check if all bytes are zero (warp-level OR reduction)
        // 3. Set flag indicating zero/non-zero
        // 4. DOES NOT perform LZ4 compression!
    }
}
```

**Missing Functionality**:
- No LZ4 literal/match encoding
- No hash table for match finding
- No sequence encoding (token + literals + offset + matchlen)
- No output buffer management
- No compression ratio tracking

### 1.2 Why This Matters

| Metric | Current GPU Path | CPU SIMD Path | Target GPU |
|--------|------------------|---------------|------------|
| Throughput | 0.08-0.16 GB/s | 4-5 GB/s | ≥30 GB/s |
| vs Kernel ZRAM | 0.01x (99% slower) | 0.7-0.8x | 5x |
| Bottleneck | CPU fallback | SIMD saturation | PCIe bandwidth |

### 1.3 5X PCIe Rule

GPU compression is only beneficial when:

```
T_kernel > 5 × T_transfer

Where:
- T_transfer = (data_size / PCIe_bandwidth) × 2  // H2D + D2H
- PCIe Gen4 x16: 32 GB/s theoretical, ~25 GB/s practical
- For 5000 pages (20 MB): T_transfer ≈ 1.6 ms
- Minimum T_kernel for benefit: 8 ms

Current T_kernel (zero detection only): ~0.01 ms
Required T_kernel (full LZ4): ~0.7 ms at 30 GB/s
```

---

## 2. LZ4 Algorithm Overview

### 2.1 LZ4 Block Format

LZ4 is a byte-oriented LZ77 variant optimized for decompression speed:

```
LZ4 Block Structure:
┌─────────────────────────────────────────────────────────────┐
│ Sequence 1 │ Sequence 2 │ ... │ Sequence N │ Last Literals │
└─────────────────────────────────────────────────────────────┘

Each Sequence:
┌───────────┬──────────┬────────┬────────────┐
│   Token   │ Literals │ Offset │ Match Len  │
│  (1 byte) │ (0-270+) │(2 byte)│ (0-270+)   │
└───────────┴──────────┴────────┴────────────┘

Token Byte:
┌─────────────────────────────────────────────┐
│ High 4 bits: Literal length (0-15, 15=more) │
│ Low 4 bits:  Match length (4-18, 15=more)   │
└─────────────────────────────────────────────┘
```

### 2.2 Compression Algorithm

```
LZ4 Compression Algorithm:
1. Initialize hash table (4096 entries, 2 bytes each = 8KB)
2. For each position in input:
   a. Hash 4 bytes at current position
   b. Look up hash table for potential match
   c. If match found AND match_len >= 4:
      - Emit literals since last match
      - Emit match (offset, length)
      - Update hash table
      - Advance by match_length
   d. Else:
      - Update hash table
      - Advance by 1
3. Emit remaining literals
```

### 2.3 GPU Parallelization Strategy

**Challenge**: LZ4 is inherently sequential (each match depends on previous output).

**Solution**: Page-level parallelism with warp-cooperative compression.

```
GPU Parallelization:
┌──────────────────────────────────────────────────────────────┐
│ Block 0 (128 threads = 4 warps)                               │
│ ├── Warp 0: Page 0   ──────────────────────────────────────▶ │
│ ├── Warp 1: Page 1   ──────────────────────────────────────▶ │
│ ├── Warp 2: Page 2   ──────────────────────────────────────▶ │
│ └── Warp 3: Page 3   ──────────────────────────────────────▶ │
├──────────────────────────────────────────────────────────────┤
│ Block 1 (128 threads = 4 warps)                               │
│ ├── Warp 0: Page 4   ──────────────────────────────────────▶ │
│ ...                                                           │
└──────────────────────────────────────────────────────────────┘

Within each warp (32 threads compressing 1 page):
- Thread 0: Main compression loop
- Threads 1-31: Parallel hash lookup acceleration
- Shared memory: Hash table (8KB) + page data (4KB)
```

---

## 3. PTX Kernel Specification

### 3.1 Kernel Interface

```rust
/// GPU LZ4 compression kernel
///
/// Input:
///   - pages: Device buffer containing N pages (N × 4096 bytes)
///   - page_count: Number of pages to compress
///
/// Output:
///   - compressed: Device buffer for compressed data (pre-allocated, worst case N × 4096)
///   - lengths: Device buffer for compressed lengths per page (N × u32)
///   - flags: Device buffer for status flags (N × u8)
///       - 0x00: Successfully compressed (length < 4096)
///       - 0x01: Incompressible (store original)
///       - 0x02: Zero page (special case)
///
/// Launch configuration:
///   - Grid: (page_count / 4, 1, 1)  // 4 pages per block
///   - Block: (128, 1, 1)            // 4 warps per block
///   - Shared memory: 48KB per block // Hash tables + page data
pub struct Lz4CompressKernel {
    page_size: usize,        // 4096 bytes
    hash_table_size: usize,  // 4096 entries
    min_match_len: usize,    // 4 bytes
}
```

### 3.2 PTX Structure

```ptx
// LZ4 Compression Kernel - trueno-gpu
// Pure Rust PTX generation

.version 8.0
.target sm_70
.address_size 64

// Shared memory layout per block (4 warps, 4 pages)
// Offset 0-16383:    Hash tables (4 × 4096 × 2 bytes = 32KB)
// Offset 16384-32767: Page data (4 × 4096 bytes = 16KB)
// Total: 48KB shared memory

.visible .entry lz4_compress_pages(
    .param .u64 pages_ptr,      // Input pages
    .param .u64 compressed_ptr, // Output compressed data
    .param .u64 lengths_ptr,    // Output lengths per page
    .param .u64 flags_ptr,      // Output status flags
    .param .u32 page_count      // Number of pages
) {
    // Register declarations
    .reg .pred %p<16>;
    .reg .b32 %r<64>;
    .reg .b64 %rd<32>;
    .reg .f32 %f<8>;

    // Shared memory
    .shared .align 16 .b8 smem[49152];  // 48KB

    // Calculate warp and lane IDs
    mov.u32 %r1, %tid.x;
    shr.u32 %r2, %r1, 5;        // warp_id = tid / 32
    and.b32 %r3, %r1, 31;       // lane_id = tid % 32

    // Calculate page index for this warp
    mov.u32 %r4, %ctaid.x;
    shl.b32 %r5, %r4, 2;        // block_id * 4
    add.u32 %r6, %r5, %r2;      // page_idx = block_id * 4 + warp_id

    // Bounds check
    ld.param.u32 %r7, [page_count];
    setp.ge.u32 %p1, %r6, %r7;
    @%p1 bra exit;

    // Load page to shared memory (coalesced)
    // Each thread loads 128 bytes (4096 / 32 = 128)
    // ... (page loading code)

    bar.sync 0;

    // Zero page detection (warp-level reduction)
    // ... (existing zero detection, optimized)

    // Main compression loop (thread 0 of warp)
    setp.ne.u32 %p2, %r3, 0;
    @%p2 bra wait_compress;

compress_loop:
    // Hash current 4 bytes
    // ... (hash computation)

    // Look up hash table for match
    // ... (match finding)

    // If match found, emit sequence
    // ... (sequence encoding)

    // If no match, advance
    // ... (literal handling)

    // Loop until end of page
    // ... (loop control)

wait_compress:
    // Threads 1-31 assist with parallel operations
    // ... (parallel hash lookup, memory prefetch)

    bar.sync 1;

    // Write results
    // ... (output writing)

exit:
    ret;
}
```

### 3.3 Warp-Cooperative Compression

```rust
/// Warp-cooperative LZ4 compression strategy
///
/// Problem: LZ4 compression is sequential within a page
/// Solution: Use warp cooperation for parallel operations
///
/// Thread 0 (Leader):
///   - Runs main compression loop
///   - Makes match/literal decisions
///   - Writes output sequence
///
/// Threads 1-31 (Helpers):
///   - Parallel hash table lookups (speculative)
///   - Memory prefetching
///   - Warp shuffle for data sharing
///
/// Communication via warp shuffles (no shared memory sync needed)
pub struct WarpCooperativeStrategy {
    /// Number of speculative hash lookups per iteration
    speculative_lookups: usize,  // 4-8 typical

    /// Prefetch distance in bytes
    prefetch_distance: usize,    // 64-128 bytes ahead
}
```

### 3.4 Hash Table Design

```rust
/// GPU-optimized hash table for LZ4 match finding
///
/// Requirements:
///   - 4096 entries (covers 16KB sliding window)
///   - 2 bytes per entry (position offset, max 4096)
///   - Must fit in shared memory
///   - Bank-conflict-free access pattern
///
/// Hash function: hash = ((val * 2654435761) >> 20) & 0xFFF
pub struct GpuHashTable {
    /// Entries store 12-bit position offsets
    entries: [u16; 4096],

    /// Padding for bank conflict avoidance
    /// Stride = 4097 instead of 4096
    padding: u16,
}

impl GpuHashTable {
    /// Hash 4 bytes using multiplicative hash
    #[inline]
    pub fn hash(val: u32) -> u16 {
        ((val.wrapping_mul(2654435761)) >> 20) as u16 & 0xFFF
    }

    /// Bank-conflict-free index
    #[inline]
    pub fn index(&self, hash: u16, warp_id: u32) -> u32 {
        // Offset by warp_id to avoid bank conflicts across warps
        (hash as u32 + warp_id * 4097) % (4096 * 4)
    }
}
```

---

## 4. Performance Requirements

### 4.1 Throughput Targets

| Batch Size | Min Throughput | Target Throughput | vs Kernel ZRAM |
|------------|----------------|-------------------|----------------|
| 1000 pages | 15 GB/s | 20 GB/s | 2.5-3.3x |
| 2000 pages | 20 GB/s | 30 GB/s | 3.3-5x |
| 5000 pages | 25 GB/s | 40 GB/s | 4.2-6.7x |
| 10000 pages | 30 GB/s | 50 GB/s | 5-8.3x |

### 4.2 5X PCIe Rule Compliance

```rust
/// Verify GPU path is beneficial per 5X PCIe rule
///
/// GPU beneficial when: T_kernel > 5 × T_transfer
///
/// For PCIe Gen4 x16 (~25 GB/s effective):
///   - 1000 pages (4 MB): T_transfer = 0.32 ms, min T_kernel = 1.6 ms
///   - 5000 pages (20 MB): T_transfer = 1.6 ms, min T_kernel = 8 ms
///
/// At 30 GB/s compression:
///   - 1000 pages: T_kernel = 0.13 ms (FAILS - need >1.6 ms)
///   - 5000 pages: T_kernel = 0.67 ms (FAILS - need >8 ms)
///
/// INSIGHT: GPU LZ4 is TOO FAST for PCIe overhead!
/// Must batch larger or use async pipelining.
pub fn pcie_rule_satisfied(
    batch_size: usize,
    kernel_time_ns: u64,
    transfer_time_ns: u64,
) -> bool {
    let transfer_time = transfer_time_ns * 2; // H2D + D2H
    kernel_time_ns > 5 * transfer_time
}
```

### 4.3 Memory Bandwidth Analysis

```
RTX 4090 Memory Bandwidth Analysis:
├── Peak Bandwidth: 1,008 GB/s (GDDR6X)
├── Effective Bandwidth: ~850 GB/s (with coalescing)
│
├── LZ4 Compression Memory Access Pattern:
│   ├── Read: 1× input page (4096 bytes)
│   ├── Write: ~0.5× output (2048 bytes avg with 2:1 ratio)
│   ├── Hash table: ~4× read, ~1× write per byte (scattered)
│   └── Total: ~3 bytes accessed per input byte
│
├── Arithmetic Intensity: ~0.5 ops/byte (low - memory bound)
│
└── Theoretical Max: 850 GB/s / 3 = 283 GB/s
    Realistic Target: 30-50 GB/s (hash table overhead)
```

---

## 5. Implementation Roadmap

### 5.1 Phase 1: Core Kernel (P0 - Week 1)

| ID | Task | Acceptance Criteria |
|----|------|---------------------|
| LZ4-001 | PTX hash table implementation | Bank-conflict-free, 4096 entries |
| LZ4-002 | PTX match finding | Find matches ≥4 bytes, correct offset |
| LZ4-003 | PTX sequence encoding | Valid LZ4 block format |
| LZ4-004 | PTX output buffer management | Handle worst-case expansion |
| LZ4-005 | Integration with GpuBatchCompressor | Replace CPU fallback |

### 5.2 Phase 2: Optimization (P1 - Week 2)

| ID | Task | Acceptance Criteria |
|----|------|---------------------|
| LZ4-006 | Warp-cooperative compression | Thread cooperation working |
| LZ4-007 | Speculative hash lookups | 4 parallel lookups per iteration |
| LZ4-008 | Memory prefetching | Prefetch 64 bytes ahead |
| LZ4-009 | Async pipelining | Overlap H2D/kernel/D2H |
| LZ4-010 | Performance tuning | ≥30 GB/s on RTX 4090 |

### 5.3 Phase 3: Validation (P1 - Week 3)

| ID | Task | Acceptance Criteria |
|----|------|---------------------|
| LZ4-011 | Correctness tests | All compressed data decompresses correctly |
| LZ4-012 | Edge case tests | Empty pages, incompressible, max compression |
| LZ4-013 | Benchmark suite | Compare vs CPU SIMD and kernel ZRAM |
| LZ4-014 | Integration tests | trueno-ublk end-to-end working |
| LZ4-015 | Documentation | API docs, performance guide |

---

## 6. Test-Driven Development Requirements

### 6.1 Test-First Approach

**CRITICAL**: All implementation MUST follow extreme TDD:

1. Write failing test FIRST
2. Implement minimum code to pass
3. Refactor with tests green
4. Coverage requirement: ≥95%

### 6.2 Required Test Categories

```rust
#[cfg(test)]
mod tests {
    /// 1. Unit Tests - PTX generation
    #[test]
    fn test_lz4_kernel_ptx_valid() {
        let kernel = Lz4CompressKernel::new();
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry lz4_compress_pages"));
        assert!(ptx.contains(".shared"));
        // Verify PTX syntax is valid
    }

    /// 2. Correctness Tests - Round-trip
    #[test]
    fn test_gpu_compress_decompress_roundtrip() {
        let pages = generate_test_pages(1000);
        let compressed = gpu_compress(&pages);
        let decompressed = lz4_decompress(&compressed);
        assert_eq!(pages, decompressed);
    }

    /// 3. Edge Case Tests
    #[test]
    fn test_zero_page_detection() {
        let zero_page = [0u8; PAGE_SIZE];
        let result = gpu_compress(&[zero_page]);
        assert_eq!(result.flags[0], FLAG_ZERO_PAGE);
    }

    #[test]
    fn test_incompressible_page() {
        let random_page = generate_random_page();
        let result = gpu_compress(&[random_page]);
        assert_eq!(result.flags[0], FLAG_INCOMPRESSIBLE);
    }

    /// 4. Performance Tests
    #[test]
    fn test_throughput_minimum() {
        let pages = generate_test_pages(5000);
        let start = Instant::now();
        let _ = gpu_compress(&pages);
        let elapsed = start.elapsed();
        let throughput = (5000 * PAGE_SIZE) as f64 / elapsed.as_secs_f64() / 1e9;
        assert!(throughput >= 25.0, "Throughput {throughput} GB/s < 25 GB/s minimum");
    }

    /// 5. Property-Based Tests
    #[test]
    fn test_compression_ratio_sane() {
        proptest!(|(page in any::<[u8; PAGE_SIZE]>())| {
            let result = gpu_compress(&[page]);
            // Compressed size should be <= original + small overhead
            assert!(result.lengths[0] <= PAGE_SIZE + 16);
        });
    }
}
```

### 6.3 Benchmark Requirements

```rust
/// Benchmark suite for GPU LZ4 kernel
///
/// Must run on every PR, block merge if regression >5%
use criterion::{criterion_group, criterion_main, Criterion, Throughput};

fn bench_gpu_lz4(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_lz4");

    for batch_size in [100, 500, 1000, 2000, 5000, 10000] {
        let pages = generate_mixed_pages(batch_size);
        let input_bytes = batch_size * PAGE_SIZE;

        group.throughput(Throughput::Bytes(input_bytes as u64));
        group.bench_function(
            format!("{} pages", batch_size),
            |b| b.iter(|| gpu_compress(&pages))
        );
    }

    group.finish();
}

criterion_group!(benches, bench_gpu_lz4);
criterion_main!(benches);
```

---

## 7. Integration with trueno-zram

### 7.1 GpuBatchCompressor Changes

```rust
/// Updated GpuBatchCompressor to use real GPU LZ4
impl GpuBatchCompressor {
    /// Compress batch using GPU kernel
    ///
    /// BEFORE (broken): Zero detection only, CPU fallback for compression
    /// AFTER (working): Full LZ4 compression on GPU
    pub fn compress_batch(&mut self, pages: &[[u8; PAGE_SIZE]]) -> Result<BatchResult> {
        // 1. Transfer pages H2D
        let h2d_start = Instant::now();
        self.gpu.copy_h2d(&self.page_buffer, pages)?;
        let h2d_time = h2d_start.elapsed();

        // 2. Launch LZ4 compression kernel (FULL COMPRESSION, not just zero detection!)
        let kernel_start = Instant::now();
        self.gpu.launch_lz4_compress(
            &self.page_buffer,
            &self.compressed_buffer,
            &self.lengths_buffer,
            &self.flags_buffer,
            pages.len() as u32,
        )?;
        self.gpu.synchronize()?;
        let kernel_time = kernel_start.elapsed();

        // 3. Transfer results D2H
        let d2h_start = Instant::now();
        let lengths = self.gpu.copy_d2h_lengths(&self.lengths_buffer, pages.len())?;
        let flags = self.gpu.copy_d2h_flags(&self.flags_buffer, pages.len())?;
        let compressed_data = self.gpu.copy_d2h_compressed(
            &self.compressed_buffer,
            &lengths,
        )?;
        let d2h_time = d2h_start.elapsed();

        // 4. Build result
        Ok(BatchResult {
            pages: self.build_compressed_pages(compressed_data, lengths, flags),
            h2d_time_ns: h2d_time.as_nanos() as u64,
            kernel_time_ns: kernel_time.as_nanos() as u64,
            d2h_time_ns: d2h_time.as_nanos() as u64,
            // ...
        })
    }
}
```

### 7.2 Fallback Strategy

```rust
/// Automatic fallback to CPU SIMD when GPU not beneficial
pub fn select_compression_backend(
    batch_size: usize,
    gpu_available: bool,
) -> CompressionBackend {
    if !gpu_available {
        return CompressionBackend::CpuSimd;
    }

    // GPU beneficial for large batches (amortize PCIe overhead)
    if batch_size >= GPU_MIN_BATCH_SIZE {
        CompressionBackend::Gpu
    } else {
        CompressionBackend::CpuSimd
    }
}

/// Minimum batch size for GPU benefit
/// Based on 5× PCIe rule with async pipelining
pub const GPU_MIN_BATCH_SIZE: usize = 2000;
```

---

## 8. Risk Analysis

### 8.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| PTX complexity | High | Medium | Incremental development, extensive testing |
| Hash table bank conflicts | Medium | High | Padding + swizzling strategies |
| PCIe overhead dominates | High | High | Async pipelining, larger batches |
| Compression ratio lower than CPU | Low | Medium | Same algorithm, should be identical |
| Register pressure | Medium | Medium | Careful PTX register management |

### 8.2 Alternative Approaches

If pure-Rust PTX proves too complex:

1. **nvCOMP Integration**: NVIDIA's compression library (requires CUDA SDK)
   - Pro: Optimized, well-tested
   - Con: C++ dependency, licensing concerns

2. **Parallel CPU with rayon**: Already partially implemented
   - Pro: Works now, no GPU required
   - Con: Limited by CPU cores, max ~10 GB/s

3. **wgpu Compute Shaders**: Cross-platform GPU
   - Pro: Works on AMD, Intel GPUs
   - Con: Less optimized than PTX, WebGPU overhead

---

## 9. Verification Matrix

| ID | Requirement | Test | Status |
|----|-------------|------|--------|
| REQ-001 | GPU kernel generates valid PTX | Unit test: PTX syntax | TODO |
| REQ-002 | Compressed data decompresses correctly | Round-trip test | TODO |
| REQ-003 | Zero pages detected and flagged | Unit test | EXISTING |
| REQ-004 | Incompressible pages stored raw | Edge case test | TODO |
| REQ-005 | Throughput ≥25 GB/s for 5000 pages | Benchmark | TODO |
| REQ-006 | 5× speedup over kernel ZRAM | Integration benchmark | TODO |
| REQ-007 | Test coverage ≥95% | Coverage report | TODO |
| REQ-008 | No memory leaks | Valgrind/CUDA-memcheck | TODO |
| REQ-009 | Async pipelining working | Integration test | TODO |
| REQ-010 | Fallback to CPU when beneficial | Selection test | EXISTING |

---

## 10. References

### LZ4 Algorithm

[1] Y. Collet, "LZ4 - Extremely Fast Compression," GitHub, 2024. https://github.com/lz4/lz4

[2] Y. Collet, "LZ4 Block Format Description," GitHub, 2024. https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md

### GPU Compression

[3] A. Ozsoy, M. Swany, and A. Chauhan, "Pipelined Parallel LZSS for Streaming Data Compression on GPGPUs," in *IPDPS*, 2012.

[4] NVIDIA Corporation, "nvCOMP: NVIDIA GPU Lossless Compression Library," NVIDIA Developer, 2024.

[5] R. A. Patel et al., "Parallel Lossless Data Compression on the GPU," in *Innovative Parallel Computing*, 2012.

### trueno-gpu Foundation

[6] Batuta Team, "trueno-gpu: Pure Rust First-Principles GPU Compute Specification," trueno docs, 2025.

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-05 | Batuta Team | Initial specification |

**Next Steps**:
1. Review and approve spec
2. Create test stubs (TDD-first)
3. Implement PTX kernel incrementally
4. Benchmark and optimize
5. Integrate with trueno-zram
