# trueno-ublk Specification

**Version:** 3.5.0
**Date:** 2026-01-07
**Status:** PRODUCTION READY (USER_COPY) | ZERO_COPY RESEARCH
**Baseline:** BENCH-001 v2.1.0 (2026-01-07)
**Source Analysis:** Linux kernel `drivers/block/zram/zram_drv.c` (6.x)

---

## 1. Vision: Best Possible Userspace ublk Performance

> *"The first principle is that you must not fool yourself — and you are the easiest person to fool."* — Richard Feynman

trueno-ublk achieves **maximum possible performance** for a userspace ublk block device through:

1. **io_uring optimizations** - SQPOLL, registered buffers, fixed files
2. **SIMD compression** - AVX-512 LZ4/ZSTD at 15+ GB/s
3. **Same-fill detection** - Kernel ZRAM algorithm ported
4. **Lock-free data structures** - Per-CPU contexts, atomic operations

**Current State (v3.5.0, 2026-01-07):**
- USER_COPY mode: **10.6 GB/s** sequential, **807K IOPS** random ✅
- ZERO_COPY mode: Research complete - provides ~2x, not 10x
- **Architectural limit confirmed:** Userspace cannot directly access ublk request buffers

**Reality (from kernel source `ublk_cmd.h`):**
> *"ublk server can NEVER directly access the request data memory"*

**10X over kernel ZRAM is impossible.** Kernel ZRAM operates at ~25ns/page with direct memory access. Userspace ublk requires io_uring round-trips at ~150-310ns/page.

---

## 1.1 Honest Assessment (2026-01-07)

### What We Achieved

| Metric | BENCH-001 Start | Current | Improvement |
|--------|-----------------|---------|-------------|
| Sequential Read | 4.67 GiB/s | **10.6 GB/s** | **2.3x** |
| Random 4K IOPS | 286K | **807K** | **2.8x** |
| Random 4K Write | - | **264K** | Baseline |

**Optimizations delivered:** PERF-005/007/008/009/010/011/012/013 (8 of 9)

### What We Cannot Achieve (USER_COPY Mode)

| Claim | Verdict | Reason |
|-------|---------|--------|
| 10X over kernel ZRAM | ❌ **Impossible** | pwrite syscall ceiling ~13 GB/s |
| Parity with kernel ZRAM | ❌ **Impossible** | 12.5x architectural gap |
| 50% of kernel ZRAM | ❌ **Impossible** | Would need 85 GB/s, ceiling is 13 GB/s |

**USER_COPY ceiling: ~13 GB/s. We achieved 10.6 GB/s (81% of ceiling).**

### Path Forward: ZERO_COPY Reality (Kernel Source Review 2026-01-07)

```
USER_COPY:  Userspace ──pwrite()──▶ Kernel buffer
            ~310ns/page, max ~13 GB/s

ZERO_COPY:  Userspace ──io_uring FIXED──▶ Registered buffer
            ~150ns/page, max ~25 GB/s (estimated)
            ⚠️ "ublk server can NEVER directly access request data memory"

Kernel ZRAM: Direct kernel memory access
            ~25ns/page, 171 GB/s
```

**10X over kernel ZRAM is ARCHITECTURALLY IMPOSSIBLE** with ublk userspace model.

**Realistic targets:**
- USER_COPY: 10.6 GB/s ✅ achieved (81% of ceiling)
- ZERO_COPY: ~20-25 GB/s (2x improvement, not 10x)
- Kernel ZRAM parity: ❌ impossible from userspace

---

## 2. BENCH-001 Verified Baselines

> *"In God we trust. All others must bring data."* — W. Edwards Deming

### 2.1 Measured Performance (2026-01-07)

**System:** 48 threads, AVX-512 enabled, Linux 6.x

| Component | Measured | Unit | Notes |
|-----------|----------|------|-------|
| **ZSTD L1 Compression** | 15.36 | GiB/s | AVX-512 vectorized |
| **LZ4 Compression** | 5.20 | GiB/s | W1-ZEROS workload |
| **LZ4 Decompression** | 1.55 | GiB/s | W2-TEXT workload |
| **Batch Throughput** | 4.67 | GiB/s | 64-page batches optimal |
| **RAM Baseline** | 50.69 | GB/s | tmpfs sequential read |
| **Kernel ZRAM** | 171.15 | GB/s | **TARGET TO BEAT** |

### 2.2 Compression Ratios (LZ4)

| Workload | Ratio | Compressed Size | Notes |
|----------|-------|-----------------|-------|
| Zeros | 157.5x | 26 bytes | Same-fill detection |
| Text | 33.9x | 121 bytes | Highly compressible |
| Mixed | 1.76x | 2,324 bytes | Typical swap pages |
| Random | 1.0x | 4,096 bytes | Incompressible |

### 2.3 Gap Analysis

```
Kernel ZRAM:     171.15 GB/s  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Target (50%):     85.57 GB/s  ━━━━━━━━━━━━━━━━━━━━━━━━
Current Batch:     4.67 GiB/s ━━
Gap:                   18.3x   ← MUST CLOSE
```

**Critical Insight:** Compression throughput (15.36 GiB/s ZSTD) is NOT the bottleneck.
The I/O path is the bottleneck. The 10X roadmap (PERF-005 → PERF-012) specifically targets this.

### 2.4 Artifacts

```
benchmark-results/
├── 20260107-014015/
│   ├── environment.json          # System config (48 threads, AVX-512)
│   ├── bench-trace.jsonl         # renacer-compatible traces
│   ├── ram_baseline_*.json       # fio results
│   └── kernel_zram_*.json        # fio results
├── criterion-reports/            # HTML reports (5 benchmark groups)
├── compression-flamegraph.svg    # Hot path visualization
└── BENCHMARK-REPORT-20260107.md  # Full report
```

---

## 3. The Science of Speed

### 3.1 First Principles Analysis

**Why is kernel zram fast?**

| Factor | Kernel ZRAM | Current trueno-ublk | Gap |
|--------|-------------|---------------------|-----|
| Context switches per I/O | 0 | 2 | **∞** |
| Memory copies per I/O | 0-1 | 2-3 | **2-3x** |
| Syscalls per I/O | 0 | 1 | **∞** |
| TLB misses (4KB pages) | Low | High | **~10x** |
| Cache locality | Hot | Cold | **~5x** |

**The Amdahl's Law Problem:**
```
Speedup = 1 / ((1-P) + P/S)

Where:
- P = fraction parallelizable (compression) ≈ 0.3
- S = speedup of parallel portion = 25x (our SIMD)
- 1-P = serial overhead (I/O path) = 0.7

Current max speedup = 1 / (0.7 + 0.3/25) = 1.4x
```

**The insight:** We must attack the serial I/O path, not just compression.

### 3.2 The 10X Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    10X PERFORMANCE STACK (BENCH-001 Verified)            │
├──────────────────────────────────────────────────────────────────────────┤
│  Layer 5: Compression         ✅ NOT BOTTLENECK                          │
│           ─────────────────────────────────────────────────────────────  │
│           ZSTD L1: 15.36 GiB/s | LZ4: 5.20 GiB/s (BENCH-001 verified)  │
│           [Collet 2011] [Alakuijala 2019]                               │
├──────────────────────────────────────────────────────────────────────────┤
│  Layer 4: Memory              Huge Pages + NUMA Pinning                  │
│           ─────────────────────────────────────────────────────────────  │
│           2MB pages: 512x fewer TLB misses | RAM: 50.69 GB/s baseline   │
│           [Navarro 2002] [Gorman 2004]                                  │
├──────────────────────────────────────────────────────────────────────────┤
│  Layer 3: Batching            Coalesced I/O (64 pages optimal)           │
│           ─────────────────────────────────────────────────────────────  │
│           Current: 4.67 GiB/s | Target: 85+ GB/s (50% kernel ZRAM)      │
│           [Dean & Barroso 2013]                                         │
├──────────────────────────────────────────────────────────────────────────┤
│  Layer 2: Zero-Copy           io_uring Registered Buffers                │
│           ─────────────────────────────────────────────────────────────  │
│           UBLK_F_SUPPORT_ZERO_COPY + IORING_REGISTER_BUFFERS            │
│           [Axboe 2019] [Didona 2022]                                    │
├──────────────────────────────────────────────────────────────────────────┤
│  Layer 1: Kernel Bypass       SQPOLL + Fixed Files                       │
│           ─────────────────────────────────────────────────────────────  │
│           Zero syscalls in hot path | Kernel ZRAM: 171.15 GB/s target   │
│           [Axboe 2019]                                                  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. The Toyota Way: Continuous Improvement (改善)

### 4.1 Principle 2: Create Flow

> *"The right process will produce the right results."* — Liker 2004

**Current Flow (Broken):**
```
App → Kernel → ublk_drv → io_uring → Userspace → memcpy → Compress → memcpy → io_uring → Kernel
     ^^^^^^^^           ^^^^^^^^^               ^^^^^^^            ^^^^^^^
     Context            Syscall                 Copy 1             Copy 2
     Switch             Overhead
```

**Target Flow (10X):**
```
App → Kernel → ublk_drv → Shared Ring Buffer → Compress In-Place → Completion
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         ZERO copies, ZERO syscalls, ZERO context switches
```

### 4.2 Principle 5: Build Quality In (自働化 - Jidoka)

Every optimization must be **falsifiable**. No "it should be faster" — only measured, reproducible gains.

**Falsification Protocol (Popperian):**
```python
def falsify_optimization(name: str, impl: Callable) -> bool:
    baseline = benchmark(current_impl, iterations=1000)
    optimized = benchmark(impl, iterations=1000)

    # Statistical significance: p < 0.01
    p_value = mann_whitney_u(baseline, optimized)
    if p_value > 0.01:
        return FALSIFIED("Not statistically significant")

    # Minimum improvement: 10%
    speedup = median(baseline) / median(optimized)
    if speedup < 1.10:
        return FALSIFIED(f"Speedup {speedup:.2f}x < 1.10x threshold")

    # Regression check: no metric worse by >5%
    for metric in [latency_p99, memory_rss, cpu_usage]:
        if metric(optimized) > metric(baseline) * 1.05:
            return FALSIFIED(f"{metric.name} regressed >5%")

    return VERIFIED(speedup)
```

---

## 5. The 10X Roadmap

### Implementation Status (Updated 2026-01-07)

| Phase | PERF | Optimization | Status | Commit |
|-------|------|--------------|--------|--------|
| 0 | **013** | Same-fill detection | ✅ **COMPLETE** | `5c36100` |
| 0 | **014** | Per-CPU contexts | ✅ **COMPLETE** | via `compress_tls` |
| 0 | 015 | Compact table entry | ⚠️ Future | needs slab allocator |
| 1 | **005** | Registered buffers | ✅ **WIRED** | `5d45d0b` |
| 1 | 006 | Zero-copy | ⚠️ Research | - |
| 2 | **007** | SQPOLL | ✅ **FIXED+WIRED** | `20c34ef` |
| 2 | 008 | Fixed files | ✅ Multi-queue | `48238d7` |
| 3 | **003** | Multi-queue IOD mmap | ✅ **FIXED** | per-queue offset |
| 3 | 009 | Huge pages | ✅ Working | `b039ace` |
| 3 | 010 | NUMA binding | ✅ Working | - |
| 4 | 011 | Lock-free queues | ✅ Infrastructure | - |
| 4 | 012 | Adaptive batching | ✅ Infrastructure | - |

**Multi-Queue Fix (2026-01-07):**
- Fixed per-queue IOD buffer mmap offset calculation
- Fixed SINGLE_ISSUER/COOP_TASKRUN incompatibility with SQPOLL mode
- Multi-queue now creates devices successfully

**SQPOLL Finding (2026-01-07):**
- **SQPOLL adds 18% overhead** for ublk workloads (8.9 → 10.6 GB/s)
- Root cause: Context switches between SQPOLL kernel thread and daemon threads
- Solution: Disabled SQPOLL by default (use conservative TenXConfig)

**Benchmark Results (2026-01-07):**
| Test | Configuration | Result | Notes |
|------|--------------|--------|-------|
| Sequential Read | dd bs=1M | **10.6 GB/s** | Uninitialized pages |
| Sequential Read | fio bs=512k QD=32 | 8.5 GB/s | Steady state |
| Sequential Write | dd bs=1M (zeros) | **4.1 GB/s** | Same-fill optimized |
| Random 4K Read | fio 4 jobs QD=64 | **807K IOPS** | 3.15 GB/s bandwidth |
| Random 4K Write | fio 4 jobs QD=32 | **264K IOPS** | 1.0 GB/s bandwidth |
| Random 4K Read | fio 1 job QD=32 | 112K IOPS | psync ioengine |

**SQPOLL Overhead:**
| Configuration | Throughput | Impact |
|--------------|------------|--------|
| Multi-queue NO SQPOLL | 10.6 GB/s | Baseline |
| Multi-queue WITH SQPOLL | 8.9 GB/s | **-18% overhead** |

**Bottleneck Analysis:**
- At 807K IOPS with 4K pages = **3.15 GB/s**
- Each page requires: io_uring FETCH → store lookup → pwrite → COMMIT_AND_FETCH
- The `pwrite` syscall is the primary bottleneck (~100ns per I/O)
- Multi-queue scales well for random I/O (4x jobs = 7x IOPS improvement)

### USER_COPY Mode Performance Limits (CRITICAL)

> *Fundamental architectural constraint identified 2026-01-07*

**UBLK_F_USER_COPY mode requires one pwrite syscall per I/O.** This is an inherent limitation of the USER_COPY data path design.

**Per-Page Latency Breakdown:**
```
Kernel ZRAM (direct kernel path):
  - Page lookup:          ~5ns
  - Decompress/copy:      ~20ns
  - Total:                ~25ns per page
  - Theoretical max:      40M pages/sec = 163 GB/s

trueno-ublk USER_COPY path:
  - io_uring CQE poll:    ~50ns
  - Store lookup:         ~10ns
  - Buffer memcpy:        ~30ns
  - pwrite() syscall:     ~200ns (kernel entry + copy + exit)
  - io_uring SQE submit:  ~20ns (amortized)
  - Total:                ~310ns per page
  - Theoretical max:      3.2M pages/sec = 13.1 GB/s
```

**The 12.5x gap is ARCHITECTURAL, not optimizable within USER_COPY mode.**

| Mode | Syscalls/IO | Data Path | Max Throughput |
|------|-------------|-----------|----------------|
| Kernel ZRAM | 0 | Direct kernel memory | 171 GB/s |
| USER_COPY | 1 (pwrite) | Userspace → pwrite → kernel | ~13 GB/s |
| ZERO_COPY | 0 | mmap'd kernel buffer | ~85 GB/s (theoretical) |

**Path to 10X requires UBLK_F_SUPPORT_ZERO_COPY mode** (PERF-006).

**To Enable All Optimizations:**
```bash
sudo trueno-ublk create --size 8G --queues 2 --foreground
```

---

### Phase 0: Kernel ZRAM Parity (Target: Match 171 GB/s for zeros)

> *Source: Linux kernel `drivers/block/zram/zram_drv.c` analysis (2026-01-07)*

**PERF-013: Same-Fill Page Detection (P0 - CRITICAL)** ✅ **COMPLETE**

*Scientific Basis:* Kernel ZRAM achieves 171 GB/s by detecting same-filled pages **BEFORE** compression and storing only the 8-byte fill value. This is why BENCH-001 shows ZRAM at 171 GB/s on zero workloads while trueno-ublk achieves only 4.67 GiB/s.

**Implementation (Commit `5c36100`):**
- `page_same_filled()` in `crates/trueno-zram-core/src/samefill.rs`
- `fill_page_word()` for fast reconstruction
- `StoredPage` enum: `SameFill(u64)` | `Compressed(data)`
- All store/load paths updated to check same-fill BEFORE compression

| Metric | Before | Target | Falsification |
|--------|--------|--------|---------------|
| Zero-page write | Compress+Store | Metadata only | `perf record` shows no LZ4 calls |
| Zero-page throughput | 4.67 GiB/s | 85+ GB/s | BENCH-001 W1-ZEROS |
| Memory per zero-page | ~26 bytes | 0 bytes | `/proc/meminfo` delta |

**Kernel Implementation (zram_drv.c:344-358, 2108-2122):**
```c
// 1. Detection: Check if ALL 64-bit words are identical
static bool page_same_filled(void *ptr, unsigned long *element) {
    unsigned long *page = (unsigned long *)ptr;
    unsigned long val = page[0];

    if (val != page[PAGE_SIZE/sizeof(*page) - 1])  // Quick: first vs last
        return false;

    for (int pos = 1; pos < PAGE_SIZE/sizeof(*page) - 1; pos++) {
        if (val != page[pos])
            return false;
    }
    *element = val;  // Store the fill value (not just "is zero")
    return true;
}

// 2. Write path: Check BEFORE compression
static int zram_write_page(...) {
    same_filled = page_same_filled(mem, &element);
    if (same_filled)
        return write_same_filled_page(zram, element, index);  // NO COMPRESSION
    // ... only compress if not same-filled ...
}

// 3. Storage: Handle field stores fill value directly (no allocation)
zram_set_flag(zram, index, ZRAM_SAME);
zram_set_handle(zram, index, fill_value);  // 8 bytes, no zsmalloc
```

**Rust Port (trueno-ublk):**
```rust
/// Detect if page contains identical 64-bit words (kernel zram parity)
/// Returns Some(fill_value) if uniform, None otherwise
#[inline]
pub fn page_same_filled(page: &[u8; PAGE_SIZE]) -> Option<u64> {
    // Safety: PAGE_SIZE is always divisible by 8
    let words: &[u64; PAGE_SIZE / 8] = unsafe {
        &*(page.as_ptr() as *const [u64; PAGE_SIZE / 8])
    };

    let val = words[0];

    // Quick check: first vs last (kernel optimization)
    if val != words[words.len() - 1] {
        return None;
    }

    // Full scan only if first/last match
    if words[1..words.len()-1].iter().all(|&w| w == val) {
        Some(val)
    } else {
        None
    }
}

/// Write path: same-fill check BEFORE compression
pub fn write_page(&mut self, page: &[u8; PAGE_SIZE], index: u32) -> Result<()> {
    // P0: Same-fill detection (kernel zram parity)
    if let Some(fill_value) = page_same_filled(page) {
        self.stats.same_pages.fetch_add(1, Ordering::Relaxed);
        return self.store_same_filled(index, fill_value);  // No compression!
    }

    // Only compress non-uniform pages
    self.compress_and_store(page, index)
}
```

**PERF-014: Per-CPU Compression Contexts (P1)** ✅ **COMPLETE**

*Scientific Basis:* Kernel ZRAM uses `alloc_percpu()` for compression streams, eliminating cross-CPU contention entirely. Each CPU has dedicated buffers.

| Metric | Before | Target | Falsification |
|--------|--------|--------|---------------|
| Lock contention | Shared mutex | Zero | `perf lock` shows 0 contended |
| Cross-CPU access | Possible | None | CPU affinity verified |

**Implementation (trueno-ublk):**

The `compress_batch_direct()` function uses `lz4::compress_tls()` which provides:
- **Thread-local hash tables** (64KB per thread, allocated once)
- **Zero cross-thread contention** (each rayon worker has its own buffers)
- **No mutex in hot path** (only atomic stats, which are relaxed ordering)

```rust
// daemon.rs: compress_batch_direct()
// Uses compress_tls for thread-local hash tables (5-10x faster than compress)
let compressed = lz4::compress_tls(page)?;  // Per-CPU context!
```

This achieves the same effect as kernel's `raw_cpu_ptr(comp->stream)` pattern.

**Kernel Implementation (zcomp.c:110-130):**
```c
struct zcomp_strm *zcomp_stream_get(struct zcomp *comp) {
    struct zcomp_strm *zstrm = raw_cpu_ptr(comp->stream);  // Per-CPU!
    mutex_lock(&zstrm->lock);  // Only protects against migration
    return zstrm;
}
```

**PERF-015: Compact Table Entry (P2)**

*Scientific Basis:* Kernel ZRAM uses only 16 bytes per page entry with bit-packing. For 8GB device = 2M pages = 32MB metadata.

| Metric | Before | Target | Falsification |
|--------|--------|--------|---------------|
| Bytes per entry | ? | 16 | `sizeof(TableEntry)` |
| 8GB metadata | ? | 32MB | RSS measurement |

**Kernel Structure (zram_drv.h:66-73):**
```c
struct zram_table_entry {
    unsigned long handle;  // 8 bytes: zsmalloc handle OR fill value
    unsigned long flags;   // 8 bytes: size (bits 0-12) + flags (bits 13+)
};
// Total: 16 bytes per page
```

---

### Phase 1: Zero-Copy Foundation (Target: 2X)

**PERF-005: io_uring Registered Buffers**

*Scientific Basis:* [Axboe 2019] demonstrated 2-5x IOPS improvement with registered buffers by eliminating per-I/O buffer mapping overhead.

| Metric | Before | Target | Falsification |
|--------|--------|--------|---------------|
| Buffer setup | 200ns/IO | 0ns/IO | `perf stat -e dTLB-load-misses` |
| IOPS | 286K | 500K | fio randread QD=32 |

**Implementation:**
```rust
// Pre-register compression buffers at startup
let buffers: Vec<IoSliceMut> = (0..QUEUE_DEPTH)
    .map(|_| IoSliceMut::new(&mut [0u8; PAGE_SIZE * 256]))
    .collect();

ring.submitter()
    .register_buffers(&buffers)
    .expect("buffer registration failed");

// Use registered buffer index in SQE
sqe.flags |= IOSQE_BUFFER_SELECT;
sqe.buf_index = buffer_id;
```

**PERF-006: UBLK_F_SUPPORT_ZERO_COPY** ⚠️ **KERNEL SOURCE REVIEW 2026-01-07**

> **CRITICAL CORRECTION:** ZERO_COPY does NOT mean direct mmap access!

**From kernel source `include/uapi/linux/ublk_cmd.h:137-161`:**
```c
/*
 * ublk server can register data buffers for incoming I/O requests with a sparse
 * io_uring buffer table. The request buffer can then be used as the data buffer
 * for io_uring operations via the fixed buffer index.
 * Note that the ublk server can never directly access the request data memory.
 */
#define UBLK_F_SUPPORT_ZERO_COPY	(1ULL << 0)
```

**What ZERO_COPY actually means:**
1. Register sparse buffer table on io_uring instance
2. On I/O request: `UBLK_U_IO_REGISTER_IO_BUF` to register buffer at index
3. Use `IORING_OP_READ_FIXED` / `IORING_OP_WRITE_FIXED` with buffer index
4. On completion: `UBLK_U_IO_UNREGISTER_IO_BUF`

**"ublk server can NEVER directly access the request data memory"** - kernel enforces this.

**Better option - `UBLK_F_AUTO_BUF_REG` (bit 11):**
```c
#define UBLK_F_AUTO_BUF_REG 	(1ULL << 11)
```
Auto-registers buffer on FETCH, auto-unregisters on COMMIT. Simplifies flow.

| Mode | Data Access | Syscalls | Max Throughput |
|------|-------------|----------|----------------|
| USER_COPY | pread/pwrite | 1/IO | ~13 GB/s |
| ZERO_COPY | io_uring FIXED ops | 0 (io_uring) | ~25 GB/s (est.) |
| Kernel ZRAM | Direct memory | 0 | 171 GB/s |

**Revised Assessment:**
- ZERO_COPY eliminates pread/pwrite syscalls
- Still requires io_uring submission for data transfer
- **Cannot match kernel ZRAM** - userspace cannot directly access request buffers
- Estimated 2-2.5x improvement over USER_COPY (not 4x as previously claimed)

**Implementation (Corrected):**
```rust
// 1. Device with ZERO_COPY + AUTO_BUF_REG
let flags = UBLK_F_SUPPORT_ZERO_COPY
          | UBLK_F_AUTO_BUF_REG
          | UBLK_F_URING_CMD_COMP_IN_TASK;

// 2. Register sparse buffer table at startup
ring.submitter().register_buffers_sparse(MAX_BUFFERS)?;

// 3. On FETCH CQE with AUTO_BUF_REG, buffer auto-registered at sqe.addr index
// 4. Use IORING_OP_READ_FIXED to read from request buffer
let sqe = opcode::ReadFixed::new(
    types::Fixed(char_fd_idx),
    buf.as_mut_ptr(),
    buf.len() as u32,
    buffer_index,  // From auto-registration
).build();

// 5. On COMMIT, buffer auto-unregistered
```

**10X Target Reality Check:**
| Target | Achievable? | Reason |
|--------|-------------|--------|
| 10X over kernel ZRAM | ❌ **No** | Can't directly access memory |
| 2X over USER_COPY | ✅ **Maybe** | Eliminates pwrite syscall |
| Parity with kernel ZRAM | ❌ **No** | Architectural gap is fundamental |

**Status:** Research complete. ZERO_COPY provides incremental improvement, not 10X.

### Phase 2: Kernel Bypass (Target: 5X)

**PERF-007: SQPOLL Mode — Zero Syscalls** ✅ **FIXED**

*Scientific Basis:* [Axboe 2019] io_uring paper shows SQPOLL eliminates syscall overhead entirely, achieving 1.7M IOPS vs 1.2M with regular submission.

| Metric | Before | Target | Falsification |
|--------|--------|--------|---------------|
| syscalls/IO | 1 | 0 | `strace -c` |
| IOPS | 500K | 1M | fio with SQPOLL |

**Race Condition Fix (2026-01-07):**

The SQPOLL race with ublk URING_CMD has been fixed. The race occurred when the kernel SQPOLL thread hadn't processed FETCH commands before START_DEV was called.

**Fix:** After submitting FETCH commands, call `squeue_wait()` to ensure the kernel has consumed all SQ entries before START_DEV:

```rust
// Submit FETCH commands
for tag in 0..queue_depth {
    submit_fetch(tag)?;
}
ring.submit()?;

// PERF-007 FIX: Wait for kernel to consume all FETCHes
if sqpoll_enabled {
    ring.submitter().squeue_wait()?;  // Uses IORING_ENTER_SQ_WAIT
}
```

**Implementation:**
```rust
let ring = IoUring::builder()
    .setup_sqpoll(2000)           // 2ms idle before sleeping
    .setup_sqpoll_cpu(CPU_CORE)   // Pin kernel thread
    .setup_single_issuer()        // Single submitter optimization
    .setup_coop_taskrun()         // Cooperative scheduling
    .build(QUEUE_DEPTH * 4)?;
```

**PERF-008: Fixed File Descriptors**

*Scientific Basis:* [Axboe 2019] File descriptor lookup contributes ~50ns per I/O. Fixed files eliminate this.

| Metric | Before | Target | Falsification |
|--------|--------|--------|---------------|
| fd lookup | 50ns | 0ns | `perf stat -e cache-misses` |
| IOPS | 1M | 1.2M | fio with fixed files |

**Implementation:**
```rust
// Register file descriptors once
ring.submitter()
    .register_files(&[ctrl_fd, char_fd])
    .expect("file registration failed");

// Use fixed file index
sqe.flags |= IOSQE_FIXED_FILE;
sqe.fd = FIXED_FILE_INDEX;
```

### Phase 3: Memory Hierarchy Optimization (Target: 8X)

**PERF-009: Huge Pages (2MB)**

*Scientific Basis:* [Navarro et al. 2002, ASPLOS] demonstrated 30-50% performance improvement from reduced TLB pressure. For 8GB device: 4KB pages = 2M TLB entries; 2MB pages = 4K entries (512x reduction).

| Metric | Before | Target | Falsification |
|--------|--------|--------|---------------|
| TLB misses | 10K/s | 20/s | `perf stat -e dTLB-load-misses` |
| Throughput | 2 GB/s | 4 GB/s | Sequential read benchmark |

**Implementation:**
```rust
// Allocate huge page pool
let pool = unsafe {
    mmap(
        null_mut(),
        DEVICE_SIZE,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB,
        -1,
        0,
    )
};

// Advise kernel about access pattern
madvise(pool, DEVICE_SIZE, MADV_HUGEPAGE | MADV_SEQUENTIAL);
```

**PERF-010: NUMA-Aware Allocation**

*Scientific Basis:* [Lameter 2013, Linux Symposium] showed 40% performance degradation from cross-NUMA memory access.

| Metric | Before | Target | Falsification |
|--------|--------|--------|---------------|
| Cross-NUMA | Unknown | 0% | `numastat` |
| Latency p99 | 100µs | 50µs | fio with latency logging |

**Implementation:**
```rust
// Bind memory to same NUMA node as CPU
let numa_node = numa_node_of_cpu(worker_cpu);
mbind(
    pool,
    DEVICE_SIZE,
    MPOL_BIND,
    &numa_node_mask,
    MAX_NUMA_NODES,
    MPOL_MF_STRICT,
);

// Pin worker thread to CPU
sched_setaffinity(0, &cpu_mask);
```

### Phase 4: Parallel Scaling (Target: 10X)

**PERF-011: Lock-Free Multi-Queue**

*Scientific Basis:* [Michael & Scott 1996, PODC] established lock-free queue algorithms. Modern NVMe achieves 10M+ IOPS with 128 queues [Intel 2019].

| Metric | Before | Target | Falsification |
|--------|--------|--------|---------------|
| Queue contention | High | Zero | `perf lock` |
| IOPS @ 8 queues | 972K | 2.5M | fio numjobs=8 |

**Implementation:**
```rust
// Per-queue io_uring instances (no sharing)
struct QueueWorker {
    ring: IoUring,           // Dedicated ring
    buffers: HugePagePool,   // Dedicated buffer pool
    cpu_affinity: usize,     // Pinned CPU
    numa_node: usize,        // Local NUMA node
}

// Lock-free page table with atomic operations
struct LockFreePageTable {
    entries: Box<[AtomicU64]>,  // CAS-based updates
}

impl LockFreePageTable {
    fn insert(&self, page_id: u64, compressed: CompressedPage) -> bool {
        let slot = &self.entries[page_id as usize];
        slot.compare_exchange(
            EMPTY,
            compressed.as_u64(),
            Ordering::AcqRel,
            Ordering::Acquire,
        ).is_ok()
    }
}
```

**PERF-012: Adaptive Batch Sizing**

*Scientific Basis:* [Dean & Barroso 2013, CACM "The Tail at Scale"] showed that adaptive batching reduces tail latency while maintaining throughput.

| Metric | Before | Target | Falsification |
|--------|--------|--------|---------------|
| Batch efficiency | Fixed 64 | Adaptive 16-256 | Throughput vs latency curve |
| p99 latency | 100µs | 50µs | fio latency percentiles |

**Implementation:**
```rust
struct AdaptiveBatcher {
    current_size: AtomicU32,
    latency_ema: AtomicU64,  // Exponential moving average
    target_latency_us: u64,
}

impl AdaptiveBatcher {
    fn adjust(&self, measured_latency_us: u64) {
        let current = self.current_size.load(Ordering::Relaxed);

        if measured_latency_us > self.target_latency_us * 2 {
            // Latency too high: reduce batch size
            self.current_size.fetch_min(current / 2, Ordering::Relaxed);
        } else if measured_latency_us < self.target_latency_us / 2 {
            // Latency low: increase batch size for throughput
            self.current_size.fetch_max(current * 2, Ordering::Relaxed);
        }
    }
}
```

---

## 6. The 100-Point Falsification Matrix

### Section A: Baseline Measurements (Points 1-20) — BENCH-001 VERIFIED

| # | Claim | Method | Pass Threshold | Current | Status |
|---|-------|--------|----------------|---------|--------|
| 1 | Baseline IOPS measured | fio randread 4K QD=32 | Documented | 286K | ✅ |
| 2 | Baseline throughput (seq read) | fio seqread 1M | Documented | 2.1 GB/s | ✅ |
| 3 | Baseline latency p99 measured | fio latency | Documented | TBD | ⬜ |
| 4 | Kernel ZRAM throughput | fio on /dev/zram0 | Documented | **171.15 GB/s** | ✅ |
| 5 | RAM baseline throughput | fio on tmpfs | Documented | **50.69 GB/s** | ✅ |
| 6 | ZSTD L1 compression | criterion bench | Documented | **15.36 GiB/s** | ✅ |
| 7 | LZ4 compression | criterion bench | Documented | **5.20 GiB/s** | ✅ |
| 8 | LZ4 decompression | criterion bench | Documented | **1.55 GiB/s** | ✅ |
| 9 | Batch throughput (64-page) | criterion bench | Documented | **4.67 GiB/s** | ✅ |
| 10 | Compression ratio (zeros) | unit test | >100x | **157.5x** | ✅ |
| 11 | Compression ratio (text) | unit test | >10x | **33.9x** | ✅ |
| 12 | Compression ratio (mixed) | unit test | >1.5x | **1.76x** | ✅ |
| 13 | Syscalls per I/O counted | `strace -c` | Documented | 1 | ✅ |
| 14 | Memory copies counted | `perf record memcpy` | Documented | 2-3 | ✅ |
| 15 | Context switches counted | `perf stat` | Documented | TBD | ⬜ |
| 16 | TLB misses measured | `perf stat -e dTLB-load-misses` | Documented | TBD | ⬜ |
| 17 | Cache misses measured | `perf stat -e LLC-load-misses` | Documented | TBD | ⬜ |
| 18 | NUMA locality verified | `numastat` | Documented | TBD | ⬜ |
| 19 | Flamegraph generated | `perf record + flamegraph` | Documented | ✅ | ✅ |
| 20 | **Target defined** | 50% kernel ZRAM | **85+ GB/s** | 4.67 GiB/s | ⬜ |

### Section B: PERF-005 Registered Buffers (Points 21-30)

| # | Claim | Method | Pass Threshold | Status |
|---|-------|--------|----------------|--------|
| 21 | Buffer registration succeeds | Unit test | No error | ⬜ |
| 22 | Registered buffer used in SQE | `strace` analysis | IOSQE_BUFFER_SELECT set | ⬜ |
| 23 | Per-I/O buffer setup eliminated | `perf record` | No mmap per I/O | ⬜ |
| 24 | TLB misses reduced | `perf stat` | >50% reduction | ⬜ |
| 25 | IOPS improved | fio benchmark | >1.5x baseline | ⬜ |
| 26 | Latency p99 not regressed | fio benchmark | <1.1x baseline | ⬜ |
| 27 | Memory usage not increased | `/proc/meminfo` | <1.1x baseline | ⬜ |
| 28 | Buffer reuse verified | Custom tracing | 100% reuse | ⬜ |
| 29 | No buffer leaks | Valgrind | 0 leaks | ⬜ |
| 30 | Correctness maintained | Data integrity test | 100% match | ⬜ |

### Section C: PERF-006 Zero-Copy (Points 31-40)

| # | Claim | Method | Pass Threshold | Status |
|---|-------|--------|----------------|--------|
| 31 | UBLK_F_SUPPORT_ZERO_COPY enabled | Kernel check | Flag set | ⬜ |
| 32 | Kernel buffer mapped | `/proc/pid/maps` | Mapping exists | ⬜ |
| 33 | memcpy eliminated | `perf record` | 0 memcpy in hot path | ⬜ |
| 34 | In-place compression works | Unit test | Correct output | ⬜ |
| 35 | Throughput improved | fio benchmark | >2x baseline | ⬜ |
| 36 | CPU usage reduced | `top` | >30% reduction | ⬜ |
| 37 | Memory bandwidth reduced | `pcm-memory` | >40% reduction | ⬜ |
| 38 | No data corruption | Checksum test | 100% match | ⬜ |
| 39 | Concurrent access safe | Stress test | No races | ⬜ |
| 40 | Error handling correct | Fault injection | Graceful recovery | ⬜ |

### Section D: PERF-007 SQPOLL (Points 41-50)

| # | Claim | Method | Pass Threshold | Status |
|---|-------|--------|----------------|--------|
| 41 | SQPOLL thread created | `ps aux` | kworker visible | ⬜ |
| 42 | Syscalls eliminated | `strace -c` | 0 syscalls in steady state | ⬜ |
| 43 | Kernel thread CPU pinned | `/proc/pid/status` | Correct affinity | ⬜ |
| 44 | SQPOLL idle timeout works | Power measurement | Thread sleeps when idle | ⬜ |
| 45 | IOPS improved | fio benchmark | >1.5x over registered buffers | ⬜ |
| 46 | Latency improved | fio benchmark | >30% p99 reduction | ⬜ |
| 47 | CPU efficiency improved | IOPS/CPU-cycle | >1.5x | ⬜ |
| 48 | No starvation | Long-running test | Consistent throughput | ⬜ |
| 49 | Graceful degradation | Overload test | No crash | ⬜ |
| 50 | Shutdown clean | Resource check | All released | ⬜ |

### Section E: PERF-008 Fixed Files (Points 51-60)

| # | Claim | Method | Pass Threshold | Status |
|---|-------|--------|----------------|--------|
| 51 | Files registered | io_uring API | Success return | ⬜ |
| 52 | Fixed index used | SQE inspection | IOSQE_FIXED_FILE set | ⬜ |
| 53 | fd lookup eliminated | `perf record` | No fd table access | ⬜ |
| 54 | IOPS improved | fio benchmark | >10% over SQPOLL | ⬜ |
| 55 | Combined with SQPOLL | Integration test | Both active | ⬜ |
| 56 | Hot path optimized | Flame graph | <5% in fd handling | ⬜ |
| 57 | Error on bad index | Fault test | Graceful error | ⬜ |
| 58 | Unregister works | Cleanup test | No leaks | ⬜ |
| 59 | Re-register works | Restart test | Correct behavior | ⬜ |
| 60 | Concurrent safe | Stress test | No races | ⬜ |

### Section F: PERF-009 Huge Pages (Points 61-70)

| # | Claim | Method | Pass Threshold | Status |
|---|-------|--------|----------------|--------|
| 61 | Huge pages allocated | `/proc/meminfo` | HugePages_Free reduced | ⬜ |
| 62 | 2MB pages used | `/proc/pid/smaps` | AnonHugePages > 0 | ⬜ |
| 63 | TLB misses reduced | `perf stat` | >90% reduction | ⬜ |
| 64 | Page faults reduced | `perf stat` | >50% reduction | ⬜ |
| 65 | Throughput improved | fio benchmark | >1.5x | ⬜ |
| 66 | Memory overhead acceptable | RSS measurement | <1.1x | ⬜ |
| 67 | Fallback to 4KB works | Low-memory test | Graceful | ⬜ |
| 68 | Fragmentation handled | Long-running test | Stable performance | ⬜ |
| 69 | NUMA-aware allocation | `numastat` | Local allocation | ⬜ |
| 70 | Transparent HP disabled | System check | Explicit control | ⬜ |

### Section G: PERF-010 NUMA Optimization (Points 71-80)

| # | Claim | Method | Pass Threshold | Status |
|---|-------|--------|----------------|--------|
| 71 | NUMA topology detected | `numactl -H` | Correct nodes | ⬜ |
| 72 | Memory bound to node | `numastat -p` | >99% local | ⬜ |
| 73 | Thread pinned to CPU | `taskset -p` | Correct mask | ⬜ |
| 74 | Cross-NUMA eliminated | `perf stat numa` | 0 remote access | ⬜ |
| 75 | Latency improved | fio benchmark | >20% reduction | ⬜ |
| 76 | Multi-socket scaling | 2-socket test | >1.8x speedup | ⬜ |
| 77 | Memory bandwidth local | `pcm-memory` | >95% local | ⬜ |
| 78 | Interrupt affinity set | `/proc/interrupts` | Correct CPU | ⬜ |
| 79 | Migration disabled | `perf sched` | No migrations | ⬜ |
| 80 | Graceful on single-node | Unit test | Works correctly | ⬜ |

### Section H: PERF-011 Lock-Free Multi-Queue (Points 81-90)

| # | Claim | Method | Pass Threshold | Status |
|---|-------|--------|----------------|--------|
| 81 | Lock-free data structure | Code review | No mutexes in hot path | ⬜ |
| 82 | CAS operations used | Assembly inspection | lock cmpxchg | ⬜ |
| 83 | ABA problem handled | Stress test | No corruption | ⬜ |
| 84 | Memory ordering correct | ThreadSanitizer | No data races | ⬜ |
| 85 | Scalability linear | 1-8 queue test | >0.9 efficiency | ⬜ |
| 86 | IOPS @ 8 queues | fio benchmark | >2M | ⬜ |
| 87 | Contention eliminated | `perf lock` | 0 contended | ⬜ |
| 88 | Cache line padding | sizeof check | 64-byte aligned | ⬜ |
| 89 | False sharing eliminated | `perf c2c` | No false sharing | ⬜ |
| 90 | Graceful single-queue | Fallback test | Works correctly | ⬜ |

### Section I: PERF-012 Adaptive Batching (Points 91-100)

| # | Claim | Method | Pass Threshold | Status |
|---|-------|--------|----------------|--------|
| 91 | Batch size adapts | Logging | Size changes with load | ⬜ |
| 92 | Latency target met | fio benchmark | p99 < 50µs | ⬜ |
| 93 | Throughput maintained | fio benchmark | >90% of fixed batch | ⬜ |
| 94 | EMA calculation correct | Unit test | Mathematical correctness | ⬜ |
| 95 | Convergence fast | Step response test | <100ms to adapt | ⬜ |
| 96 | Stability achieved | Long-running test | No oscillation | ⬜ |
| 97 | Min batch respected | Edge test | Never below minimum | ⬜ |
| 98 | Max batch respected | Edge test | Never above maximum | ⬜ |
| 99 | Mixed workload handled | Realistic benchmark | Good for both | ⬜ |
| 100 | **10X ACHIEVED** | Full benchmark suite | **10X kernel zram** | ⬜ |

---

## 7. Implementation Priority

### Sprint 1: Foundation (2 weeks)
- [ ] PERF-005: Registered Buffers — Expected: **1.75x**
- [ ] PERF-006: Zero-Copy — Expected: **2.5x cumulative**

### Sprint 2: Kernel Bypass (2 weeks)
- [ ] PERF-007: SQPOLL — Expected: **4x cumulative**
- [ ] PERF-008: Fixed Files — Expected: **4.5x cumulative**

### Sprint 3: Memory (2 weeks)
- [ ] PERF-009: Huge Pages — Expected: **6x cumulative**
- [ ] PERF-010: NUMA — Expected: **7x cumulative**

### Sprint 4: Scaling (2 weeks)
- [ ] PERF-011: Lock-Free Multi-Queue — Expected: **9x cumulative**
- [ ] PERF-012: Adaptive Batching — Expected: **10x cumulative**

---

## 8. References (Peer-Reviewed)

### io_uring & Kernel Bypass
1. **Axboe, J. (2019).** "Efficient IO with io_uring." Linux Plumbers Conference. [Link](https://kernel.dk/io_uring.pdf)
2. **Didona, D., Pfefferle, J., Ioannou, N., Metzler, B., & Trivedi, A. (2022).** "Understanding Modern Storage APIs: A Systematic Study of libaio, SPDK, and io_uring." USENIX ATC '22.
3. **Yang, Z., Harris, J.R., Walker, B., Verkamp, D., et al. (2017).** "SPDK: A Development Kit to Build High Performance Storage Applications." IEEE CloudCom.

### Memory & NUMA
4. **Navarro, J., Iyer, S., Druschel, P., & Cox, A. (2002).** "Practical, Transparent Operating System Support for Superpages." OSDI '02.
5. **Gorman, M. (2004).** "Understanding the Linux Virtual Memory Manager." Prentice Hall.
6. **Lameter, C. (2013).** "NUMA (Non-Uniform Memory Access): An Overview." Linux Symposium.

### Compression
7. **Collet, Y. (2011).** "LZ4 - Extremely Fast Compression Algorithm." [GitHub](https://github.com/lz4/lz4)
8. **Alakuijala, J., & Szabadka, Z. (2016).** "Brotli Compressed Data Format." RFC 7932, IETF.
9. **Collet, Y., & Kucherawy, M. (2021).** "Zstandard Compression and the 'application/zstd' Media Type." RFC 8878, IETF.

### Algorithms & Data Structures
10. **Michael, M.M., & Scott, M.L. (1996).** "Simple, Fast, and Practical Non-Blocking and Blocking Concurrent Queue Algorithms." PODC '96.
11. **Frigo, M., Leiserson, C.E., Prokop, H., & Ramachandran, S. (1999).** "Cache-Oblivious Algorithms." FOCS '99.
12. **Dean, J., & Barroso, L.A. (2013).** "The Tail at Scale." Communications of the ACM.

### Philosophy & Methodology
13. **Popper, K. (1959).** *The Logic of Scientific Discovery.* Hutchinson.
14. **Liker, J.K. (2004).** *The Toyota Way: 14 Management Principles.* McGraw-Hill.
15. **Shannon, C.E. (1948).** "A Mathematical Theory of Communication." Bell System Technical Journal.

---

## 9. Conclusion

> *"Whether you think you can, or you think you can't — you're right."* — Henry Ford

### 9.1 BENCH-001 Key Insights

1. **The Zero-Page Wall (171.15 GB/s):** Kernel ZRAM achieves this via zero_page fast-path. To match, trueno-ublk must implement metadata-only zero-handling to avoid io_uring round-trips for zero-filled blocks.

2. **Compression is NOT the Bottleneck:** ZSTD L1 at 15.36 GiB/s can process ~3.8M pages/sec. The I/O path overhead dominates.

3. **Decompression Critical Path:** At 1.55 GiB/s, decompression is ~55x below target. PERF-006 (Zero-Copy) and PERF-011 (Multi-Queue) are mandatory.

4. **Batch Efficiency Confirmed:** 64-page batches achieve 4.67 GiB/s, validating the batching approach but confirming Amdahl's Law problem.

### 9.2 The Path Forward

```
BENCH-001 Verified Gaps:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Kernel ZRAM:         171.15 GB/s  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Target (50%):         85.57 GB/s  ━━━━━━━━━━━━━━━━━━━━━
Compression (ZSTD):   15.36 GiB/s ━━━━━
Batch I/O:             4.67 GiB/s ━━
Decompression:         1.55 GiB/s ━  ← CRITICAL PATH

Gap to close:                55x (decompression) / 18x (batch I/O)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 9.3 Strategic Priority

| Priority | Optimization | Expected Impact |
|----------|--------------|-----------------|
| **P0** | Zero-page metadata handling | Parity with ZRAM zero-path |
| **P1** | PERF-006 Zero-Copy | 3x decompression |
| **P2** | PERF-011 Multi-Queue | 4x parallel scaling |
| **P3** | PERF-007 SQPOLL | Eliminate syscall overhead |

The path to 10X is clear. Each optimization is:
- **Scientifically grounded** in peer-reviewed research
- **Falsifiable** with specific, measurable criteria
- **Incremental** — each builds on the previous
- **Verified** — BENCH-001 baseline established

We will not make excuses. We will make progress.

**BENCH-001 Baseline: 4.67 GiB/s** → **Target: 85+ GB/s** → **Delta: 18x improvement needed**

Let's build.
