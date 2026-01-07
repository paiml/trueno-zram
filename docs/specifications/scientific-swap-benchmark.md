# Scientific Swap Technology Benchmark Specification

**Document ID:** BENCH-001
**Version:** 2.1.0
**Status:** ACTIVE
**Created:** 2026-01-06
**Updated:** 2026-01-07
**Priority:** P0 - BLOCKING RELEASE

## 1. Purpose

Provide reproducible, falsifiable performance comparison between:
1. **RAM Baseline** - Direct tmpfs memory access (theoretical maximum)
2. **Regular Swap** - Disk-backed swap partition/file (NVMe SSD)
3. **Kernel ZRAM** - Linux kernel compressed RAM swap
4. **trueno-ublk** - SIMD/GPU accelerated userspace compression via ublk

### 1.1 Design Principles

This benchmark follows the **Batuta Stack** methodology:
- **bashrs**: All shell operations validated for safety and portability
- **renacer**: Structured JSON tracing for observability (no raw println!)
- **criterion**: Statistical microbenchmarks with regression detection
- **flamegraph**: Visual profiling for hotspot identification

### 1.2 Volume Lifecycle

**CRITICAL:** All test volumes are ephemeral:
- Created fresh for each benchmark run
- Isolated from system swap (no interference)
- Torn down immediately after test completion
- No persistent state between runs

## 2. Scientific Methodology

### 2.1 Falsification Criteria

Each claim must be:
- **Reproducible**: Same results within 10% variance across 3 runs
- **Falsifiable**: Clear pass/fail thresholds defined upfront
- **Isolated**: Single variable tested at a time
- **Documented**: Full environment specification recorded

### 2.2 Null Hypothesis

**H0:** trueno-zram provides no statistically significant performance improvement over kernel ZRAM.

**Acceptance Criteria:** Reject H0 if trueno-zram shows ≥20% improvement with p < 0.05 across 3+ independent runs.

## 3. Test Environment Specification

### 3.1 Hardware Requirements

```yaml
minimum:
  cpu: x86_64 with AVX2
  ram: 32GB
  storage: NVMe SSD (for swap file baseline)

recommended:
  cpu: x86_64 with AVX-512
  ram: 64GB+
  gpu: NVIDIA RTX 4090 / A100 (for GPU tests)
```

### 3.2 Software Requirements

```yaml
kernel: >= 6.0 (ublk support)
modules:
  - zram
  - ublk_drv
tools:
  - fio >= 3.35
  - sysbench >= 1.0.20
  - stress-ng >= 0.17
```

### 3.3 Environment Isolation

```bash
# Disable CPU frequency scaling
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable THP (can cause variance)
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled

# Drop caches before each test
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches

# Disable swap before reconfiguring
sudo swapoff -a
```

### 3.4 SIMD Capability Verification (MANDATORY)

The performance delta between AVX2 and AVX-512 is often 2-3x for SIMD compression.
This MUST be logged at benchmark start.

```bash
# Log CPU SIMD capabilities
verify_simd_capabilities() {
    log_json "INFO" "Verifying SIMD capabilities"

    local flags
    flags=$(grep -m1 "^flags" /proc/cpuinfo | cut -d: -f2)

    # Check for critical SIMD extensions
    local avx2=false avx512f=false avx512bw=false

    [[ "$flags" == *"avx2"* ]] && avx2=true
    [[ "$flags" == *"avx512f"* ]] && avx512f=true
    [[ "$flags" == *"avx512bw"* ]] && avx512bw=true

    # Emit structured log
    printf '{"simd":{"avx2":%s,"avx512f":%s,"avx512bw":%s}}\n' \
        "$avx2" "$avx512f" "$avx512bw"

    # Warn if only AVX2 available
    if [[ "$avx512f" == "false" ]]; then
        log_json "WARN" "AVX-512 not available; expect ~2x lower SIMD throughput"
    fi
}
```

## 4. Test Matrix

### 4.1 Workload Types

| Workload | Description | Data Pattern |
|----------|-------------|--------------|
| W1-ZEROS | All zero pages | 100% compressible |
| W2-TEXT | Text/code data | ~70% compressible |
| W3-MIXED | 50/50 random/text | ~35% compressible |
| W4-RANDOM | Incompressible | 0% compressible |
| W5-REALWORLD | Memory allocator stress | Variable |

### 4.2 Access Patterns

| Pattern | Description | fio Parameters |
|---------|-------------|----------------|
| P1-SEQ-R | Sequential read | rw=read, bs=1M |
| P2-SEQ-W | Sequential write | rw=write, bs=1M |
| P3-RAND-R | Random read | rw=randread, bs=4k, iodepth=32 |
| P4-RAND-W | Random write | rw=randwrite, bs=4k, iodepth=32 |
| P5-MIXED | Mixed read/write | rw=randrw, bs=4k, rwmixread=70 |
| P6-BATCH | High-depth batched I/O | rw=randrw, bs=4k, iodepth=128, iodepth_batch_submit=64 |

**P6-BATCH Rationale:** This pattern specifically tests the trueno-ublk Phase 4 scaling
goals (PERF-011, PERF-012). High queue depth with batch submission exercises:
- Lock-free multi-queue contention
- Adaptive batch sizing under load
- SQPOLL kernel thread saturation

### 4.3 Full Test Matrix

Each swap technology tested against each workload × pattern combination:
- 4 technologies (RAM, NVMe, kernel zram, trueno-ublk) × 5 workloads × 6 patterns = 120 test cases
- Each test case run 3 times for statistical validity
- Total: 360 test runs
- Estimated duration: ~6 hours (30s per run × 360 runs × 2 setup overhead)

## 5. Metrics Collected

### 5.1 Primary Metrics

| Metric | Unit | Collection Method |
|--------|------|-------------------|
| Throughput | GB/s | fio bandwidth |
| IOPS | ops/s | fio iops |
| Latency P50 | μs | fio clat percentile |
| Latency P99 | μs | fio clat percentile |
| CPU Usage | % | mpstat during test |

### 5.2 Efficiency Metrics (v2.1.0)

| Metric | Unit | Collection Method | Rationale |
|--------|------|-------------------|-----------|
| IOPS/CPU-cycle | ops/cycle | fio + `perf stat -e cycles` | SQPOLL efficiency verification |
| Context Switches | count/s | `perf stat -e context-switches` | Kernel bypass validation |
| Syscalls/IO | count | `strace -c -e io_uring_enter` | Zero-syscall claim |
| CPU Efficiency | % useful | user_cycles / total_cycles | Kernel vs userspace time |

**Context Switch Monitoring (CRITICAL):**

A key claim of the 10X stack is elimination of context switches via SQPOLL and
registered buffers. This MUST be verified:

```bash
# Baseline: Measure context switches during benchmark
measure_context_switches() {
    local name="$1" duration="${2:-30}"

    perf stat -e context-switches,cpu-migrations,page-faults \
        -a -o "$RESULTS_DIR/${name}-perf-stat.txt" \
        -- sleep "$duration" &
    local perf_pid=$!

    # Run workload while perf is collecting
    run_workload "$name"

    wait "$perf_pid"

    # Parse and log
    local ctx_switches
    ctx_switches=$(grep context-switches "$RESULTS_DIR/${name}-perf-stat.txt" | awk '{print $1}')
    log_json "INFO" "Context switches for $name: $ctx_switches"
}
```

### 5.3 Secondary Metrics

| Metric | Unit | Collection Method |
|--------|------|-------------------|
| Compression Ratio | x | (orig_size / comp_size) |
| Memory Usage | MB | /proc/meminfo |
| Page Faults | count | /proc/vmstat |
| Swap I/O | MB/s | iostat |

## 6. Benchmark Implementation

### 6.1 Volume Lifecycle Management (bashrs-validated)

All volume operations MUST be validated by bashrs for safety. The benchmark
creates isolated volumes that do not interfere with system swap.

```bash
#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# bashrs-validated: DET002,SC2086,SEC001
# bench-volume-lifecycle.sh - Ephemeral volume management

set -euo pipefail

# Configuration
readonly BENCH_SIZE_GB="${BENCH_SIZE_GB:-4}"
readonly BENCH_DEVICE_ID="${BENCH_DEVICE_ID:-99}"  # High ID to avoid conflicts
readonly BENCH_ZRAM_ID="${BENCH_ZRAM_ID:-15}"      # High zram ID
readonly BENCH_SWAP_FILE="/tmp/bench-swap-${RANDOM}"
readonly BENCH_TMPFS_DIR="/tmp/bench-tmpfs-${RANDOM}"

# renacer-compatible JSON logging
log_json() {
    local level="$1" msg="$2"
    printf '{"timestamp":"%s","level":"%s","target":"bench","message":"%s"}\n' \
        "$(date -Iseconds)" "$level" "$msg"
}

# ═══════════════════════════════════════════════════════════════════════════
# VOLUME SETUP (Ephemeral - Created Fresh Each Run)
# ═══════════════════════════════════════════════════════════════════════════

setup_ram_baseline() {
    # RAM baseline via tmpfs (theoretical maximum performance)
    log_json "INFO" "Creating tmpfs RAM baseline: ${BENCH_SIZE_GB}G"
    mkdir -p "$BENCH_TMPFS_DIR"
    sudo mount -t tmpfs -o size="${BENCH_SIZE_GB}G" tmpfs "$BENCH_TMPFS_DIR"
    echo "$BENCH_TMPFS_DIR"
}

setup_nvme_swap() {
    # NVMe-backed swap file (disk baseline)
    log_json "INFO" "Creating NVMe swap file: ${BENCH_SIZE_GB}G"
    sudo fallocate -l "${BENCH_SIZE_GB}G" "$BENCH_SWAP_FILE"
    sudo chmod 600 "$BENCH_SWAP_FILE"
    sudo mkswap "$BENCH_SWAP_FILE" >/dev/null 2>&1
    echo "$BENCH_SWAP_FILE"
}

setup_kernel_zram() {
    # Kernel zram with LZ4 (kernel baseline)
    log_json "INFO" "Creating kernel zram: ${BENCH_SIZE_GB}G (ID: $BENCH_ZRAM_ID)"

    # Load zram module with specific device
    if ! grep -q "^zram" /proc/modules; then
        sudo modprobe zram num_devices=16
    fi

    # Configure fresh zram device
    echo lz4 | sudo tee "/sys/block/zram${BENCH_ZRAM_ID}/comp_algorithm" >/dev/null
    echo "${BENCH_SIZE_GB}G" | sudo tee "/sys/block/zram${BENCH_ZRAM_ID}/disksize" >/dev/null
    echo "/dev/zram${BENCH_ZRAM_ID}"
}

setup_trueno_ublk() {
    # trueno-ublk with full 10X stack
    local preset="${1:-high-perf}"  # conservative, high-perf, max-perf
    log_json "INFO" "Creating trueno-ublk: ${BENCH_SIZE_GB}G (ID: $BENCH_DEVICE_ID, preset: $preset)"

    sudo ./target/release/trueno-ublk create \
        --size "${BENCH_SIZE_GB}G" \
        --dev-id "$BENCH_DEVICE_ID" \
        --algorithm lz4 \
        "--${preset}" \
        --foreground &

    local pid=$!
    sleep 2

    # Verify device created
    if [[ ! -b "/dev/ublkb${BENCH_DEVICE_ID}" ]]; then
        log_json "ERROR" "Failed to create ublk device"
        return 1
    fi

    echo "/dev/ublkb${BENCH_DEVICE_ID}:${pid}"
}

# ═══════════════════════════════════════════════════════════════════════════
# VOLUME TEARDOWN (Guaranteed Cleanup)
# ═══════════════════════════════════════════════════════════════════════════

teardown_ram_baseline() {
    log_json "INFO" "Tearing down tmpfs RAM baseline"
    sudo umount "$BENCH_TMPFS_DIR" 2>/dev/null || true
    rmdir "$BENCH_TMPFS_DIR" 2>/dev/null || true
}

teardown_nvme_swap() {
    log_json "INFO" "Tearing down NVMe swap"
    sudo swapoff "$BENCH_SWAP_FILE" 2>/dev/null || true
    sudo rm -f "$BENCH_SWAP_FILE"
}

teardown_kernel_zram() {
    log_json "INFO" "Tearing down kernel zram"
    sudo swapoff "/dev/zram${BENCH_ZRAM_ID}" 2>/dev/null || true
    echo 1 | sudo tee "/sys/block/zram${BENCH_ZRAM_ID}/reset" >/dev/null 2>&1 || true
}

teardown_trueno_ublk() {
    local pid="$1"
    log_json "INFO" "Tearing down trueno-ublk (PID: $pid)"
    sudo swapoff "/dev/ublkb${BENCH_DEVICE_ID}" 2>/dev/null || true
    sudo ./target/release/trueno-ublk destroy "$BENCH_DEVICE_ID" 2>/dev/null || true
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

teardown_all() {
    log_json "INFO" "Cleanup: Tearing down all benchmark volumes"
    teardown_ram_baseline
    teardown_nvme_swap
    teardown_kernel_zram
    # Note: trueno_ublk requires PID, handled separately
}

# Trap for guaranteed cleanup
trap teardown_all EXIT
```

### 6.2 Profiling Integration

#### 6.2.1 renacer Structured Tracing

All benchmark operations emit structured JSON traces compatible with renacer:

```rust
// In bins/trueno-ublk/src/perf/tenx/bench.rs
use tracing::{info, instrument, span, Level};

#[derive(Debug, Serialize)]
pub struct BenchmarkTrace {
    pub benchmark_id: String,
    pub technology: Technology,
    pub workload: Workload,
    pub pattern: AccessPattern,
    pub metrics: BenchmarkMetrics,
    pub environment: EnvironmentInfo,
}

#[instrument(level = "info", skip(config))]
pub fn run_benchmark(config: &BenchConfig) -> BenchmarkResult {
    let span = span!(Level::INFO, "benchmark_run",
        technology = %config.technology,
        workload = %config.workload,
    );
    let _enter = span.enter();

    info!(target: "renacer",
        benchmark_id = %config.id,
        "Starting benchmark run"
    );

    // ... benchmark execution ...
}
```

#### 6.2.2 Criterion Statistical Benchmarks

```rust
// In bins/trueno-ublk/benches/swap_comparison.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};

fn bench_technologies(c: &mut Criterion) {
    let mut group = c.benchmark_group("swap_technologies");
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(30));

    for size in [4096, 65536, 1048576].iter() { // 4KB, 64KB, 1MB
        group.throughput(Throughput::Bytes(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("ram_baseline", size),
            size,
            |b, &size| b.iter(|| bench_ram_read(size)),
        );

        group.bench_with_input(
            BenchmarkId::new("kernel_zram", size),
            size,
            |b, &size| b.iter(|| bench_zram_read(size)),
        );

        group.bench_with_input(
            BenchmarkId::new("trueno_ublk", size),
            size,
            |b, &size| b.iter(|| bench_ublk_read(size)),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_technologies);
criterion_main!(benches);
```

#### 6.2.3 Flame Graph Generation

```bash
#!/usr/bin/env bash
# generate-flamegraph.sh - Performance profiling with flame graphs
# Reference: Gregg, B. (2016). "The Flame Graph." CACM 59(6):48-57.

set -euo pipefail

DURATION_SEC="${1:-30}"
OUTPUT_DIR="benchmark-results/flamegraphs"
mkdir -p "$OUTPUT_DIR"

# Record with perf (requires perf_event_paranoid <= 1)
record_perf() {
    local name="$1"
    log_json "INFO" "Recording perf data for: $name (${DURATION_SEC}s)"

    sudo perf record -F 99 -ag --call-graph dwarf \
        -o "$OUTPUT_DIR/${name}.perf.data" \
        -- sleep "$DURATION_SEC"
}

# Generate flame graph SVG
generate_flamegraph() {
    local name="$1"
    log_json "INFO" "Generating flame graph: $name"

    sudo perf script -i "$OUTPUT_DIR/${name}.perf.data" | \
        stackcollapse-perf.pl | \
        flamegraph.pl --title "$name" > "$OUTPUT_DIR/${name}.svg"

    # Differential flame graph if baseline exists
    if [[ -f "$OUTPUT_DIR/baseline.folded" ]]; then
        difffolded.pl "$OUTPUT_DIR/baseline.folded" "$OUTPUT_DIR/${name}.folded" | \
            flamegraph.pl --title "Diff: $name vs baseline" \
            > "$OUTPUT_DIR/${name}-diff.svg"
    fi
}

# Hotspot analysis report
analyze_hotspots() {
    local name="$1"
    log_json "INFO" "Analyzing hotspots for: $name"

    sudo perf report -i "$OUTPUT_DIR/${name}.perf.data" \
        --stdio --no-children --percent-limit 1 \
        > "$OUTPUT_DIR/${name}-hotspots.txt"
}
```

### 6.3 Benchmark Runner (bashrs-validated)

```bash
#!/usr/bin/env bash
# scientific-swap-benchmark.sh - Main benchmark orchestrator
# bashrs-validated: DET002,SC2086,SEC001

set -euo pipefail

source "$(dirname "$0")/bench-volume-lifecycle.sh"

readonly BENCHMARK_ID="BENCH-001-$(date +%Y%m%d-%H%M%S)"
readonly RESULTS_DIR="benchmark-results/${BENCHMARK_ID}"

main() {
    log_json "INFO" "Starting benchmark: $BENCHMARK_ID"
    mkdir -p "$RESULTS_DIR"

    # Save environment
    save_environment > "$RESULTS_DIR/environment.json"

    # Run benchmarks in sequence (isolated volumes)
    run_ram_baseline
    run_nvme_swap
    run_kernel_zram
    run_trueno_ublk "conservative"
    run_trueno_ublk "high-perf"
    run_trueno_ublk "max-perf"

    # Generate comparative report
    generate_report

    log_json "INFO" "Benchmark complete: $RESULTS_DIR"
}

run_ram_baseline() {
    log_json "INFO" "=== RAM Baseline Benchmark ==="
    local tmpfs_dir
    tmpfs_dir=$(setup_ram_baseline)

    # Direct I/O to tmpfs (theoretical max)
    fio --name=ram_baseline \
        --directory="$tmpfs_dir" \
        --filename=test.dat \
        --size=1G \
        --rw=randrw \
        --bs=4k \
        --iodepth=32 \
        --numjobs=4 \
        --runtime=30 \
        --group_reporting \
        --output-format=json \
        > "$RESULTS_DIR/ram_baseline.json"

    teardown_ram_baseline
}

# ... similar functions for other technologies ...

main "$@"
```

## 7. Statistical Analysis

### 7.1 Data Processing

```python
# Required: 3+ runs per test case
# Outlier removal: IQR method (remove > 1.5*IQR from Q1/Q3)
# Central tendency: Median (robust to outliers)
# Variance: MAD (Median Absolute Deviation)
```

### 7.2 Significance Testing

- Mann-Whitney U test for non-parametric comparison
- p < 0.05 required to claim significant difference
- Effect size: Cohen's d ≥ 0.8 for "large" improvement

## 8. Falsification Tests

### 8.1 Claim: "12x faster than kernel ZRAM"

**Test:** P1-SEQ-R with W2-TEXT workload

**Falsification Threshold:**
- PASS: trueno-zram ≥ 5x kernel ZRAM median throughput
- FAIL: < 5x improvement

**Rationale:** Original claim of 12x may be workload-specific. 5x is conservative baseline.

### 8.2 Claim: "228K IOPS random read"

**Test:** P3-RAND-R with W3-MIXED workload

**Falsification Threshold:**
- PASS: ≥ 180K IOPS (20% variance allowed)
- FAIL: < 180K IOPS

### 8.3 Claim: "3.7x compression ratio"

**Test:** W3-MIXED workload, measure actual ratio

**Falsification Threshold:**
- PASS: ratio ≥ 2.5x
- FAIL: ratio < 2.5x

**Note:** Ratio depends heavily on workload. 3.7x observed on specific mixed data.

### 8.4 Claim: "No swap deadlock (DT-007)"

**Test:** W5-REALWORLD with memory pressure to 95% RAM

**Falsification Threshold:**
- PASS: Daemon State=S, VmLck > 100MB, responsive after 60s stress
- FAIL: Daemon State=D OR VmLck=0 OR unresponsive

## 9. Output Format

### 9.1 Raw Data (JSON)

```json
{
  "benchmark_id": "BENCH-001-20260106-001",
  "environment": {
    "hostname": "lambda-lab",
    "kernel": "6.8.0-90-generic",
    "cpu": "AMD Ryzen Threadripper 7960X",
    "ram_gb": 125,
    "gpu": "NVIDIA RTX 4090"
  },
  "results": [
    {
      "technology": "trueno-zram",
      "workload": "W2-TEXT",
      "pattern": "P1-SEQ-R",
      "run": 1,
      "throughput_gbps": 16.5,
      "iops": null,
      "latency_p50_us": 45,
      "latency_p99_us": 120,
      "cpu_percent": 35,
      "compression_ratio": 3.87
    }
  ]
}
```

### 9.2 Summary Report (Markdown)

```markdown
## Swap Technology Comparison - [Date]

| Metric | Regular Swap | Kernel ZRAM | trueno-zram | Winner |
|--------|--------------|-------------|-------------|--------|
| Seq Read (GB/s) | X.X | X.X | X.X | ... |
| Seq Write (GB/s) | X.X | X.X | X.X | ... |
| Rand IOPS (K) | X | X | X | ... |
| Compression | 1.0x | X.Xx | X.Xx | ... |
| CPU Usage (%) | X | X | X | ... |
```

## 10. CI Integration

### 10.1 GitHub Actions Workflow

```yaml
name: Swap Benchmark
on:
  workflow_dispatch:
  schedule:
    - cron: '0 2 * * 0'  # Weekly

jobs:
  benchmark:
    runs-on: self-hosted  # Requires bare metal
    steps:
      - uses: actions/checkout@v4
      - name: Run benchmark
        run: sudo ./scripts/scientific-swap-benchmark.sh
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: benchmark-results/
```

## 11. Known Limitations

1. **Hardware dependency**: Results vary significantly by CPU (AVX-512 vs AVX2)
2. **Kernel version**: ublk performance improved significantly in kernel 6.0+
3. **Memory pressure**: Real swap behavior differs from synthetic benchmarks
4. **GPU results**: PCIe bandwidth limits end-to-end GPU throughput

## 12. Peer-Reviewed References

### Benchmarking Methodology

1. **Yi, J.J., Lilja, D.J., & Hawkins, D.M. (2005).** "Improving Computer Architecture Simulation Methodology by Adding Statistical Rigor." *IEEE Transactions on Computers*, 54(11), 1360-1373. doi:10.1109/TC.2005.181

2. **John, L.K., & Eeckhout, L. (Eds.). (2018).** *Performance Evaluation and Benchmarking.* CRC Press. ISBN: 978-0367392161

3. **Mytkowicz, T., Diwan, A., Hauswirth, M., & Sweeney, P.F. (2009).** "Producing Wrong Data Without Doing Anything Obviously Wrong!" *ASPLOS '09*, pp. 265-276. doi:10.1145/1508244.1508275

### Memory and Swap Systems

4. **Huang, J., Qureshi, M.K., & Schwan, K. (2016).** "An Evolutionary Study of Linux Memory Management for Fun and Profit." *USENIX ATC '16*, pp. 465-478. [Link](https://www.usenix.org/system/files/conference/atc16/atc16_paper-huang.pdf)

5. **Cervera, R., Cortes, T., & Becerra, Y. (1999).** "Improving Application Performance through Swap Compression." *USENIX ATC '99*. [Link](https://www.usenix.org/conference/1999-usenix-annual-technical-conference/improving-application-performance-through-swap)

6. **Ruan, Z., et al. (2020).** "AIFM: High-Performance, Application-Integrated Far Memory." *OSDI '20*, pp. 315-332. [Link](https://www.usenix.org/system/files/osdi20-ruan.pdf)

### Performance Profiling

7. **Gregg, B. (2016).** "The Flame Graph." *Communications of the ACM*, 59(6), 48-57. doi:10.1145/2909476

8. **Gregg, B. (2013).** "Thinking Methodically About Performance." *Communications of the ACM*, 56(2), 45-51. doi:10.1145/2408776.2408791

9. **Bezemer, C.-P., Pouwelse, J., & Gregg, B. (2015).** "Understanding Software Performance Regressions Using Differential Flame Graphs." *IEEE SANER '15*, pp. 535-539.

### io_uring and Storage

10. **Axboe, J. (2019).** "Efficient IO with io_uring." *Linux Plumbers Conference*. [Link](https://kernel.dk/io_uring.pdf)

11. **Didona, D., et al. (2022).** "Understanding Modern Storage APIs: A Systematic Study of libaio, SPDK, and io_uring." *USENIX ATC '22*.

### Statistical Analysis

12. **Mann, H.B., & Whitney, D.R. (1947).** "On a Test of Whether One of Two Random Variables is Stochastically Larger than the Other." *Annals of Mathematical Statistics*, 18(1), 50-60.

13. **Wilcoxon, F. (1945).** "Individual Comparisons by Ranking Methods." *Biometrics Bulletin*, 1(6), 80-83.

14. **Cohen, J. (1988).** *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.

### Compression

15. **Collet, Y. (2011).** "LZ4 - Extremely Fast Compression Algorithm." [GitHub](https://github.com/lz4/lz4)

16. **Collet, Y., & Kucherawy, M. (2021).** "Zstandard Compression and the 'application/zstd' Media Type." *RFC 8878*, IETF.

## 13. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-01-06 | Claude Code | Initial specification |
| 2.0.0 | 2026-01-07 | Claude Code | Added RAM baseline, volume lifecycle, bashrs integration, renacer tracing, criterion benchmarks, flamegraph profiling, peer-reviewed citations |
| 2.1.0 | 2026-01-07 | Claude Code | Added SIMD verification (AVX-512), P6-BATCH pattern, efficiency metrics (IOPS/cycle), context switch monitoring per reviewer feedback |
