# Scientific Swap Technology Benchmark Specification

**Document ID:** BENCH-001
**Status:** DRAFT
**Created:** 2026-01-06
**Priority:** P0 - BLOCKING RELEASE

## 1. Purpose

Provide reproducible, falsifiable performance comparison between:
1. **Regular Swap** - Disk-backed swap partition/file
2. **Kernel ZRAM** - Linux kernel compressed RAM swap
3. **trueno-zram** - SIMD/GPU accelerated userspace compression

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

### 4.3 Full Test Matrix

Each swap technology tested against each workload × pattern combination:
- 3 technologies × 5 workloads × 5 patterns = 75 test cases
- Each test case run 3 times for statistical validity
- Total: 225 test runs

## 5. Metrics Collected

### 5.1 Primary Metrics

| Metric | Unit | Collection Method |
|--------|------|-------------------|
| Throughput | GB/s | fio bandwidth |
| IOPS | ops/s | fio iops |
| Latency P50 | μs | fio clat percentile |
| Latency P99 | μs | fio clat percentile |
| CPU Usage | % | mpstat during test |

### 5.2 Secondary Metrics

| Metric | Unit | Collection Method |
|--------|------|-------------------|
| Compression Ratio | x | (orig_size / comp_size) |
| Memory Usage | MB | /proc/meminfo |
| Page Faults | count | /proc/vmstat |
| Swap I/O | MB/s | iostat |

## 6. Benchmark Implementation

### 6.1 Setup Scripts

```bash
#!/bin/bash
# bench-setup.sh - Configure swap technologies

setup_regular_swap() {
    local size_gb=$1
    sudo dd if=/dev/zero of=/swapfile bs=1G count=$size_gb
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile -p 100
}

setup_kernel_zram() {
    local size_gb=$1
    sudo modprobe zram
    echo lz4 | sudo tee /sys/block/zram0/comp_algorithm
    echo ${size_gb}G | sudo tee /sys/block/zram0/disksize
    sudo mkswap /dev/zram0
    sudo swapon /dev/zram0 -p 150
}

setup_trueno_zram() {
    local size_gb=$1
    sudo ./target/release/trueno-ublk --size ${size_gb}G --id 0
    sleep 2
    sudo mkswap /dev/ublkb0
    sudo swapon /dev/ublkb0 -p 200
}
```

### 6.2 Benchmark Runner

See: `scripts/scientific-swap-benchmark.sh`

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

## 12. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-01-06 | Claude Code | Initial specification |
