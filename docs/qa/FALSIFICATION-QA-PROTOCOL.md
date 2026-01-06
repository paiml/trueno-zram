# trueno-zram Falsification QA Protocol

**Document ID:** QA-FALSIFY-001
**Version:** 2.0
**Date:** 2026-01-06
**Status:** READY FOR EXTERNAL QA

## Executive Summary

This document provides a comprehensive falsification testing protocol for trueno-zram performance claims. All claims are designed to be **independently verifiable** and **falsifiable** - if a claim fails testing, it should be rejected.

**Critical Principle:** We want QA to find failures. A claim that cannot be falsified is not scientific.

---

## 1. Environment Requirements

### 1.1 Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| CPU | x86_64 AVX2 | x86_64 AVX-512 | ARM NEON also supported |
| RAM | 32 GB | 64+ GB | More RAM = larger test datasets |
| Storage | NVMe SSD | NVMe Gen4 | For swap file baseline |
| GPU | None | RTX 4090/A100 | For GPU decompression tests |

### 1.2 Software Requirements

```bash
# Required packages
sudo apt install -y fio jq bc stress-ng sysstat

# Verify versions
fio --version          # >= 3.35
jq --version           # >= 1.6
stress-ng --version    # >= 0.17

# Kernel requirements
uname -r               # >= 6.0 (ublk support)
modprobe -n ublk_drv   # Must succeed
modprobe -n zram       # Must succeed
```

### 1.3 Pre-Test Checklist

```bash
#!/bin/bash
# Run this before any testing

# 1. Record baseline system state
cat /proc/cpuinfo | grep -E "model name|flags" | head -5
free -h
cat /proc/swaps

# 2. Disable existing swap
sudo swapoff -a

# 3. Set CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 4. Disable transparent huge pages (reduces variance)
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled

# 5. Drop caches
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches

# 6. Build trueno-zram
cd /path/to/trueno-zram
cargo build --release -p trueno-ublk
cargo build --release -p trueno-zram-core --examples
```

---

## 2. Performance Claims to Falsify

### Claim Matrix

| ID | Claim | Threshold | Falsification Criteria |
|----|-------|-----------|------------------------|
| C1 | Sequential I/O 1.8x vs kernel ZRAM | >= 1.5x | FAIL if < 1.5x on mixed data |
| C2 | 228K random IOPS | >= 180K | FAIL if < 180K IOPS |
| C3 | 3.87x compression ratio | >= 2.5x | FAIL if < 2.5x on mixed workload |
| C4 | CPU compression 20-30 GB/s | >= 15 GB/s | FAIL if < 15 GB/s parallel |
| C5 | CPU decompression 50 GB/s | >= 30 GB/s | FAIL if < 30 GB/s |
| C6 | Same-fill 2048:1 ratio | == 2048:1 | FAIL if != 2048:1 for zero pages |
| C7 | P99 latency < 100us | < 150us | FAIL if >= 150us |
| C8 | DT-007 no swap deadlock | mlock active | FAIL if VmLck < 100MB |

---

## 3. Test Procedures

### 3.1 Test T1: Sequential I/O Comparison

**Claim:** trueno-zram is 1.8x faster than kernel ZRAM for sequential I/O

**Setup:**
```bash
# Terminal 1: Setup kernel ZRAM
sudo modprobe zram num_devices=1
echo lz4 | sudo tee /sys/block/zram0/comp_algorithm
echo 8G | sudo tee /sys/block/zram0/disksize
sudo mkswap /dev/zram0
sudo swapon /dev/zram0 -p 150

# Run fio with MIXED data (not zeros!)
sudo fio --name=seq_read \
    --filename=/dev/zram0 \
    --rw=read \
    --bs=1M \
    --direct=1 \
    --numjobs=8 \
    --runtime=30 \
    --time_based \
    --group_reporting \
    --output-format=json \
    --output=kernel_zram_seq.json

# Record bandwidth
KERNEL_BW=$(jq -r '.jobs[0].read.bw_bytes' kernel_zram_seq.json)
echo "Kernel ZRAM: $((KERNEL_BW / 1000000000)) GB/s"

# Cleanup
sudo swapoff /dev/zram0
echo 1 | sudo tee /sys/block/zram0/reset
sudo rmmod zram
```

```bash
# Terminal 2: Setup trueno-zram
sudo ./target/release/trueno-ublk create --size 8G --dev-id 5 --foreground &
sleep 3
sudo mkswap /dev/ublkb5
sudo swapon /dev/ublkb5 -p 200

# Run same fio test
sudo fio --name=seq_read \
    --filename=/dev/ublkb5 \
    --rw=read \
    --bs=1M \
    --direct=1 \
    --numjobs=8 \
    --runtime=30 \
    --time_based \
    --group_reporting \
    --output-format=json \
    --output=trueno_zram_seq.json

TRUENO_BW=$(jq -r '.jobs[0].read.bw_bytes' trueno_zram_seq.json)
echo "trueno-zram: $((TRUENO_BW / 1000000000)) GB/s"

# Calculate speedup
SPEEDUP=$(echo "scale=2; $TRUENO_BW / $KERNEL_BW" | bc)
echo "Speedup: ${SPEEDUP}x"

# Cleanup
sudo swapoff /dev/ublkb5
sudo pkill trueno-ublk
```

**Pass Criteria:** SPEEDUP >= 1.5
**Fail Criteria:** SPEEDUP < 1.5

**IMPORTANT:** If using zero-filled data, kernel ZRAM will show 170+ GB/s (misleading). Use `--buffer_pattern=0xdeadbeef` or pre-fill with random data for fair comparison.

---

### 3.2 Test T2: Random IOPS

**Claim:** trueno-zram achieves 228K random read IOPS

```bash
# Setup trueno-zram (same as above)
sudo ./target/release/trueno-ublk create --size 8G --dev-id 5 --foreground &
sleep 3
sudo mkswap /dev/ublkb5
sudo swapon /dev/ublkb5 -p 200

# Run random read test
sudo fio --name=rand_read \
    --filename=/dev/ublkb5 \
    --rw=randread \
    --bs=4k \
    --direct=1 \
    --iodepth=32 \
    --numjobs=4 \
    --runtime=30 \
    --time_based \
    --group_reporting \
    --output-format=json \
    --output=trueno_rand.json

IOPS=$(jq -r '.jobs[0].read.iops' trueno_rand.json)
echo "IOPS: $IOPS"

# Cleanup
sudo swapoff /dev/ublkb5
sudo pkill trueno-ublk
```

**Pass Criteria:** IOPS >= 180000
**Fail Criteria:** IOPS < 180000

---

### 3.3 Test T3: Compression Ratio

**Claim:** 3.87x compression ratio on mixed workloads

```bash
# Run compression benchmark with text data
cargo run --release -p trueno-zram-core --example compress_benchmark 2>&1 | tee compress_results.txt

# Extract ratio for text workload
grep -E "Text.*Lz4" compress_results.txt

# Expected output format:
#    1000        Lz4       4.44 GB/s       5.37 GB/s      3.21x Avx512
```

**Pass Criteria:** Ratio >= 2.5x for text/mixed data
**Fail Criteria:** Ratio < 2.5x

**Note:** Random data will show ~1.0x ratio (incompressible). This is expected and correct behavior.

---

### 3.4 Test T4: CPU Compression Throughput

**Claim:** 20-30 GB/s parallel compression

```bash
# Run parallel compression benchmark
cargo run --release -p trueno-zram-core --example compress_benchmark 2>&1 | tee compress_perf.txt

# Look for parallel throughput numbers
grep -E "GB/s" compress_perf.txt
```

**Pass Criteria:** Compression >= 15 GB/s (with AVX-512)
**Fail Criteria:** Compression < 15 GB/s

**Note:** Results vary by CPU. AVX-512 > AVX2 > Scalar. Record CPU model.

---

### 3.5 Test T5: Same-Fill Detection

**Claim:** 2048:1 compression for zero/repeated pages

```bash
# Create 4KB zero page
dd if=/dev/zero bs=4096 count=1 of=/tmp/zero_page

# Test same-fill detection
cargo run --release -p trueno-zram-core --example compress_benchmark 2>&1 | grep -i zero

# Expected: "Zeros" pattern should show 2048.00x ratio
```

**Pass Criteria:** Zero pages compress to exactly 2048:1
**Fail Criteria:** Ratio != 2048:1 for zero pages

---

### 3.6 Test T6: Latency P99

**Claim:** P99 latency < 100us

```bash
# Run with latency measurement
sudo fio --name=latency_test \
    --filename=/dev/ublkb5 \
    --rw=randread \
    --bs=4k \
    --direct=1 \
    --iodepth=1 \
    --numjobs=1 \
    --runtime=30 \
    --time_based \
    --output-format=json \
    --output=latency.json

P99=$(jq -r '.jobs[0].read.clat_ns.percentile["99.000000"]' latency.json)
P99_US=$((P99 / 1000))
echo "P99 Latency: ${P99_US}us"
```

**Pass Criteria:** P99 < 150us
**Fail Criteria:** P99 >= 150us

---

### 3.7 Test T7: DT-007 Swap Deadlock Prevention

**Claim:** Daemon memory is mlocked to prevent swap deadlock

```bash
# Start daemon
sudo ./target/release/trueno-ublk create --size 8G --dev-id 5 --foreground &
DAEMON_PID=$!
sleep 2

# Check VmLck
VMLCK=$(grep VmLck /proc/$DAEMON_PID/status | awk '{print $2}')
echo "VmLck: ${VMLCK} kB"

# Check process state under memory pressure
sudo stress-ng --vm 4 --vm-bytes 80% --timeout 30s &
sleep 10

STATE=$(cat /proc/$DAEMON_PID/status | grep State | awk '{print $2}')
echo "Daemon state under pressure: $STATE"

# Cleanup
sudo pkill stress-ng
sudo pkill trueno-ublk
```

**Pass Criteria:**
- VmLck >= 100000 kB (100MB)
- State == 'S' (sleeping) or 'R' (running), NOT 'D' (uninterruptible sleep)

**Fail Criteria:**
- VmLck < 100000 kB
- State == 'D' (indicates deadlock)

---

### 3.8 Test T8: GPU Decompression (Optional)

**Claim:** 137 GB/s GPU decompression (kernel-only, not end-to-end)

```bash
# Requires NVIDIA GPU with CUDA
cargo run --release -p trueno-zram-core --example compress_benchmark --features cuda 2>&1 | tee gpu_results.txt

# Look for GPU backend
grep -i "cuda\|gpu" gpu_results.txt
```

**Note:** The 137 GB/s is kernel execution time only. End-to-end throughput is limited by PCIe bandwidth (~6-10 GB/s). This is documented and expected.

**Pass Criteria:** GPU kernel shows >= 50 GB/s
**Fail Criteria:** GPU not detected or < 50 GB/s kernel throughput

---

## 4. Statistical Requirements

### 4.1 Run Requirements

- **Minimum runs:** 3 per test
- **Recommended runs:** 5 per test
- **Outlier removal:** IQR method (remove > 1.5*IQR from Q1/Q3)
- **Report:** Median (not mean) for robustness

### 4.2 Variance Thresholds

| Metric | Acceptable Variance |
|--------|---------------------|
| Throughput | < 10% |
| IOPS | < 15% |
| Latency | < 20% |
| Compression ratio | < 5% |

### 4.3 Statistical Significance

For comparing trueno-zram vs kernel ZRAM:
- Use Mann-Whitney U test (non-parametric)
- p < 0.05 required to claim significant difference
- Effect size (Cohen's d) >= 0.8 for "large" improvement

---

## 5. Reporting Template

```markdown
# Falsification Test Report

**Tester:** [Name]
**Date:** [Date]
**Hardware:** [CPU Model, RAM, GPU if applicable]
**Kernel:** [uname -r output]
**trueno-zram version:** [git rev-parse HEAD]

## Test Results

| Test | Claim | Result | Pass/Fail |
|------|-------|--------|-----------|
| T1 | 1.8x seq I/O | [X.Xx] | [PASS/FAIL] |
| T2 | 228K IOPS | [X K] | [PASS/FAIL] |
| T3 | 3.87x ratio | [X.Xx] | [PASS/FAIL] |
| T4 | 20-30 GB/s compress | [X GB/s] | [PASS/FAIL] |
| T5 | 2048:1 same-fill | [X:1] | [PASS/FAIL] |
| T6 | P99 < 100us | [X us] | [PASS/FAIL] |
| T7 | mlock active | [X kB] | [PASS/FAIL] |
| T8 | GPU 137 GB/s | [X GB/s] | [PASS/FAIL/SKIP] |

## Raw Data

[Attach JSON output files]

## Observations

[Any anomalies, notes, or concerns]

## Conclusion

[Overall assessment: VALIDATED / PARTIALLY VALIDATED / FALSIFIED]
```

---

## 6. Known Limitations

1. **Zero-data bias:** Kernel ZRAM shows 170+ GB/s with zeros (misleading). Always use mixed/random data for fair comparison.

2. **CPU variance:** Results depend heavily on CPU features (AVX-512 vs AVX2). Always record CPU model.

3. **GPU PCIe limit:** GPU kernel throughput (137 GB/s) != end-to-end throughput (~6-10 GB/s due to PCIe).

4. **Orphaned devices:** Previous crashed runs may leave `/dev/ublkc*` devices. Use device IDs 2-9 to avoid.

5. **Memory pressure:** Tests under memory pressure may vary. Ensure consistent baseline.

---

## 7. Automated Test Script

```bash
#!/bin/bash
# Run full falsification suite
cd /path/to/trueno-zram
sudo ./scripts/scientific-swap-benchmark.sh --full 2>&1 | tee qa_results.txt
```

This runs all tests with proper statistical rigor (5 runs each, 60s runtime).

---

## 8. Contact

For questions or clarification:
- Repository: https://github.com/paiml/trueno-zram
- Issues: https://github.com/paiml/trueno-zram/issues

**Remember:** The goal is to find failures. A robust claim survives falsification attempts.
