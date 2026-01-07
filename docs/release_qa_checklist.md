# trueno-ublk v3.17.0 Release Verification Matrix
## The "Falsify First" Protocol

**Version:** 1.0.0
**Target:** trueno-ublk v3.17.0
**Philosophy:** "Do not prove it works. Try to prove it is broken. If you cannot prove it is broken, it is ready."

---

### 🟢 Section 1: Performance Falsification (The Speed Barriers)
*Reject the release if ANY metric falls below the Falsification Threshold.*

| ID | Test Case | Command / Method | Hypothesis (Success) | **Falsification Threshold (FAIL)** | Verified? |
|----|-----------|------------------|----------------------|------------------------------------|-----------|
| **P1** | **The Zero-Page Wall** | `fio --name=zero --rw=read --bs=1M --ioengine=io_uring --iodepth=64 --filename=/dev/ublkb0` | Same-fill detection bypasses compression, hitting > 7.5 GB/s. | **< 7.2 GB/s** (Implies same-fill logic broken) | [ ] |
| **P2** | **The IOPS Floor** | `fio --name=iops --rw=randread --bs=4k --ioengine=io_uring --iodepth=64 --numjobs=4` | Multi-queue scaling delivers > 600K IOPS. | **< 550K IOPS** (Implies lock contention/single-queue regression) | [ ] |
| **P3** | **ZSTD Velocity** | `cargo bench --bench zstd_simd` (AVX-512 capable host) | SIMD ZSTD-1 compresses at > 15 GiB/s. | **< 12 GiB/s** (Implies AVX-512 fallback failure) | [ ] |
| **P4** | **Latency Spike** | `fio` (same as P2) check `lat_ns` p99 | p99 Latency remains < 200µs under load. | **> 300µs** (Implies garbage collection/lock stalling) | [ ] |

---

### 🟡 Section 2: Architectural Falsification (The Logic Gates)
*Verify the "Kernel-Cooperative" routing logic is actually working and not just passing data through.*

| ID | Test Case | Method | Hypothesis (Success) | **Falsification Criteria (FAIL)** | Verified? |
|----|-----------|--------|----------------------|-----------------------------------|-----------|
| **A1** | **Entropy Discrimination** | Write 1GB text (H~4.5) -> Check ZRAM stats | Low entropy routes to Kernel ZRAM (`/dev/zram0`). | **I/O appears in NVMe tier** OR **trueno RSS balloons** (Tier 1 leak) | [ ] |
| **A2** | **The Random Reject** | Write 1GB `/dev/urandom` (H~7.9) -> Check NVMe stats | High entropy routes to NVMe/Cold Tier directly. | **CPU usage spikes** (Compression was attempted on noise) | [ ] |
| **A3** | **The ZRAM Barrier** | Run `dd` on Kernel ZRAM tier | Throughput matches kernel native (~1.3 GB/s bulk). | **< 1.0 GB/s** (Userspace overhead > 30% is unacceptable) | [ ] |

---

### 🔴 Section 3: Integrity Falsification (The "Bit Rot" Check)
*Critical Path. One failure here halts the release immediately.*

| ID | Test Case | Command / Method | Hypothesis (Success) | **Falsification Criteria (FAIL)** | Verified? |
|----|-----------|------------------|----------------------|-----------------------------------|-----------|
| **I1** | **The Round Trip** | `fio --verify=crc32c ...` (Mixed patterns) | Data read back is bit-perfect. | **ANY checksum mismatch** | [ ] |
| **I2** | **Tear-Down Safety** | `kill -9` daemon during heavy write -> Restart -> Read | Journal/WAL recovers state (or clean crash). | **Panic on restart** or **Data corruption** | [ ] |
| **I3** | **Same-Fill Edge** | Write 4KB zeros -> Write 4KB random -> Read both | Correct switching between `SameFill` and `Compressed`. | **Stale zero page** returned for random data | [ ] |

---

### 🔵 Section 4: Observability Truth (The "Gaslight" Check)
*Verify the telemetry is not lying.*

| ID | Test Case | Method | Hypothesis (Success) | **Falsification Criteria (FAIL)** | Verified? |
|----|-----------|--------|----------------------|-----------------------------------|-----------|
| **O1** | **Renacer Truth** | Run `trueno-ublk --visualize` vs `iostat -x 1` | TUI throughput matches system stats. | **Delta > 5%** (Telemetry is hallucinating) | [ ] |
| **O2** | **Tier Visibility** | Fill Tier 0 (ZRAM), watch spill to Tier 1 | Renacer heatmap shows Tier 0 -> Tier 1 transition. | **Heatmap static** while swap fills up | [ ] |

---

### 📋 Release Sign-Off

**Tester:** ____________________  **Date:** _______________

- [ ] All **Critical (Red)** tests passed.
- [ ] All **Performance (Green)** tests met thresholds.
- [ ] **Falsification Report** generated and attached.

**Status:**
[ ] GO for v3.17.0
[ ] NO-GO (Stop the Line)
