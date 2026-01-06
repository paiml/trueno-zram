# trueno-ublk Testing, Debugging, and Troubleshooting Specification

**Version:** 1.1.0
**Date:** 2026-01-06
**Status:** Active Development
**Classification:** Engineering Specification

## Abstract

This specification defines the testing, debugging, and troubleshooting methodology for trueno-ublk, a userspace block device implementation using Linux ublk and io_uring. The document incorporates Toyota Production System (TPS) principles, Five-Whys root cause analysis, and a comprehensive 100-point falsification matrix. All destructive testing MUST be conducted in disposable Docker containers to prevent system destabilization.

---

## Table of Contents

1. [Known Bugs and Issues](#1-known-bugs-and-issues)
2. [Five-Whys Root Cause Analysis](#2-five-whys-root-cause-analysis)
3. [Toyota Way Testing Strategy](#3-toyota-way-testing-strategy)
4. [Docker-Based Testing Infrastructure](#4-docker-based-testing-infrastructure)
5. [100-Point Falsification Matrix](#5-100-point-falsification-matrix)
6. [Debugging Procedures](#6-debugging-procedures)
7. [Troubleshooting Guide](#7-troubleshooting-guide)
8. [Peer-Reviewed References](#8-peer-reviewed-references)

---

## 1. Known Bugs and Issues

### 1.1 Critical: I/O Data Loss Bug (F083)

**Severity:** Critical (System Destabilization)
**Status:** RESOLVED (2026-01-06)
**Discovered:** 2025-01-06
**Environment:** Linux 6.8.0-90-generic, Ubuntu 24.04

#### Root Cause Analysis (Five-Whys)

1. **Why did I/O fail?** - Kernel reported I/O errors on multi-segment reads
2. **Why multi-segment errors?** - Stale ublk device state from crashed daemon
3. **Why stale state?** - Module kept reference count after daemon crash
4. **Why reference count stuck?** - Device cleanup not called on abnormal exit
5. **Why no cleanup?** - Daemon killed without graceful shutdown

#### Resolution

The core USER_COPY I/O implementation is **correct**. Fresh devices work perfectly:

| Test | Status |
|------|--------|
| F016 (read-after-write) | ✓ PASSED |
| F086 (mkfs.ext4 + mount) | ✓ PASSED |
| F092 (swap enable/disable) | ✓ PASSED |

**Follow-up Task:** DT-003 will add signal handlers for graceful shutdown and orphan cleanup.

#### Historical Symptom Description

The trueno-ublk daemon creates block devices that appear functional but exhibit silent data corruption when stale kernel state exists:

1. Device creation succeeds (`/dev/ublkbN` appears)
2. Block device node is correctly typed (block special)
3. Write operations return success
4. Read operations return zero/empty data instead of written data
5. Filesystem creation (mkfs.btrfs, mkfs.ext4) fails with checksum errors
6. System swap on ublk device leads to orphaned swap pages after daemon crash

#### Observed Errors

```
# BTRFS checksum failure
BTRFS error (device /dev/ublkb5): checksum verify failed on logical 30441472
mirror 1 wanted 0xe11496c0 found 0x158cfe07

# ext4 mkfs failure
mkfs.ext4: Input/output error while writing superblock

# I/O verification failure
Write MD5: 9a5939e795c9c840ccb805032200f29e
Read MD5:  d41d8cd98f00b204e9800998ecf8427e  (empty file hash)

# dd error
dd: error writing '/dev/ublkb5': Input/output error
```

#### System Impact

- Orphaned swap pages that persist until reboot
- Module `ublk_drv` cannot be unloaded while swap is orphaned
- Potential data loss if used for persistent storage
- System instability requiring reboot for full recovery

---

## 2. Five-Whys Root Cause Analysis

### 2.1 Analysis of F083 (I/O Data Loss)

The Five-Whys technique, developed by Sakichi Toyoda and integral to the Toyota Production System, systematically identifies root causes by iteratively asking "why" until the fundamental cause is revealed [1].

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FIVE-WHYS ANALYSIS: F083                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PROBLEM: Filesystem creation fails with checksum errors                 │
│                                                                          │
│  WHY #1: Why did filesystem creation fail?                               │
│  ├─► Checksum verification failed during mkfs write verification        │
│  │                                                                       │
│  WHY #2: Why did checksum verification fail?                             │
│  ├─► Data read back from device differs from data written               │
│  │                                                                       │
│  WHY #3: Why does read data differ from written data?                    │
│  ├─► Read operations return zero/empty buffers instead of stored data   │
│  │                                                                       │
│  WHY #4: Why do reads return empty data?                                 │
│  ├─► HYPOTHESIS A: pread() from char device not receiving data          │
│  ├─► HYPOTHESIS B: USER_COPY offset calculation incorrect               │
│  ├─► HYPOTHESIS C: io_uring completion not signaling data ready         │
│  ├─► HYPOTHESIS D: PageStore read returning wrong sector data           │
│  │                                                                       │
│  WHY #5: Why is pread/offset/completion/store failing?                   │
│  ├─► ROOT CAUSE CANDIDATES:                                              │
│  │   RC-A: ublk_user_copy_offset() returns wrong offset for tag/queue   │
│  │   RC-B: UBLK_F_USER_COPY flag not properly negotiated with kernel    │
│  │   RC-C: io_uring URING_CMD completion races with pread/pwrite        │
│  │   RC-D: Sector-to-page mapping arithmetic error in PageStore         │
│  │   RC-E: Compression/decompression buffer size mismatch               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Verification Matrix for Root Causes

| ID   | Root Cause Hypothesis | Verification Method | Docker Test |
|------|----------------------|---------------------|-------------|
| RC-A | Offset calculation | Compare with libublk reference | T-001 |
| RC-B | USER_COPY negotiation | Trace UBLK_F_* flags at setup | T-002 |
| RC-C | io_uring race condition | Add memory barriers, trace timing | T-003 |
| RC-D | Sector mapping error | Unit test with known sector values | T-004 |
| RC-E | Buffer size mismatch | Assert buffer sizes at boundaries | T-005 |

### 2.3 Secondary Five-Whys: Orphaned Swap

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FIVE-WHYS ANALYSIS: Orphaned Swap                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PROBLEM: Swap pages remain after ublk daemon crash                      │
│                                                                          │
│  WHY #1: Why do swap pages remain?                                       │
│  ├─► Kernel swap subsystem still references /dev/ublkbN                 │
│  │                                                                       │
│  WHY #2: Why does kernel still reference the device?                     │
│  ├─► swapoff requires reading pages back to migrate to RAM              │
│  │                                                                       │
│  WHY #3: Why can't pages be read back?                                   │
│  ├─► Device node marked as "(deleted)" - daemon is gone                 │
│  │                                                                       │
│  WHY #4: Why did daemon exit without proper cleanup?                     │
│  ├─► Daemon crashed due to I/O error loop or was killed                 │
│  │                                                                       │
│  WHY #5: Why wasn't swap properly disabled before daemon exit?           │
│  ├─► ROOT CAUSE: No graceful shutdown protocol for swap devices         │
│  ├─► MITIGATION: Implement SIGTERM handler that calls swapoff first     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Toyota Way Testing Strategy

### 3.1 Foundational Principles

The Toyota Way, documented by Liker [2], provides 14 management principles that form the basis of Toyota's quality culture. We adapt the following principles for software testing:

| TPS Principle | Testing Application |
|---------------|---------------------|
| **Genchi Genbutsu** (Go and See) | Test on real hardware, observe actual behavior |
| **Jidoka** (Automation with Human Touch) | Automated tests with manual verification gates |
| **Kaizen** (Continuous Improvement) | Iterative test refinement based on failures |
| **Hansei** (Reflection) | Post-mortem analysis of all test failures |
| **Heijunka** (Leveling) | Distribute test load across CI/CD pipeline |
| **Andon** (Stop and Fix) | Halt deployment on any test failure |

### 3.2 Test Pyramid Structure

Following the test pyramid model [3], tests are organized by execution cost and scope:

```
                          ┌─────────────┐
                          │  Production │  ← Canary (1 per release)
                          │   Smoke     │
                         ─┴─────────────┴─
                        ┌─────────────────┐
                        │   Integration   │  ← Docker (10-20 tests)
                        │   (Full Stack)  │
                       ─┴─────────────────┴─
                      ┌─────────────────────┐
                      │     Component       │  ← Docker (50-100 tests)
                      │   (Subsystem)       │
                     ─┴─────────────────────┴─
                    ┌─────────────────────────┐
                    │         Unit            │  ← Native (500+ tests)
                    │    (Function Level)     │
                   ─┴─────────────────────────┴─
```

### 3.3 Poka-Yoke (Mistake-Proofing) Guards

Implement compile-time and runtime guards to prevent defects:

```rust
// Compile-time: Ensure buffer alignment
#[repr(align(4096))]
struct PageBuffer([u8; PAGE_SIZE]);

// Runtime: Validate sector bounds before I/O
fn validate_io_request(req: &IoRequest, dev_sectors: u64) -> Result<(), IoError> {
    // Poka-yoke: Cannot proceed with invalid request
    if req.start_sector + req.nr_sectors as u64 > dev_sectors {
        return Err(IoError::OutOfBounds);
    }
    Ok(())
}
```

### 3.4 Andon Cord: Stop-the-Line Criteria

Tests MUST halt the CI/CD pipeline if any of the following occur:

1. **Data Integrity Failure**: Any read ≠ write verification
2. **Memory Corruption**: AddressSanitizer or MIRI detection
3. **Kernel Oops**: dmesg shows ublk-related kernel errors
4. **Resource Leak**: File descriptors, memory, or device nodes not cleaned up
5. **Timeout**: Any test exceeds 5x expected duration

### 3.5 Mandatory Quality Tooling

To ensure consistency with the "Batuta Stack" standards, the following tooling MUST be employed throughout the development lifecycle:

| Tool | Scope | Requirement |
|------|-------|-------------|
| **bashrs** | Shell, Make, Docker | All `*.sh` scripts, `Makefile`s, and `Dockerfile`s MUST be validated and executed via `bashrs` to ensure safety and portability. |
| **probador** | Testing | All test suites (unit, integration, fuzzing) MUST be orchestrated by `probador`. Standard `cargo test` should be wrapped or invoked by `probador` for reporting. |
| **renacer** | Tracing | All subsystem observability MUST utilize `renacer` structured tracing. Raw `println!` debugging is PROHIBITED in production paths. |
| **pmat** | QA & Tracking | All deliverables MUST pass the `pmat` automated QA gate. Project status and metrics MUST be tracked via `pmat`. |

---

## 4. Docker-Based Testing Infrastructure

### 4.1 Rationale

Testing ublk devices on production systems risks:
- Kernel module state corruption
- Orphaned block devices
- Swap subsystem instability
- Data loss on mounted filesystems

Docker containers with `--privileged` mode provide isolated environments where kernel modules can be loaded/unloaded without affecting the host [4].

### 4.2 Base Docker Image

```dockerfile
# Dockerfile.ublk-test
FROM ubuntu:24.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    linux-headers-generic \
    kmod \
    btrfs-progs \
    e2fsprogs \
    xfsprogs \
    fio \
    stress-ng \
    strace \
    perf-tools-unstable \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Add test scripts
COPY scripts/docker-test-harness.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-test-harness.sh

# Working directory
WORKDIR /workspace

# Default command runs test harness
CMD ["/usr/local/bin/docker-test-harness.sh"]
```

### 4.3 Docker Compose for Test Matrix

```yaml
# docker-compose.test.yml
version: '3.8'

services:
  # Unit tests (no privileges needed)
  unit-tests:
    build:
      context: .
      dockerfile: Dockerfile.ublk-test
    command: cargo test --lib
    volumes:
      - .:/workspace:ro
      - cargo-cache:/root/.cargo/registry

  # Component tests (privileged for ublk)
  component-tests:
    build:
      context: .
      dockerfile: Dockerfile.ublk-test
    privileged: true
    command: /usr/local/bin/docker-test-harness.sh component
    volumes:
      - .:/workspace:ro
      - /lib/modules:/lib/modules:ro
    cap_add:
      - SYS_ADMIN
      - MKNOD
    devices:
      - /dev/ublk-control:/dev/ublk-control

  # Integration tests (privileged, full stack)
  integration-tests:
    build:
      context: .
      dockerfile: Dockerfile.ublk-test
    privileged: true
    command: /usr/local/bin/docker-test-harness.sh integration
    volumes:
      - .:/workspace:ro
      - /lib/modules:/lib/modules:ro
    cap_add:
      - SYS_ADMIN
      - MKNOD
    tmpfs:
      - /mnt/test:size=4G

  # Stress tests (privileged, memory pressure)
  stress-tests:
    build:
      context: .
      dockerfile: Dockerfile.ublk-test
    privileged: true
    command: /usr/local/bin/docker-test-harness.sh stress
    volumes:
      - .:/workspace:ro
      - /lib/modules:/lib/modules:ro
    mem_limit: 8g
    memswap_limit: 16g

volumes:
  cargo-cache:
```

### 4.4 Test Harness Script

```bash
#!/bin/bash
# docker-test-harness.sh
set -euo pipefail

TEST_LEVEL="${1:-unit}"
RESULTS_DIR="/workspace/test-results"
mkdir -p "$RESULTS_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

cleanup() {
    log "Cleaning up..."
    # Stop any running trueno-ublk daemons
    pkill trueno-ublk 2>/dev/null || true
    # Reset any ublk devices
    for id in $(seq 0 9); do
        /workspace/target/release/trueno-ublk reset $id 2>/dev/null || true
    done
    # Unload module if loaded
    rmmod ublk_drv 2>/dev/null || true
    log "Cleanup complete"
}

trap cleanup EXIT

case "$TEST_LEVEL" in
    unit)
        log "Running unit tests..."
        cargo test --lib -- --test-threads=4 2>&1 | tee "$RESULTS_DIR/unit.log"
        ;;

    component)
        log "Loading ublk_drv module..."
        modprobe ublk_drv || { log "ERROR: Cannot load ublk_drv"; exit 1; }

        log "Building release binary..."
        cargo build --release -p trueno-ublk

        log "Running component tests..."
        # Test 1: Device creation
        log "T-001: Device creation..."
        timeout 30 /workspace/target/release/trueno-ublk create \
            --size 1G --dev-id 0 --foreground &
        DAEMON_PID=$!
        sleep 2

        if [ -b /dev/ublkb0 ]; then
            log "T-001: PASS - Device created"
        else
            log "T-001: FAIL - Device not created"
            exit 1
        fi

        # Test 2: Basic I/O
        log "T-002: Basic I/O verification..."
        TEST_DATA=$(dd if=/dev/urandom bs=4096 count=1 2>/dev/null | base64)
        echo "$TEST_DATA" | base64 -d | dd of=/dev/ublkb0 bs=4096 count=1 2>/dev/null
        READ_DATA=$(dd if=/dev/ublkb0 bs=4096 count=1 2>/dev/null | base64)

        if [ "$TEST_DATA" = "$READ_DATA" ]; then
            log "T-002: PASS - I/O verification successful"
        else
            log "T-002: FAIL - I/O verification failed"
            log "  Written: ${TEST_DATA:0:64}..."
            log "  Read:    ${READ_DATA:0:64}..."
            kill $DAEMON_PID 2>/dev/null
            exit 1
        fi

        kill $DAEMON_PID 2>/dev/null
        wait $DAEMON_PID 2>/dev/null || true
        ;;

    integration)
        log "Running integration tests..."
        modprobe ublk_drv
        cargo build --release -p trueno-ublk

        # Test filesystem creation and mount
        log "T-010: Filesystem integration..."
        /workspace/target/release/trueno-ublk create \
            --size 2G --dev-id 0 --foreground &
        DAEMON_PID=$!
        sleep 2

        # Try ext4
        log "T-010a: ext4 filesystem..."
        if mkfs.ext4 -F /dev/ublkb0 2>&1; then
            mkdir -p /mnt/test
            if mount /dev/ublkb0 /mnt/test 2>&1; then
                echo "test data" > /mnt/test/testfile
                sync
                if grep -q "test data" /mnt/test/testfile; then
                    log "T-010a: PASS - ext4 read/write successful"
                else
                    log "T-010a: FAIL - ext4 read verification failed"
                fi
                umount /mnt/test
            else
                log "T-010a: FAIL - ext4 mount failed"
            fi
        else
            log "T-010a: FAIL - mkfs.ext4 failed"
        fi

        kill $DAEMON_PID 2>/dev/null
        wait $DAEMON_PID 2>/dev/null || true
        ;;

    stress)
        log "Running stress tests..."
        modprobe ublk_drv
        cargo build --release -p trueno-ublk

        # Create device
        /workspace/target/release/trueno-ublk create \
            --size 4G --dev-id 0 --foreground &
        DAEMON_PID=$!
        sleep 2

        # Run fio workload
        log "T-020: fio stress test..."
        fio --name=randwrite --ioengine=libaio --rw=randwrite \
            --bs=4k --direct=1 --size=1G --numjobs=4 \
            --filename=/dev/ublkb0 --time_based --runtime=60 \
            --group_reporting 2>&1 | tee "$RESULTS_DIR/fio.log"

        kill $DAEMON_PID 2>/dev/null
        wait $DAEMON_PID 2>/dev/null || true
        ;;

    *)
        log "Unknown test level: $TEST_LEVEL"
        exit 1
        ;;
esac

log "Tests completed successfully"
```

### 4.5 CI/CD Integration

```yaml
# .github/workflows/ublk-tests.yml
name: ublk Integration Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - name: Run unit tests
        run: cargo test --lib

  docker-component-tests:
    runs-on: ubuntu-24.04
    needs: unit-tests
    steps:
      - uses: actions/checkout@v4
      - name: Build test image
        run: docker build -t ublk-test -f Dockerfile.ublk-test .
      - name: Run component tests
        run: |
          docker run --privileged \
            -v /lib/modules:/lib/modules:ro \
            -v $(pwd):/workspace:ro \
            ublk-test component

  docker-integration-tests:
    runs-on: ubuntu-24.04
    needs: docker-component-tests
    steps:
      - uses: actions/checkout@v4
      - name: Run integration tests
        run: |
          docker run --privileged \
            -v /lib/modules:/lib/modules:ro \
            -v $(pwd):/workspace:ro \
            --tmpfs /mnt/test:size=4G \
            ublk-test integration
```

---

## 5. 100-Point Falsification Matrix

The falsification matrix follows Popper's principle of falsifiability [5]: each test attempts to disprove a hypothesis about system correctness. Tests are organized by subsystem and numbered F001-F100.

### 5.1 Device Lifecycle (F001-F015)

| ID | Hypothesis | Falsification Test | Expected | Docker |
|----|------------|-------------------|----------|--------|
| F001 | Device creation succeeds with valid config | Create device with size=1G, algorithm=lz4 | /dev/ublkbN exists | Yes |
| F002 | Device creation fails with size=0 | Create device with size=0 | Error returned | Yes |
| F003 | Device creation fails with size > physical RAM | Create 1PB device | Error or OOM | Yes |
| F004 | Device ID -1 auto-assigns | Create with dev_id=-1 | Valid ID assigned | Yes |
| F005 | Device ID collision rejected | Create two devices with same ID | Second fails | Yes |
| F006 | Device reset removes block device | Reset device 0 | /dev/ublkb0 gone | Yes |
| F007 | Device reset while I/O pending | Reset during fio run | Clean shutdown | Yes |
| F008 | Module unload with active device | rmmod with device mounted | Refused or clean | Yes |
| F009 | Daemon restart recovers device | Kill -9 daemon, restart | Device functional | Yes |
| F010 | Multiple devices coexist | Create devices 0, 1, 2 | All functional | Yes |
| F011 | Device survives daemon SIGTERM | Send SIGTERM to daemon | Clean shutdown | Yes |
| F012 | Device survives daemon SIGKILL | Send SIGKILL to daemon | Recoverable state | Yes |
| F013 | Queue depth configuration | Create with queue_depth=256 | 256 concurrent I/Os | Yes |
| F014 | Max I/O size configuration | Create with max_io=1M | 1MB I/Os succeed | Yes |
| F015 | Character device permissions | Check /dev/ublkcN perms | root:root 0600 | Yes |

### 5.2 I/O Operations (F016-F035)

| ID | Hypothesis | Falsification Test | Expected | Docker |
|----|------------|-------------------|----------|--------|
| F016 | Read returns written data | Write pattern, read back | Identical | Yes |
| F017 | Read unwritten sector returns zeros | Read sector 1000000 | All zeros | Yes |
| F018 | Partial page write preserves other data | Write 512B at offset 0, read full page | Partial preserved | Yes |
| F019 | Sequential write throughput > 1 GB/s | fio sequential write 4KB | > 1 GB/s | Yes |
| F020 | Random read IOPS > 100K | fio random read 4KB | > 100K IOPS | Yes |
| F021 | Concurrent I/O from multiple threads | 16 threads, random I/O | No corruption | Yes |
| F022 | I/O at device boundary | Write last sector | Success | Yes |
| F023 | I/O beyond device boundary | Write past last sector | Error | Yes |
| F024 | Zero-length I/O handled | Write 0 bytes | No-op or error | Yes |
| F025 | Very large I/O handled | Write 64MB at once | Success or chunked | Yes |
| F026 | Discard frees memory | Write, discard, check mem | Memory freed | Yes |
| F027 | Discard returns zeros on read | Discard sector, read | Zeros | Yes |
| F028 | Write-zeroes operation | Write-zeroes 1MB | Zeros on read | Yes |
| F029 | Flush operation | Flush after write | Data persisted | Yes |
| F030 | I/O error propagation | Induce OOM during write | Error returned | Yes |
| F031 | Timeout handling | Slow compression, check timeout | Handled gracefully | Yes |
| F032 | Queue full handling | Submit 10000 I/Os | Backpressure works | Yes |
| F033 | I/O alignment requirements | Unaligned 513B I/O | Error or handled | Yes |
| F034 | Direct I/O mode | O_DIRECT flag | Bypasses cache | Yes |
| F035 | Vectored I/O (readv/writev) | Submit scatter-gather | Correct assembly | Yes |

### 5.3 Compression (F036-F055)

| ID | Hypothesis | Falsification Test | Expected | Docker |
|----|------------|-------------------|----------|--------|
| F036 | LZ4 compression ratio > 2x on text | Compress /usr/share/dict/words | Ratio > 2x | Yes |
| F037 | ZSTD compression ratio > 3x on text | Compress dictionary | Ratio > 3x | Yes |
| F038 | Zero pages achieve max compression | Write all-zero page | Minimal storage | Yes |
| F039 | Random data stored uncompressed | Write /dev/urandom | Ratio ~1x | Yes |
| F040 | Compression preserves data | Compress, decompress, verify | Identical | Yes |
| F041 | SIMD path selected for medium entropy | Write semi-compressible | SIMD stats > 0 | Yes |
| F042 | GPU path selected for batch | Write 1000+ pages quickly | GPU stats > 0 | Yes* |
| F043 | Scalar path for high entropy | Write encrypted data | Scalar stats > 0 | Yes |
| F044 | Algorithm switch mid-stream | Change algorithm, continue | Works | Yes |
| F045 | Compression level configuration | Set level=1 vs level=9 | Different ratios | Yes |
| F046 | Dictionary compression | Provide dictionary | Improved ratio | Yes |
| F047 | Streaming compression | Large file in chunks | Correct output | Yes |
| F048 | Decompression after restart | Restart daemon, read old data | Correct data | No** |
| F049 | Corrupted compressed data detected | Flip bits in storage | Error on read | Yes |
| F050 | Compression buffer overflow | Incompressible data > buffer | Handled | Yes |
| F051 | Empty input handling | Compress 0 bytes | Valid output | Yes |
| F052 | Maximum input size | Compress 64MB page | Success | Yes |
| F053 | Parallel compression | 48 threads compressing | No race conditions | Yes |
| F054 | Compression statistics accurate | Compare stats to actual I/O | Match | Yes |
| F055 | Memory limit enforcement | Set 1GB limit, write 2GB | Limit respected | Yes |

*Requires GPU passthrough to Docker
**Requires persistent storage, not applicable to RAM-only device

### 5.4 io_uring Integration (F056-F070)

| ID | Hypothesis | Falsification Test | Expected | Docker |
|----|------------|-------------------|----------|--------|
| F056 | FETCH command receives I/O | Submit fetch, trigger I/O | Completion received | Yes |
| F057 | COMMIT_AND_FETCH chains operations | Commit result, fetch next | Both complete | Yes |
| F058 | io_uring submission queue handling | Fill SQ completely | No loss | Yes |
| F059 | io_uring completion queue handling | 1000 concurrent completions | All processed | Yes |
| F060 | URING_CMD16 format correct | Trace syscall, verify format | Matches spec | Yes |
| F061 | USER_COPY offset calculation | Trace pread/pwrite offsets | Correct per spec | Yes |
| F062 | Tag reuse after completion | Complete tag 0, reuse | Works | Yes |
| F063 | Queue ID isolation | Multi-queue device | Queues independent | Yes |
| F064 | EINTR handling | Send signals during I/O | Retry works | Yes |
| F065 | EAGAIN handling | Induce backpressure | Retry works | Yes |
| F066 | Timeout configuration | Set 1s timeout | Timeout fires | Yes |
| F067 | Linked operations | Link read after write | Correct ordering | Yes |
| F068 | Fixed buffers optimization | Register fixed buffers | Performance gain | Yes |
| F069 | Ring size configuration | Create ring size 4096 | Uses 4096 entries | Yes |
| F070 | Kernel version compatibility | Test on 6.0, 6.1, 6.5, 6.8 | All work | Yes |

### 5.5 Memory Management (F071-F085)

| ID | Hypothesis | Falsification Test | Expected | Docker |
|----|------------|-------------------|----------|--------|
| F071 | No memory leaks on normal operation | Valgrind 1-hour test | Zero leaks | Yes |
| F072 | No leaks on error paths | Induce errors, check memory | Zero leaks | Yes |
| F073 | Memory usage proportional to data | Write 1GB, check RSS | ~1GB + overhead | Yes |
| F074 | Memory freed on discard | Discard all, check RSS | Back to baseline | Yes |
| F075 | OOM killer integration | Hit memory limit | Clean termination | Yes |
| F076 | mmap regions properly unmapped | Close device, check /proc/maps | No orphaned mappings | Yes |
| F077 | Page-aligned allocations | Check all alloc addresses | 4K aligned | Yes |
| F078 | NUMA awareness | Run on NUMA system | Local memory preferred | No*** |
| F079 | Huge page support | Enable THP, run tests | Uses huge pages | Yes |
| F080 | Memory cgroup integration | Set memory.max, test | Limit respected | Yes |
| F081 | Swap interaction | Enable swap, fill device | Swaps gracefully | Yes |
| F082 | mlock for latency-critical | mlock data buffers | No page faults | Yes |
| F083 | Buffer pool efficiency | Reuse buffers | No excessive alloc | Yes |
| F084 | Stack usage reasonable | Check stack size | < 1MB per thread | Yes |
| F085 | Thread-local storage | Check TLS usage | No contention | Yes |

***Requires NUMA hardware

### 5.6 Filesystem Integration (F086-F095)

| ID | Hypothesis | Falsification Test | Expected | Docker |
|----|------------|-------------------|----------|--------|
| F086 | ext4 mkfs succeeds | mkfs.ext4 /dev/ublkb0 | Success | Yes |
| F087 | ext4 mount/umount cycle | Mount, write, umount, remount | Data intact | Yes |
| F088 | btrfs mkfs succeeds | mkfs.btrfs /dev/ublkb0 | Success | Yes |
| F089 | btrfs checksums pass | btrfs scrub | No errors | Yes |
| F090 | XFS mkfs succeeds | mkfs.xfs /dev/ublkb0 | Success | Yes |
| F091 | F2FS mkfs succeeds | mkfs.f2fs /dev/ublkb0 | Success | Yes |
| F092 | Swap mkswap/swapon | mkswap, swapon | Active swap | Yes |
| F093 | Swap stress test | stress-ng --vm 4 | No corruption | Yes |
| F094 | Swap off gracefully | swapoff after use | Clean | Yes |
| F095 | LVM on ublk | pvcreate, vgcreate | Works | Yes |

### 5.7 Error Handling and Recovery (F096-F100)

| ID | Hypothesis | Falsification Test | Expected | Docker |
|----|------------|-------------------|----------|--------|
| F096 | Graceful degradation under pressure | 100% CPU during I/O | Slower but correct | Yes |
| F097 | Recovery from transient errors | Inject EIO, retry | Eventually succeeds | Yes |
| F098 | Logging captures errors | Induce error, check logs | Error logged | Yes |
| F099 | Metrics updated on error | Induce error, check stats | failed_* incremented | Yes |
| F100 | Panic handler cleanup | panic!() in handler | Resources freed | Yes |

### 5.8 Fuzzing and Security (F101-F105)

| ID | Hypothesis | Falsification Test | Expected | Docker |
|----|------------|-------------------|----------|--------|
| F101 | Input validation resists random data | Fuzz `UblkIoDesc` parsing | No panic | Yes |
| F102 | Config parser resists malformed TOML | Fuzz configuration loader | Error, no panic | Yes |
| F103 | Compressed data decoder is safe | Fuzz decompressor with random bits | Error, no panic | Yes |
| F104 | CLI args parser is robust | Fuzz CLI arguments | Safe exit | Yes |
| F105 | Command injection prevented | Fuzz device names/paths | No injection | Yes |

### 5.9 Statistical Performance Verification (F106-F110)

| ID | Hypothesis | Falsification Test | Expected | Docker |
|----|------------|-------------------|----------|--------|
| F106 | Latency stable under load | 1hr run, t-test vs baseline | p > 0.05 (no regression) | Yes |
| F107 | Throughput stable | 1hr run, t-test vs baseline | p > 0.05 | Yes |
| F108 | Memory variance bounded | Measure RSS variance | < 5% variance | Yes |
| F109 | Compression ratio stability | Compress standard corpus | Low variance | Yes |
| F110 | Context switch overhead | Measure involuntary switches | < Baseline + 10% | Yes |

---

## 6. Debugging Procedures

### 6.1 Tracing with Renacer

All tracing MUST leverage the **renacer** structured observability framework.

```bash
# Enable renacer trace collection
export RENACER_TRACE=1
export RENACER_LEVEL=debug

# Filter for specific subsystems (e.g., GPU pipeline, IO)
export RENACER_FILTER="gpu,io"

# Live trace view via TUI
trueno-ublk monitor --trace
```

Legacy `strace` and kernel tracing can be used for low-level validation:

```bash
# Enable ublk kernel tracing
echo 1 > /sys/kernel/debug/tracing/events/ublk/enable
cat /sys/kernel/debug/tracing/trace_pipe

# Trace specific daemon
strace -f -e trace=ioctl,read,write,pread64,pwrite64 \
    trueno-ublk create --size 1G --foreground
```

### 6.2 Debugging USER_COPY Mode

The bug likely involves USER_COPY mode, where data is transferred via pread/pwrite on the character device rather than shared memory. Verify offsets:

```rust
// Add debug logging to daemon.rs
fn submit_fetch(&mut self, tag: u16) -> Result<(), DaemonError> {
    eprintln!("[DEBUG] FETCH: tag={}, q_id=0", tag);
    // ...
}

fn handle_read(&mut self, tag: u16, iod: &UblkIoDesc) -> i32 {
    let offset = ublk_user_copy_offset(0, tag, 0);
    eprintln!("[DEBUG] READ: tag={}, offset=0x{:X}, sectors={}",
              tag, offset, iod.nr_sectors);
    // ...
}
```

### 6.3 Verifying Offset Calculation

Compare with libublk reference implementation:

```c
// From libublk: ublk_pos()
static inline __u64 ublk_pos(__u16 q_id, __u16 tag, __u32 offset)
{
    return UBLKSRV_IO_BUF_OFFSET +
           ((__u64)q_id << UBLK_QID_OFF) +
           ((__u64)tag << UBLK_TAG_OFF) +
           offset;
}

// UBLKSRV_IO_BUF_OFFSET = 0x80000000 (2GB)
// UBLK_QID_OFF = 32
// UBLK_TAG_OFF = 16
```

---

## 7. Troubleshooting Guide

### 7.1 Device Won't Start

**Symptom:** `trueno-ublk create` hangs or fails

**Checklist:**
1. Is ublk_drv loaded? `lsmod | grep ublk`
2. Does /dev/ublk-control exist? `ls -la /dev/ublk-control`
3. Running as root? `id`
4. Kernel version 6.0+? `uname -r`

### 7.2 I/O Errors

**Symptom:** `dd` or filesystem operations fail with I/O errors

**Checklist:**
1. Check dmesg: `dmesg | grep -i ublk`
2. Check daemon logs: `journalctl -u trueno-ublk`
3. Verify device exists: `lsblk | grep ublk`
4. Run basic I/O test in Docker first (see Section 4)

### 7.3 Orphaned Swap

**Symptom:** `/proc/swaps` shows deleted device

**Resolution:**
```bash
# Force swap pages to RAM (needs free memory)
sudo swapoff -a  # Turn off all swap temporarily
sudo swapon /swapfile  # Re-enable file swap

# If swapoff hangs, reboot is required
```

### 7.4 Module Won't Unload

**Symptom:** `rmmod ublk_drv` fails with "in use"

**Resolution:**
```bash
# Find what's using it
lsof | grep ublk
fuser -v /dev/ublk*

# Kill daemon
pkill trueno-ublk

# Reset all devices
for i in $(seq 0 9); do trueno-ublk reset $i 2>/dev/null; done

# Now unload
rmmod ublk_drv
```

---

## 8. Peer-Reviewed References

[1] T. Ohno, "Toyota Production System: Beyond Large-Scale Production," Productivity Press, 1988. ISBN: 978-0915299140. *Foundational text on Five-Whys and root cause analysis.*

[2] J. K. Liker, "The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer," McGraw-Hill, 2004. ISBN: 978-0071392310. *Comprehensive treatment of TPS principles applicable to software engineering.*

[3] M. Cohn, "Succeeding with Agile: Software Development Using Scrum," Addison-Wesley, 2009. ISBN: 978-0321579362. *Introduces the test pyramid concept.*

[4] D. Merkel, "Docker: Lightweight Linux Containers for Consistent Development and Deployment," Linux Journal, vol. 2014, no. 239, 2014. *Foundational paper on container isolation.*

[5] K. Popper, "The Logic of Scientific Discovery," Routledge, 1959 (originally 1934). ISBN: 978-0415278447. *Philosophical basis for falsifiability in testing.*

[6] J. Axboe, "Efficient IO with io_uring," Kernel Recipes, 2019. https://kernel.dk/io_uring.pdf. *Authoritative reference on io_uring design.*

[7] M. Lei, "ublk: Userspace Block Device Driver," LWN.net, 2022. https://lwn.net/Articles/903855/. *Technical overview of ublk kernel interface.*

[8] Intel Corporation, "Intel 64 and IA-32 Architectures Optimization Reference Manual," 2023. *SIMD optimization guidelines for LZ4 implementation.*

[9] Y. Collet, "LZ4 Frame Format Description," 2021. https://github.com/lz4/lz4/blob/dev/doc/lz4_Frame_format.md. *Canonical LZ4 specification.*

[10] Facebook, "Zstandard Compression Format," RFC 8478, 2018. https://tools.ietf.org/html/rfc8478. *IETF standardization of ZSTD.*

[11] H. Kim et al., "QZRAM: A High-Performance Memory Expansion Scheme Based on ZRAM," Electronics, vol. 12, no. 4, 2023. *Analysis of ZRAM performance and optimization strategies.*

---

## 9. Verification of Specification

### 9.1 Meta-Verification Checklist

To ensure this specification remains valid and useful:

1.  **Link Integrity**: All URLs in references MUST be reachable.
2.  **Test Coverage**: Every hypothesis (F001-F110) MUST have a corresponding test case implemented in the `tests/` directory.
3.  **Docker Reproducibility**: The Docker environment defined in Section 4 MUST build and run on a clean host without error.
4.  **Citation Accuracy**: All references MUST be verified against original sources.

### 9.2 Verification Status (2026-01-06)

| Component | Status | Details |
|-----------|--------|---------|
| Unit Tests | ✓ PASS | 370 tests passed |
| Integration Tests | ✓ PASS | 10 tests passed |
| F001 Device Creation | ✓ PASS | Verified with fresh device IDs |
| F016 Read-after-write | ✓ PASS | MD5 data integrity verified (400KB test) |
| F086 mkfs.ext4 | ✓ PASS | Filesystem creation succeeds |
| F092 Swap | ✓ PASS | Full cycle: mkswap→swapon→swapoff→mkswap→swapon verified |
| F083 I/O Bug | ✓ RESOLVED | Root cause: stale kernel state |
| Docker Infrastructure | ✓ BUILT | Dockerfile.ublk-test builds successfully |
| DT-003 Graceful Shutdown | ✓ PASS | cleanup.rs module with signal handlers |
| Orphan Detection | ✓ PASS | Detects stale character devices |
| **DT-005 Production Swap** | ✓ PASS | 8GB device, 185MB swapped under memory pressure |

### 9.3 Swap Testing Investigation (DT-004) - 2026-01-06 - RESOLVED

**Status:** RESOLVED - Bug fixed, swap working

#### Test Sequence and Results

| Step | Command | Result |
|------|---------|--------|
| 1 | Device creation (4GB) | ✓ SUCCESS - /dev/ublkb0 created |
| 2 | Basic I/O test (400KB) | ✓ SUCCESS - MD5 verified |
| 3 | First mkswap | ✓ SUCCESS |
| 4 | First swapon -p 150 | ✓ SUCCESS - Swap active |
| 5 | swapoff | ✓ SUCCESS |
| 6 | Second mkswap | ✗ FAIL - "unable to erase bootbits sectors" |
| 7 | xxd /dev/ublkb0 | ✗ FAIL - "Input/output error" |
| 8 | Second swapon | ✗ FAIL - "read swap header failed" |

#### Observations

- Daemon process still running (PID visible)
- /proc/diskstats shows I/O activity: 122 reads, 5 writes
- No I/O operation logs in daemon output after initial FETCH submission
- Device appears responsive initially but fails after swapon/swapoff cycle

#### ROOT CAUSE IDENTIFIED (2026-01-06 10:28)

Daemon log revealed **LZ4 DECOMPRESSION FAILURE**:
```
WARN trueno_ublk::ublk::daemon: Batched I/O operation failed tag=91
     error=corrupted data: offset 19251 exceeds output position 0
```

**Analysis:**
1. mkswap writes swap header → stored compressed in page store
2. swapon reads header → decompression succeeds
3. swapoff clears swap state
4. Second mkswap attempts to "erase bootbits" → tries to READ sector
5. READ triggers decompression of previously stored data
6. LZ4 decompression fails with invalid offset reference
7. I/O error returned to kernel → device becomes unresponsive

**Root Cause:** ALGORITHM MISMATCH BUG in `compress_batch_direct()`

**File:** `bins/trueno-ublk/src/daemon.rs` lines 629-640

**Bug:** When compression doesn't save space (`compressed.len() >= PAGE_SIZE`):
```rust
let compressed_page = if compressed.len() >= PAGE_SIZE {
    CoreCompressedPage {
        data: page.to_vec(),   // RAW uncompressed data
        original_size: PAGE_SIZE,
        algorithm,             // BUG: Still set to LZ4!
    }
}
```

**Effect:**
- Raw bytes stored with `algorithm = LZ4`
- On read, `decompress()` interprets raw data as LZ4 format
- First byte of raw data (e.g., 0x4B = 'K') parsed as LZ4 token
- Token indicates match offset, but raw data isn't LZ4 encoded
- Result: "offset 19251 exceeds output position 0"

**Fix:** Set `algorithm = Algorithm::None` when storing uncompressed data

#### Resolution

1. ✓ Gracefully terminated daemon with SIGTERM
2. ✓ Cleaned up orphaned character device
3. ✓ Identified root cause in `daemon.rs:629-640` and `daemon.rs:738-745`
4. ✓ Implemented fix: `algorithm: Algorithm::None` for incompressible pages
5. ✓ Verified swap works: mkswap → swapon → swapoff → mkswap → swapon ALL PASSED
6. DT-004 COMPLETE

**Status:** RESOLVED

### 9.4 Swap Deadlock Discovery (DT-006) - 2026-01-06

**Status:** ROOT CAUSE IDENTIFIED - Fix delegated to DT-007 (duende mlock integration)

#### Failure Scenario

During stress test with ruchy `make coverage` (11254 tests):
- Memory pressure: 115GB buff/cache, 265MB swap used
- Result: System lockup, daemon in state:D (uninterruptible sleep)

#### Root Cause: Swap Deadlock

```
INFO: task trueno-ublk:59497 blocked for more than 122 seconds.
task:trueno-ublk state:D (uninterruptible sleep)
__swap_writepage+0x111/0x1a0
swap_writepage+0x5f/0xe0
```

**Deadlock Pattern:**
1. Kernel needs to swap pages OUT to trueno-ublk device
2. trueno-ublk daemon needs memory to process I/O
3. To get memory, kernel tries to swap OUT daemon's pages
4. But swap writes go to trueno-ublk device (DEADLOCK!)

#### Required Fix: mlock()

The daemon must pin its own memory to prevent being swapped out while serving swap I/O:

```rust
// Required in daemon init:
unsafe { libc::mlockall(libc::MCL_CURRENT | libc::MCL_FUTURE); }
```

This fix has been delegated to the duende project (DT-007) which will provide cross-platform daemon lifecycle management including mlock() capability.

### 9.5 Docker Limitation Discovery - 2026-01-06

**Status:** CONFIRMED - ublk devices cannot be isolated in Docker

#### Finding

ublk devices are **host kernel resources**:
- `/dev/ublk-control` is a host kernel interface
- `/dev/ublkbN` block devices exist in host namespace
- Docker `--privileged` + `/dev` mount exposes full host RAM/swap
- Container memory limits do NOT apply to ublk I/O

#### Implications

1. Docker-based stress testing of swap behavior is **not possible**
2. Must test ublk swap on host with controlled parameters
3. CI/CD can test non-privileged paths (compression, algorithms)
4. Integration tests require dedicated test VM or bare metal

#### Recommended Test Strategy

For swap-related testing, use controlled host tests:
- Small device (1GB) to limit blast radius
- Targeted swap fill (not full RAM pressure)
- Quick timeout detection for deadlock
- Monitor daemon `state:D` via `/proc/[pid]/status`

*Document generated following Toyota Way principles. Last updated: 2026-01-06.*
