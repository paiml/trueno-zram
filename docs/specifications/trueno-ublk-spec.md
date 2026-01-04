# trueno-ublk Specification: Userspace Block Device with SIMD Acceleration

**Version:** 2.3.0
**Date:** 2026-01-04
**Status:** Active Development - QA Mandated

## 1. Abstract

`trueno-ublk` is a pure Rust userspace block device implementing ZRAM-like compressed RAM storage. By leveraging `io_uring` for asynchronous I/O and SIMD-accelerated compression (AVX-512, NEON), it aims to maximize storage efficiency and throughput while minimizing CPU overhead. This specification adopts the *Toyota Way* principles [Liker 2004], Karl Popper's philosophy of falsification [Popper 1959], and a mandatory automated QA checklist (PMAT) with extreme TDD requirements.

## 2. Design Philosophy: The Toyota Way & PMAT

Our architectural decisions are grounded in the 14 principles of the Toyota Way and enforced by the PMAT (Automated Quality Checklist) framework.

### 2.1 Base Principles
*   **Principle 1: Base your management decisions on a long-term philosophy.**
    *   *Application:* We prioritize data integrity and code maintainability. The "Pure Rust" approach ensures memory safety, avoiding the technical debt of C-bindings.
*   **Principle 2: Create continuous process flow to bring problems to the surface.**
    *   *Application:* The I/O pipeline is designed as a continuous stream using `io_uring`. PMAT metrics monitor queue depth and latency to immediately surface bottlenecks.

### 2.2 Quality & Testing (Jidoka)
*   **Principle 5: Build a culture of stopping to fix problems.**
    *   *Application:* Extreme TDD (Test-Driven Development) is mandatory. Every line of code must be justified by a failing test.
*   **QA Mandate:**
    *   **Coverage:** Minimum **>95% code coverage** for the core I/O and compression logic.
    *   **TUI Testing:** 100% **probador-tested TUI** using the `jugar-probar` framework to ensure UI reliability and metric accuracy.
    *   **PMAT Compliance:** All work must pass the 10-point automated QA checklist before being merged.

## 3. Architecture

### 3.1 High-Level Design

```
┌─────────────────────────────────────────────────────────┐
│                    Application                           │
│              (file I/O, mmap, etc.)                      │
└───────────────────────┬─────────────────────────────────┘
                        │ /dev/ublkb0
┌───────────────────────▼─────────────────────────────────┐
│                   Linux Kernel                           │
│              (ublk driver + io_uring)                    │
└───────────────────────┬─────────────────────────────────┘
                        │ io_uring (submission/completion)
┌───────────────────────▼─────────────────────────────────┐
│              trueno-ublk (Pure Rust)                     │
│  ┌─────────────────────────────────────────────────────┐│
│  │               UblkTarget (Async I/O)                 ││
│  │  - Handles ublk_queue commands                      ││
│  │  - Zero-copy buffer management                      ││
│  └─────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────┐│
│  │               PageStore (The "Gemba")                ││
│  │  - L1: Active Page Cache (Hot)                      ││
│  │  - L2: Compressed Storage (LZ4/Zstd)                ││
│  │  - L3: Zero-Page Sentinel                           ││
│  └─────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────┐│
│  │              trueno-zram-core                        ││
│  │  - SIMD Dispatcher (AVX-512/NEON)                   ││
│  │  - Entropy Analyzer [Shannon 1948]                  ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

## 4. Renacer Verification Matrix: A 100-Point Poppian Checklist

This checklist is designed to *falsify* the system. Minimum coverage for Section A and B must exceed 95%.

### Section A: Data Integrity (The "Safety" Zone)
1.  [ ] Write 4KB pattern A, Read, Verify.
2.  [ ] Write 4KB pattern A, Write 4KB pattern B, Read, Verify B.
3.  [ ] Write 4KB zero-page, Read, Verify Zero.
4.  [ ] Write 4KB random high-entropy (uncompressible), Read, Verify.
5.  [ ] Write 4KB repeated byte (highly compressible), Read, Verify.
6.  [ ] Write to last sector of device boundary.
7.  [ ] Read from last sector of device boundary.
8.  [ ] Write past device boundary (expect error).
9.  [ ] Read past device boundary (expect error).
10. [ ] Read uninitialized sector (expect zeros).
11. [ ] Write 1 byte, Read 4KB (partial update logic).
12. [ ] Write 4KB, Read 1 byte (partial read logic).
13. [ ] Overwrite scalar compressed page with SIMD compressed page.
14. [ ] Overwrite SIMD compressed page with zero page.
15. [ ] Overwrite zero page with scalar compressed page.
16. [ ] Persistence: Simulate crash (kill -9), restart, verify data (if persistent backend used).
17. [ ] CRC32 check injection: Manually corrupt compressed data in RAM, Read (expect checksum error).
18. [ ] Concurrent Read/Write to same sector (Atomicity check).
19. [ ] Concurrent Write/Write to same sector (Last writer wins check).
20. [ ] Verify `discard` (TRIM) command zeroes data and frees memory.

### Section B: Resource Management (The "Muda" Zone)
21. [ ] Leak check: Create/Destroy device 1000 times (RSS stable?).
22. [ ] Zero-page deduplication: Write 1GB zeros, RSS usage < 1MB.
23. [ ] Compression ratio: Write 1GB text, RSS usage < 600MB.
24. [ ] Max connections: Create max allowed ublk devices (kernel limit).
25. [ ] OOM resilience: Restrict cgroup memory, write until full (graceful error?).
26. [ ] CPU Pinning: Verify threads adhere to affinity mask.
27. [ ] File Descriptor check: Ensure no FD leaks after reset.
28. [ ] Buffer pool exhaustion: saturate I/O depth, check for dropped requests.
29. [ ] Idle timeout: Verify resources release/sleep on idle.
30. [ ] Fragmentation: Random write pattern for 1 hour, check memory fragmentation.

### Section C: Performance & Scalability (The "Flow" Zone)
31. [ ] Throughput > 2GB/s sequential write (SIMD enabled).
32. [ ] Throughput > 4GB/s sequential read (SIMD enabled).
33. [ ] IOPS > 100k random 4K write.
34. [ ] IOPS > 200k random 4K read.
35. [ ] Latency p99 < 500us at QD=1.
36. [ ] Latency p99 < 2ms at QD=128.
37. [ ] Scalability: Linear scaling up to physical core count.
38. [ ] AVX-512 vs AVX2 fallback verification.
39. [ ] Algorithm switch: Swap LZ4 -> Zstd runtime, verify impact.
40. [ ] Dictionary training: Verify dictionary hit rate improvement.

### Section D: Chaos & Fuzzing (The "Entropy" Zone)
41. [ ] Bit-flip fuzzing on input buffer.
42. [ ] Length fuzzing (0, 1, 4095, 4097 bytes).
43. [ ] Alignment fuzzing (read/write at non-4k offsets).
44. [ ] `io_uring` submission queue overflow.
45. [ ] `io_uring` completion queue overflow.
46. [ ] Signal interruption (SIGINT/SIGTERM) during heavy I/O.
47. [ ] Device disconnect simulation.
48. [ ] Kernel module unload simulation (modprobe -r ublk_drv).
49. [ ] High system load interference (stress-ng in background).
50. [ ] NUMA node mismatch (force cross-node access).

### Section E: CLI & Usability (The "Standardization" Zone)
51. [ ] `create` with invalid algorithm (friendly error?).
52. [ ] `create` with invalid size (friendly error?).
53. [ ] `list` output matches JSON schema.
54. [ ] `stat` reflects accurate compression ratio.
55. [ ] `reset` cleans up all resources.
56. [ ] `help` text is present for all subcommands.
57. [ ] `version` matches Cargo.toml.
58. [ ] Log levels (RUST_LOG=trace) produce actionable debug info.
59. [ ] **Probador TUI Check:** TUI dashboard renders without flickering (Verified by `jugar-probar`).
60. [ ] **Probador TUI Check:** TUI handles window resize gracefully (Verified by `jugar-probar`).

### Section F-J: (Remaining 40 points omitted for brevity, focusing on QA Mandates)
...
91. [ ] **PMAT Checklist:** All tests passed on fresh VM.
92. [ ] **PMAT Checklist:** >95% code coverage for `PageStore`.
93. [ ] **PMAT Checklist:** >95% code coverage for `UblkTarget`.
94. [ ] **PMAT Checklist:** 100% of TUI components have `jugar-probar` test cases.

## 5. ublk Kernel Interface (Linux 6.0+)

### 5.1 Control Device Architecture

The ublk kernel driver (`ublk_drv`) exposes `/dev/ublk-control` for device lifecycle management. **Critical:** Starting with Linux 6.0, all control commands use `io_uring` with `IORING_OP_URING_CMD`, NOT legacy ioctl.

```
┌─────────────────────────────────────────────────────────────┐
│                    trueno-ublk daemon                        │
│  ┌───────────────────────┐    ┌───────────────────────────┐ │
│  │   Control Path        │    │   I/O Path                │ │
│  │   /dev/ublk-control   │    │   /dev/ublkcN             │ │
│  │   io_uring URING_CMD  │    │   io_uring URING_CMD      │ │
│  └───────────┬───────────┘    └─────────────┬─────────────┘ │
└──────────────┼──────────────────────────────┼───────────────┘
               │                              │
┌──────────────▼──────────────────────────────▼───────────────┐
│                    Linux Kernel (ublk_drv)                   │
│  ┌───────────────────────┐    ┌───────────────────────────┐ │
│  │   Control Device      │    │   Char Device (per-dev)   │ │
│  │   ADD/DEL/START/STOP  │    │   FETCH_REQ/COMMIT_REQ    │ │
│  └───────────────────────┘    └───────────────────────────┘ │
│                              ┌───────────────────────────┐   │
│                              │   Block Device            │   │
│                              │   /dev/ublkbN             │   │
│                              └───────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Control Commands (via io_uring)

All control commands are submitted via `io_uring` using `IORING_OP_URING_CMD`:

| Command | Opcode | Description |
|---------|--------|-------------|
| `UBLK_CMD_ADD_DEV` | 0x04 | Create new ublk device |
| `UBLK_CMD_DEL_DEV` | 0x05 | Delete ublk device |
| `UBLK_CMD_START_DEV` | 0x06 | Start device (after I/O queue ready) |
| `UBLK_CMD_STOP_DEV` | 0x07 | Stop device |
| `UBLK_CMD_SET_PARAMS` | 0x08 | Set device parameters |
| `UBLK_CMD_GET_PARAMS` | 0x09 | Get device parameters |
| `UBLK_CMD_GET_DEV_INFO` | 0x02 | Get device info |
| `UBLK_CMD_GET_QUEUE_AFFINITY` | 0x03 | Get queue CPU affinity |

### 5.3 Control Command Submission

```rust
// io_uring SQE setup for control commands
fn submit_ctrl_cmd(ring: &mut IoUring, fd: RawFd, cmd: u32, data: &UblkCtrlCmd) {
    let sqe = opcode::UringCmd80::new(types::Fd(fd), cmd)
        .cmd(unsafe { std::mem::transmute::<_, [u8; 80]>(*data) })
        .build();
    unsafe { ring.submission().push(&sqe).unwrap(); }
}
```

### 5.4 Kernel Structures

```rust
/// Control command payload (80 bytes, fits in SQE cmd field)
#[repr(C)]
pub struct UblkCtrlCmd {
    pub dev_id: u32,           // Device ID (-1 for auto-assign)
    pub queue_id: u16,         // Queue ID (for queue-specific ops)
    pub len: u16,              // Length of addr buffer
    pub addr: u64,             // Pointer to data buffer
    pub data: [u64; 2],        // Command-specific data
    pub reserved: [u8; 48],    // Reserved for future use
}

/// Device info (returned by ADD_DEV, GET_DEV_INFO)
#[repr(C)]
pub struct UblkCtrlDevInfo {
    pub nr_hw_queues: u16,     // Number of hardware queues
    pub queue_depth: u16,      // Queue depth per queue
    pub state: u16,            // Device state
    pub pad0: u16,
    pub max_io_buf_bytes: u32, // Max I/O buffer size
    pub dev_id: u32,           // Assigned device ID
    pub ublksrv_pid: i32,      // Server PID
    pub pad1: u32,
    pub flags: u64,            // Device flags (UBLK_F_*)
    pub ublksrv_flags: u64,    // Server flags
    pub owner_uid: u32,
    pub owner_gid: u32,
    pub reserved: [u64; 2],
}

/// Device parameters
#[repr(C)]
pub struct UblkParams {
    pub len: u32,              // Total params length
    pub types: u32,            // Bitmask of UBLK_PARAM_TYPE_*
    pub basic: UblkParamBasic,
    pub discard: UblkParamDiscard,
    pub devt: UblkParamDevt,
}
```

### 5.5 Device Flags

| Flag | Value | Description |
|------|-------|-------------|
| `UBLK_F_SUPPORT_ZERO_COPY` | 1 << 0 | Zero-copy I/O support |
| `UBLK_F_URING_CMD_COMP_IN_TASK` | 1 << 1 | Complete in task context |
| `UBLK_F_NEED_GET_DATA` | 1 << 2 | Need GET_DATA for writes |
| `UBLK_F_USER_RECOVERY` | 1 << 3 | User-controlled recovery |
| `UBLK_F_USER_RECOVERY_REISSUE` | 1 << 4 | Reissue on recovery |
| `UBLK_F_UNPRIVILEGED_DEV` | 1 << 5 | Unprivileged device |
| `UBLK_F_CMD_IOCTL_ENCODE` | 1 << 6 | Encode cmd in ioctl format |
| `UBLK_F_USER_COPY` | 1 << 7 | Userspace handles copy |
| `UBLK_F_ZONED` | 1 << 8 | Zoned block device |

### 5.6 I/O Path Commands

I/O commands are submitted on the per-device char device (`/dev/ublkcN`):

| Command | Opcode | Description |
|---------|--------|-------------|
| `UBLK_IO_FETCH_REQ` | 0x20 | Fetch next I/O request |
| `UBLK_IO_COMMIT_AND_FETCH_REQ` | 0x21 | Commit result and fetch next |
| `UBLK_IO_NEED_GET_DATA` | 0x22 | Request write data (if F_NEED_GET_DATA) |

### 5.7 Device Lifecycle

```
1. Open /dev/ublk-control
2. Submit UBLK_CMD_ADD_DEV via io_uring → kernel creates /dev/ublkcN
3. Submit UBLK_CMD_SET_PARAMS via io_uring → configure device
4. Open /dev/ublkcN (char device)
5. Submit UBLK_IO_FETCH_REQ for each queue slot
6. Submit UBLK_CMD_START_DEV via io_uring → kernel creates /dev/ublkbN
7. Process I/O loop: handle requests, submit UBLK_IO_COMMIT_AND_FETCH_REQ
8. On shutdown: UBLK_CMD_STOP_DEV, UBLK_CMD_DEL_DEV
```

### 5.8 Zero External Dependencies

trueno-ublk implements the ublk interface with **zero external ublk libraries**:

- **io-uring crate:** Only dependency for io_uring syscall wrapper
- **nix crate:** POSIX primitives (mmap, fork, signals)
- **No libublk:** Direct kernel interface implementation
- **No smol/tokio:** Synchronous io_uring event loop

This minimizes attack surface and ensures ABI stability across kernel versions.

## 6. References

1.  **Liker, J. K. (2004).** *The Toyota Way.* McGraw-Hill.
2.  **Popper, K. (1959).** *The Logic of Scientific Discovery.* Hutchinson.
3.  **Gregg, B. (2020).** *Systems Performance.* Pearson.
4.  **Shannon, C. E. (1948).** *A Mathematical Theory of Communication.*
5.  **Linux Kernel.** *include/uapi/linux/ublk_cmd.h* - ublk userspace API.
6.  **Lei, M. (2022).** *ublk_drv: add io_uring based userspace block driver.* Linux kernel commit.
