# PERF-003: Multi-Queue Parallelism Specification

**Status:** P0 - CRITICAL | **PHASE 1 COMPLETE** (Config plumbing)
**Created:** 2026-01-06
**Target:** 4x IOPS improvement via parallel hardware queues

## Implementation Status

### Phase 1: Config Plumbing ✅ COMPLETE
- [x] CLI `--queues` flag (1-8)
- [x] `device::DeviceConfig.nr_hw_queues`
- [x] `BatchedDaemonConfig.nr_hw_queues`
- [x] `ublk::ctrl::DeviceConfig.nr_hw_queues`
- [x] TDD tests (6 tests passing)

### Phase 2: Multi-Queue Infrastructure ✅ COMPLETE
- [x] `QueueWorkerConfig` with per-queue offsets
- [x] `MultiQueueStats` for per-queue and aggregate statistics
- [x] `MultiQueueDaemon` structure and coordination
- [x] `run_multi_queue_daemon` function scaffolding
- [x] `optimal_queue_count()` based on available CPUs
- [x] `pin_to_cpu()` for thread affinity
- [x] 8 new TDD tests (501 total tests passing)

### Phase 3: Full Per-Queue Threading ✅ COMPLETE
- [x] Per-queue io_uring instances (`QueueIoWorker`)
- [x] Per-queue mmap regions at correct offsets
- [x] Per-queue I/O processing threads (`spawn_queue_workers`)
- [x] USER_COPY mode with pread/pwrite
- [x] SendPtr wrapper for cross-thread pointer safety
- [x] Integration with `run_daemon_batched` (`run_multi_queue_batched_internal`)
- [x] 501 tests passing

### Phase 4: Live Testing & Benchmarks (Pending)
- [ ] Live benchmark with fio (4 queues)
- [ ] Verify IOPS scaling with queue count

## Problem Statement

Current architecture uses **single hardware queue** (`nr_hw_queues=1`), serializing all I/O through one io_uring instance. This is the primary bottleneck preventing us from reaching Ming Lei's 1.2M IOPS benchmark.

**Current State:**
- 162K IOPS (single queue, polling mode)
- ~200K theoretical max for single queue

**Target:**
- 500K+ IOPS with 4 queues
- 800K+ IOPS with 8 queues (approaching Ming Lei's benchmark)

## Architecture

### Current (Single Queue)
```
[Block I/O] → [Queue 0] → [io_uring] → [Daemon Thread] → [Compress] → [Reply]
```

### PERF-003 (Multi-Queue)
```
[Block I/O] → [Queue 0] → [io_uring 0] → [Thread 0] → [Compress] → [Reply]
            → [Queue 1] → [io_uring 1] → [Thread 1] → [Compress] → [Reply]
            → [Queue 2] → [io_uring 2] → [Thread 2] → [Compress] → [Reply]
            → [Queue 3] → [io_uring 3] → [Thread 3] → [Compress] → [Reply]
```

Each queue:
1. Has its own io_uring instance
2. Runs in dedicated thread pinned to a CPU core
3. Has independent IOD/data buffers (mmap at different offsets)
4. Shares the BatchedPageStore (thread-safe via RwLock)

## Implementation

### Phase 1: Multi-Queue Daemon

1. **`MultiQueueDaemon`** struct:
   - `queues: Vec<QueueWorker>` - One worker per hardware queue
   - `store: Arc<BatchedPageStore>` - Shared page store
   - `stop: Arc<AtomicBool>` - Shutdown signal

2. **`QueueWorker`** struct:
   - `queue_id: u16`
   - `ring: IoUring` - Per-queue io_uring
   - `iod_buf: *mut u8` - Queue-specific IOD buffer
   - `data_buf: *mut u8` - Queue-specific data buffer
   - `thread: JoinHandle<()>`

3. **Buffer offsets** (per ublk spec):
   ```rust
   let iod_offset = queue_id * queue_depth * sizeof(UblkIoDesc);
   let data_offset = iod_area_size + queue_id * queue_depth * max_io_size;
   ```

### Phase 2: CLI Integration

```bash
# Default: auto-detect optimal queue count
trueno-ublk create --size 8G --max-perf

# Explicit: 4 hardware queues
trueno-ublk create --size 8G --queues 4

# Maximum: 8 queues with per-queue CPU pinning
trueno-ublk create --size 8G --queues 8 --cpu-affinity "0-7"
```

### Phase 3: Optimal Queue Count

Auto-detection based on:
- `num_cpus::get()` for available cores
- NUMA topology for local cores
- Maximum of 8 queues (kernel limit)

## Success Criteria

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Random Read IOPS | 162K | 500K+ | 3x+ |
| Random Write IOPS | 90K | 300K+ | 3x+ |
| Sequential Read | 5.6 GB/s | 8 GB/s | 1.4x |
| Sequential Write | 1 GB/s | 3 GB/s | 3x |

## TDD Test Plan

### Unit Tests

1. `test_perf003_multi_queue_config` - Verify DeviceConfig accepts nr_hw_queues
2. `test_perf003_queue_buffer_offsets` - Verify correct mmap offsets per queue
3. `test_perf003_queue_isolation` - Verify queues operate independently
4. `test_perf003_shared_store_thread_safety` - Verify BatchedPageStore under concurrent access

### Integration Tests

1. `test_perf003_4queue_iops` - Verify 4-queue setup achieves 3x IOPS improvement
2. `test_perf003_cpu_affinity_per_queue` - Verify each queue thread is pinned

## References

- Ming Lei's ublk benchmark: https://lpc.events/event/16/contributions/1345/
- ublk multi-queue: https://docs.kernel.org/block/ublk.html#multi-queue-support
- io_uring SQ polling: https://kernel.dk/io_uring.pdf
