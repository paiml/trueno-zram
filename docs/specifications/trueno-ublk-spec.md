# trueno-ublk Specification

## Overview

trueno-ublk is a pure Rust userspace block device that provides ZRAM-like compressed RAM storage with SIMD/GPU acceleration via trueno-zram-core.

## Design Philosophy: First Principles

### What We Need
1. **Compressed page storage** - Store 4KB pages compressed in RAM
2. **Block device semantics** - Read/write at sector offsets
3. **zramctl compatibility** - CLI parity with Linux zram tools
4. **Performance** - SIMD-accelerated compression (AVX-512, AVX2, NEON)

### What We Don't Need (Initially)
- Complex kernel protocols (ublk, NBD)
- External C dependencies
- Privileged operations for testing

## Architecture

### Pure Rust Approach

```
┌─────────────────────────────────────────────────────────┐
│                    Application                           │
│              (file I/O, mmap, etc.)                      │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│              trueno-ublk (Pure Rust)                     │
│  ┌─────────────────────────────────────────────────────┐│
│  │                   BlockDevice                        ││
│  │  - read(offset, len) -> Vec<u8>                     ││
│  │  - write(offset, data)                              ││
│  │  - discard(offset, len)                             ││
│  │  - sync()                                           ││
│  └─────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────┐│
│  │                   PageStore                          ││
│  │  - HashMap<sector, CompressedPage>                  ││
│  │  - Entropy-based routing                            ││
│  │  - Zero-page deduplication                          ││
│  └─────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────┐│
│  │              trueno-zram-core                        ││
│  │  - LZ4/Zstd compression                             ││
│  │  - AVX-512/AVX2/SSE4.2/NEON backends                ││
│  │  - PageCompressor trait                             ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

### Kernel Integration (Phase 2)

For actual block device exposure, use pure Rust io_uring:

```
┌─────────────────────────────────────────────────────────┐
│                   Linux Kernel                           │
│              /dev/ublkb0 (block device)                  │
└───────────────────────┬─────────────────────────────────┘
                        │ io_uring (pure Rust: io-uring crate)
┌───────────────────────▼─────────────────────────────────┐
│              trueno-ublk daemon                          │
│  ┌─────────────────────────────────────────────────────┐│
│  │              UblkTarget (pure Rust)                  ││
│  │  - io-uring crate for async I/O                     ││
│  │  - Direct ublk protocol implementation              ││
│  │  - No libublk dependency                            ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Core Library (Pure Rust, No Kernel) ✅ COMPLETE

**Goal:** Prove compression works correctly with full test coverage.

```rust
// Core abstraction - no kernel required
pub struct BlockDevice {
    store: PageStore,
    size: u64,
    block_size: u32,
}

impl BlockDevice {
    pub fn new(size: u64, compressor: Box<dyn PageCompressor>) -> Self;
    pub fn read(&self, offset: u64, buf: &mut [u8]) -> Result<()>;
    pub fn write(&mut self, offset: u64, data: &[u8]) -> Result<()>;
    pub fn discard(&mut self, offset: u64, len: u64) -> Result<()>;
    pub fn stats(&self) -> DeviceStats;
}
```

**Files:**
- `src/device.rs` - BlockDevice abstraction ✅
- `src/daemon.rs` - PageStore with compression ✅
- `src/stats.rs` - Statistics tracking ✅

**Tests:**
- [ ] Write/read roundtrip with various data patterns
- [ ] Zero-page deduplication
- [ ] Entropy-based routing verification
- [ ] Compression ratio validation
- [ ] Concurrent access safety

### Phase 2: CLI Tools ✅ COMPLETE

**Goal:** zramctl-compatible command-line interface.

**Commands:**
- `trueno-ublk create` ✅
- `trueno-ublk list` ✅
- `trueno-ublk stat` ✅
- `trueno-ublk reset` ✅
- `trueno-ublk find` ✅
- `trueno-ublk compact` ✅
- `trueno-ublk idle` ✅
- `trueno-ublk writeback` ✅
- `trueno-ublk set` ✅
- `trueno-ublk top` ✅
- `trueno-ublk entropy` ✅

### Phase 3: Kernel Integration (Pure Rust io_uring)

**Goal:** Expose as real /dev/ublkbN block device using pure Rust.

**Approach:**
1. Use `io-uring` crate (pure Rust) for async I/O
2. Implement ublk protocol directly (no libublk)
3. Handle /dev/ublk-control ioctls via `nix` crate

```rust
// Pure Rust ublk implementation
use io_uring::IoUring;
use nix::sys::ioctl;

pub struct UblkDevice {
    ring: IoUring,
    ctrl_fd: RawFd,
    store: Arc<RwLock<PageStore>>,
}

impl UblkDevice {
    pub fn create(config: DeviceConfig) -> Result<Self>;
    pub fn run(&self) -> Result<()>;  // Main I/O loop
}
```

**Dependencies (all pure Rust):**
- `io-uring` - io_uring bindings
- `nix` - Unix syscalls and ioctls

### Phase 4: FUSE Alternative (Optional)

For systems without ublk support, provide FUSE-based block file:

```rust
use fuser::Filesystem;  // Pure Rust FUSE

pub struct TruenoFuse {
    store: Arc<RwLock<PageStore>>,
}

impl Filesystem for TruenoFuse {
    fn read(...);
    fn write(...);
}
```

## Testing Strategy

### Unit Tests (No Kernel)

```rust
#[test]
fn test_write_read_roundtrip() {
    let compressor = CompressorBuilder::new()
        .algorithm(Algorithm::Lz4)
        .build()
        .unwrap();

    let mut device = BlockDevice::new(1 << 30, compressor); // 1GB

    // Write test pattern
    let data = vec![0xAB; 4096];
    device.write(0, &data).unwrap();

    // Read back
    let mut buf = vec![0u8; 4096];
    device.read(0, &mut buf).unwrap();

    assert_eq!(data, buf);
}

#[test]
fn test_zero_page_dedup() {
    let mut device = BlockDevice::new(1 << 30, compressor);

    // Write zeros to multiple locations
    let zeros = vec![0u8; 4096];
    device.write(0, &zeros).unwrap();
    device.write(4096, &zeros).unwrap();
    device.write(8192, &zeros).unwrap();

    let stats = device.stats();
    assert_eq!(stats.same_pages, 3);
    // Memory used should be minimal (deduplicated)
}

#[test]
fn test_entropy_routing() {
    let mut device = BlockDevice::new(1 << 30, compressor);

    // High entropy data (random)
    let random: Vec<u8> = (0..4096).map(|_| rand::random()).collect();
    device.write(0, &random).unwrap();

    // Low entropy data (repetitive)
    let repetitive = vec![0xAA; 4096];
    device.write(4096, &repetitive).unwrap();

    let stats = device.stats();
    assert!(stats.scalar_pages > 0);  // High entropy -> scalar
    assert!(stats.simd_pages > 0);    // Low entropy -> SIMD
}
```

### Integration Tests (With Kernel)

```rust
#[test]
#[ignore]  // Requires root and ublk kernel module
fn test_real_block_device() {
    let device = UblkDevice::create(DeviceConfig {
        size: 1 << 30,
        algorithm: Algorithm::Lz4,
        ..Default::default()
    }).unwrap();

    // Device should appear as /dev/ublkbN
    assert!(Path::new(&device.path()).exists());

    // Write via standard file I/O
    let mut file = File::options()
        .read(true)
        .write(true)
        .open(device.path())
        .unwrap();

    file.write_all(&[0xAB; 4096]).unwrap();
    file.seek(SeekFrom::Start(0)).unwrap();

    let mut buf = vec![0u8; 4096];
    file.read_exact(&mut buf).unwrap();

    assert_eq!(buf, vec![0xAB; 4096]);
}
```

### Benchmark Tests

```rust
#[bench]
fn bench_sequential_write_4k() {
    let mut device = BlockDevice::new(1 << 30, compressor);
    let data = vec![0xAB; 4096];

    b.iter(|| {
        for i in 0..1000 {
            device.write(i * 4096, &data).unwrap();
        }
    });
}

#[bench]
fn bench_random_read_4k() {
    // Pre-populate device
    let device = create_populated_device();
    let mut buf = vec![0u8; 4096];

    b.iter(|| {
        let offset = (rand::random::<u64>() % 1000) * 4096;
        device.read(offset, &mut buf).unwrap();
    });
}
```

## Dependencies

### Required (Pure Rust)
- `trueno-zram-core` - SIMD compression (workspace)
- `anyhow` - Error handling
- `clap` - CLI parsing
- `serde` / `serde_json` - Serialization
- `ratatui` / `crossterm` - TUI

### Optional (For Kernel Integration)
- `io-uring` - Pure Rust io_uring bindings
- `nix` - Unix syscalls (already included)
- `fuser` - Pure Rust FUSE (alternative to ublk)

### Removed
- ~~`libublk`~~ - Replaced with pure Rust io_uring implementation

## Success Criteria

1. ✅ All CLI commands match zramctl interface
2. ✅ Entropy analysis correctly identifies compressible data
3. [ ] Write/read roundtrip preserves data integrity
4. [ ] Zero-page deduplication reduces memory usage
5. [ ] Compression ratio >= 2:1 for typical workloads
6. [ ] Throughput >= 2 GB/s with SIMD
7. [ ] No unsafe code outside of well-audited boundaries
8. [ ] 100% test coverage for core compression logic

## File Structure

```
bins/trueno-ublk/
├── Cargo.toml
├── src/
│   ├── main.rs           # Entry point
│   ├── cli/              # CLI commands (complete)
│   ├── device.rs         # BlockDevice abstraction
│   ├── daemon.rs         # PageStore + compression
│   ├── stats.rs          # Statistics tracking
│   ├── tui/              # TUI dashboard (complete)
│   └── ublk/             # Pure Rust ublk (Phase 3)
│       ├── mod.rs
│       ├── protocol.rs   # ublk protocol implementation
│       └── target.rs     # I/O handling
└── tests/
    ├── integration.rs    # Full roundtrip tests
    └── benchmark.rs      # Performance tests
```

## References

- [ublk kernel documentation](https://docs.kernel.org/block/ublk.html)
- [io-uring crate](https://crates.io/crates/io-uring)
- [nix crate](https://crates.io/crates/nix)
- [fuser crate](https://crates.io/crates/fuser)
