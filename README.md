# trueno-zram

<div align="center">

```
  ████████╗██████╗ ██╗   ██╗███████╗███╗   ██╗ ██████╗       ███████╗██████╗  █████╗ ███╗   ███╗
  ╚══██╔══╝██╔══██╗██║   ██║██╔════╝████╗  ██║██╔═══██╗      ╚══███╔╝██╔══██╗██╔══██╗████╗ ████║
     ██║   ██████╔╝██║   ██║█████╗  ██╔██╗ ██║██║   ██║  █████╗ ███╔╝ ██████╔╝███████║██╔████╔██║
     ██║   ██╔══██╗██║   ██║██╔══╝  ██║╚██╗██║██║   ██║  ╚════╝███╔╝  ██╔══██╗██╔══██║██║╚██╔╝██║
     ██║   ██║  ██║╚██████╔╝███████╗██║ ╚████║╚██████╔╝       ███████╗██║  ██║██║  ██║██║ ╚═╝ ██║
     ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝        ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝
```

**Pure Rust ublk Block Device & SIMD Compression Library**

*Educational reference implementation - Use kernel zram for production*

[![Crates.io](https://img.shields.io/crates/v/trueno-zram-core.svg)](https://crates.io/crates/trueno-zram-core)
[![Documentation](https://docs.rs/trueno-zram-core/badge.svg)](https://docs.rs/trueno-zram-core)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)]()
[![MSRV](https://img.shields.io/badge/MSRV-1.82.0-blue)]()

</div>

---

## Honest Performance Assessment

**This project is slower than alternatives for swap workloads.**

### vs Kernel ZRAM

| Metric | trueno-zram | Kernel ZRAM | Verdict |
|--------|-------------|-------------|---------|
| Random IOPS | 286K | ~1.5-2M | **0.15-0.2x (slower)** |
| Sequential Write | 651 MB/s | ~500-800 MB/s | ~Same |
| Sequential Read | 2.1 GB/s | ~2-3 GB/s | ~Same |

### vs NVMe Swap

| Metric | trueno-zram | NVMe Swap | Verdict |
|--------|-------------|-----------|---------|
| Random IOPS | 286K | 1.1M | **0.26x (slower)** |
| Sequential Write | 651 MB/s | 883 MB/s | Slower |
| Sequential Read | 2.1 GB/s | 3.4 GB/s | Slower |

### Why It's Slower

1. **Userspace overhead**: Every I/O requires kernel-to-userspace context switch
2. **ublk architectural limit**: ~1.2M IOPS theoretical max
3. **No kernel optimizations**: Kernel zram uses GFP_NOIO, direct memory access

## When to Use This

| Use Case | Recommendation |
|----------|----------------|
| Production swap | **Use kernel zram** |
| High-performance swap | **Use kernel zram or NVMe** |
| Learning ublk/io_uring | trueno-zram |
| Learning Rust systems programming | trueno-zram |
| Algorithm experimentation | trueno-zram |
| HDD-based systems | trueno-zram (faster than HDD) |

## What This Project Provides

### 1. `trueno-zram-core` - Compression Library

SIMD-accelerated page compression with clean Rust API:

```rust
use trueno_zram_core::{CompressorBuilder, Algorithm, PAGE_SIZE};

let compressor = CompressorBuilder::new()
    .algorithm(Algorithm::Lz4)
    .build()?;

let page = [0u8; PAGE_SIZE];
let compressed = compressor.compress(&page)?;
let decompressed = compressor.decompress(&compressed)?;
```

**Compression Throughput (microbenchmark):**

| Algorithm | Compress | Decompress |
|-----------|----------|------------|
| ZSTD L1 | 11.7 GiB/s | 9.5 GiB/s |
| ZSTD L3 | 10.3 GiB/s | 9.5 GiB/s |
| LZ4 | 5.8 GiB/s | 1.0 GiB/s |

### 2. `trueno-ublk` - Userspace Block Device

Pure Rust implementation of Linux ublk for compressed RAM storage:

```bash
# Create 8GB compressed RAM device
sudo trueno-ublk create --size 8G --algorithm lz4 --high-perf

# Use as swap
sudo mkswap /dev/ublkb0
sudo swapon -p 100 /dev/ublkb0
```

**Features:**
- io_uring for async I/O
- mlock to prevent swap deadlock
- LZ4/ZSTD compression
- Same-fill detection for zero pages
- Optional GPU acceleration (experimental)

## Installation

```bash
# Core compression library
cargo add trueno-zram-core

# Build the ublk daemon
cargo build --release -p trueno-ublk
```

## Project Structure

| Crate | Description |
|-------|-------------|
| `trueno-zram-core` | SIMD compression library (LZ4, ZSTD) |
| `trueno-ublk` | Userspace block device daemon |
| `trueno-zram-adaptive` | Entropy-based algorithm selection |

## Requirements

- Linux Kernel >= 6.0 (ublk support)
- x86_64 with AVX2/AVX-512 or AArch64 with NEON
- Rust >= 1.82.0

## Educational Value

This project demonstrates:

1. **Linux ublk subsystem** - Userspace block devices via io_uring
2. **io_uring** - Modern async I/O with SQ/CQ rings
3. **SIMD compression** - AVX2/AVX-512/NEON vectorization
4. **Memory-safe systems programming** - Rust for kernel-adjacent code
5. **Swap deadlock prevention** - mlock for daemon memory

## Benchmarking

```bash
# Compression microbenchmark
cargo bench -p trueno-zram-core

# Device I/O benchmark (requires root)
sudo fio --name=test --filename=/dev/ublkb0 --rw=randread \
    --bs=4k --numjobs=8 --iodepth=32 --runtime=10 --group_reporting
```

## Contributing

This is an educational project. Contributions welcome for:

- Documentation improvements
- Bug fixes
- New compression algorithms
- Performance optimizations (though kernel zram will still be faster)

## License

MIT OR Apache-2.0

## Related Projects

- [Linux ublk](https://docs.kernel.org/block/ublk.html) - Kernel ublk documentation
- [zram](https://www.kernel.org/doc/html/latest/admin-guide/blockdev/zram.html) - Kernel zram (recommended for production)
- [io_uring](https://kernel.dk/io_uring.pdf) - io_uring design document

---

<div align="center">

*Part of the PAIML ecosystem: [trueno](https://crates.io/crates/trueno) | [aprender](https://crates.io/crates/aprender)*

**For production swap, use kernel zram.**

</div>
