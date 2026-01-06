//! trueno-ublk - GPU-accelerated ZRAM replacement using ublk
//!
//! This crate provides a ublk-based block device with SIMD and GPU-accelerated
//! compression. It can be used as a drop-in replacement for Linux kernel ZRAM.
//!
//! # Features
//!
//! - **SIMD acceleration**: AVX2, AVX-512, and NEON support via trueno-zram-core
//! - **GPU offload**: CUDA-based batch compression for large workloads
//! - **Entropy routing**: Automatic algorithm selection based on data entropy
//! - **Zero-page deduplication**: Efficient handling of all-zero pages
//! - **zram-compatible stats**: Statistics format compatible with kernel zram
//!
//! # Example
//!
//! ```no_run
//! use trueno_ublk::BlockDevice;
//! use trueno_zram_core::{Algorithm, CompressorBuilder, PAGE_SIZE};
//!
//! // Create a compressor
//! let compressor = CompressorBuilder::new()
//!     .algorithm(Algorithm::Lz4)
//!     .build()
//!     .unwrap();
//!
//! // Create a 1GB block device
//! let mut device = BlockDevice::new(1 << 30, compressor);
//!
//! // Write data
//! let data = vec![0xAB; PAGE_SIZE];
//! device.write(0, &data).unwrap();
//!
//! // Read back
//! let mut buf = vec![0u8; PAGE_SIZE];
//! device.read(0, &mut buf).unwrap();
//! assert_eq!(data, buf);
//!
//! // Check stats
//! let stats = device.stats();
//! println!("Compression ratio: {:.2}x", stats.compression_ratio());
//! ```

#![allow(dead_code, unused_imports)] // Daemon has helper functions and re-exports for future use

pub mod cleanup;
pub mod daemon;
pub mod device;
pub mod perf;
pub mod stats;
pub mod ublk;

// Re-export commonly used types
pub use device::{BlockDevice, BlockDeviceStats, DeviceConfig, DeviceStats, UblkDevice};
pub use ublk::{DaemonError, UblkCtrl, UblkDaemon};

// Re-export duende-mlock for swap deadlock prevention (DT-007)
pub use duende_mlock::{
    is_locked as is_memory_locked, lock_all as lock_daemon_memory, MlockStatus,
};
#[cfg(not(test))]
pub use ublk::{run_daemon, run_daemon_batched, BatchedDaemonConfig};

// Re-export batched page store for direct use
pub use daemon::{spawn_flush_thread, BatchConfig, BatchedPageStore, BatchedPageStoreStats};

// Re-export performance optimization module (PERF-001)
// Note: Additional types (BatchCoalescer, NumaAllocator, etc.) are available
// in perf submodules for future PERF-001 integration
pub use perf::{HiPerfContext, PerfConfig, PollResult, PollingConfig};
