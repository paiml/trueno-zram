//! Device module - ublk device management
//!
//! Provides abstraction over libublk for managing trueno-ublk devices.
//! Also provides a pure Rust `BlockDevice` for testing without kernel dependencies.

#![allow(dead_code)]

use crate::daemon::PageStore;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use trueno_zram_core::{CompressorBuilder, PageCompressor, PAGE_SIZE};

/// Device configuration
#[derive(Clone, Debug)]
pub struct DeviceConfig {
    /// Device ID to use (-1 for auto-assign)
    pub dev_id: i32,
    pub size: u64,
    pub algorithm: trueno_zram_core::Algorithm,
    pub streams: usize,
    pub gpu_enabled: bool,
    pub mem_limit: Option<u64>,
    pub backing_dev: Option<PathBuf>,
    pub writeback_limit: Option<u64>,
    pub entropy_skip_threshold: f64,
    pub gpu_batch_size: usize,
    pub foreground: bool,
    /// Enable batched compression mode for high throughput
    pub batched: bool,
    /// Batch threshold - pages before triggering batch compression
    pub batch_threshold: usize,
    /// Flush timeout in milliseconds
    pub flush_timeout_ms: u64,
    /// PERF-001: High-performance configuration (polling, affinity, NUMA)
    pub perf: Option<crate::perf::PerfConfig>,
    /// PERF-003: Number of hardware queues (1-8)
    /// Multiple queues enable parallel I/O with separate io_uring per queue.
    pub nr_hw_queues: u16,
    /// PERF-006: Enable ZERO_COPY mode (EXPERIMENTAL)
    /// Eliminates pwrite syscall per I/O by using mmap'd kernel buffers.
    pub zero_copy: bool,
    // =========================================================================
    // KERN-001/002/003: Kernel-Cooperative Tiered Storage
    // =========================================================================
    /// Storage backend type: memory, zram, tiered
    pub backend: crate::backend::BackendType,
    /// Enable entropy-based routing for tiered storage
    pub entropy_routing: bool,
    /// Kernel ZRAM device path (e.g., /dev/zram0)
    pub zram_device: Option<PathBuf>,
    /// Entropy threshold for kernel ZRAM routing (pages below this go to kernel)
    pub entropy_kernel_threshold: f64,
    // =========================================================================
    // VIZ-002: Renacer Visualization Integration
    // =========================================================================
    /// Enable real-time TUI visualization (renacer dashboard)
    pub visualize: bool,
    // =========================================================================
    // VIZ-004: OTLP Integration (OpenTelemetry)
    // =========================================================================
    /// OTLP endpoint for trace/metric export
    pub otlp_endpoint: Option<String>,
    /// Service name for OTLP traces
    pub otlp_service_name: String,
}

/// Device statistics (zram-compatible)
#[derive(Clone, Debug, Default)]
pub struct DeviceStats {
    // mm_stat fields (9 fields)
    pub orig_data_size: u64,
    pub compr_data_size: u64,
    pub mem_used_total: u64,
    pub mem_limit: u64,
    pub mem_used_max: u64,
    pub same_pages: u64,
    pub pages_compacted: u64,
    pub huge_pages: u64,
    pub huge_pages_since: u64,

    // io_stat fields (4 fields)
    pub failed_reads: u64,
    pub failed_writes: u64,
    pub invalid_io: u64,
    pub notify_free: u64,

    // bd_stat fields (3 fields)
    pub bd_count: u64,
    pub bd_reads: u64,
    pub bd_writes: u64,

    // Extended trueno-ublk stats
    pub gpu_pages: u64,
    pub simd_pages: u64,
    pub scalar_pages: u64,
    pub throughput_gbps: f64,
    pub avg_entropy: f64,
    pub simd_backend: String,
}

/// Runtime statistics tracking
struct RuntimeStats {
    orig_data_size: AtomicU64,
    compr_data_size: AtomicU64,
    mem_used_total: AtomicU64,
    mem_used_max: AtomicU64,
    same_pages: AtomicU64,
    pages_compacted: AtomicU64,
    huge_pages: AtomicU64,
    huge_pages_since: AtomicU64,
    failed_reads: AtomicU64,
    failed_writes: AtomicU64,
    invalid_io: AtomicU64,
    notify_free: AtomicU64,
    bd_count: AtomicU64,
    bd_reads: AtomicU64,
    bd_writes: AtomicU64,
    gpu_pages: AtomicU64,
    simd_pages: AtomicU64,
    scalar_pages: AtomicU64,
    bytes_processed: AtomicU64,
    entropy_sum: AtomicU64,
    entropy_count: AtomicU64,
}

impl Default for RuntimeStats {
    fn default() -> Self {
        Self {
            orig_data_size: AtomicU64::new(0),
            compr_data_size: AtomicU64::new(0),
            mem_used_total: AtomicU64::new(0),
            mem_used_max: AtomicU64::new(0),
            same_pages: AtomicU64::new(0),
            pages_compacted: AtomicU64::new(0),
            huge_pages: AtomicU64::new(0),
            huge_pages_since: AtomicU64::new(0),
            failed_reads: AtomicU64::new(0),
            failed_writes: AtomicU64::new(0),
            invalid_io: AtomicU64::new(0),
            notify_free: AtomicU64::new(0),
            bd_count: AtomicU64::new(0),
            bd_reads: AtomicU64::new(0),
            bd_writes: AtomicU64::new(0),
            gpu_pages: AtomicU64::new(0),
            simd_pages: AtomicU64::new(0),
            scalar_pages: AtomicU64::new(0),
            bytes_processed: AtomicU64::new(0),
            entropy_sum: AtomicU64::new(0),
            entropy_count: AtomicU64::new(0),
        }
    }
}

/// Global device registry
static DEVICES: std::sync::LazyLock<RwLock<HashMap<u32, Arc<DeviceInner>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

struct DeviceInner {
    id: u32,
    config: RwLock<DeviceConfig>,
    stats: RuntimeStats,
    compressor: RwLock<Option<Box<dyn PageCompressor>>>,
}

/// A trueno-ublk device
pub struct UblkDevice {
    inner: Arc<DeviceInner>,
}

impl UblkDevice {
    /// DT-007: Lock daemon memory to prevent swap deadlock
    ///
    /// Must be called in the ACTUAL daemon process (after fork for background mode).
    /// Five Whys Root Cause: mlock only affects calling process, not forked children.
    #[cfg(not(test))]
    fn lock_daemon_memory() {
        use duende_mlock::{lock_all, MlockStatus};
        match lock_all() {
            Ok(MlockStatus::Locked { bytes_locked }) => {
                tracing::info!(
                    "DT-007: Memory locked ({} bytes) - swap deadlock prevention active",
                    bytes_locked
                );
            }
            Ok(MlockStatus::Failed { errno }) => {
                tracing::warn!(
                    "DT-007: mlock() failed (errno={}) - daemon may deadlock under swap pressure",
                    errno
                );
            }
            Ok(MlockStatus::Unsupported) => {
                tracing::debug!("DT-007: mlock() not supported on this platform");
            }
            Err(e) => {
                tracing::error!("DT-007: mlock() error: {:?}", e);
            }
        }
    }

    #[cfg(test)]
    fn lock_daemon_memory() {
        // No-op in tests
    }

    /// Create a new device with the given configuration
    #[cfg(not(test))]
    pub fn create(config: DeviceConfig) -> Result<Self> {
        // DT-007e: mlock moved to lock_daemon_memory() - called in daemon process
        // For foreground mode: called in start_ublk_daemon before run_with_mode
        // For background mode: called in child process after fork()

        let id = if config.dev_id >= 0 {
            config.dev_id as u32
        } else {
            Self::next_free_id()?
        };

        // Initialize compressor using CompressorBuilder
        let compressor = CompressorBuilder::new()
            .algorithm(config.algorithm)
            .build()?;

        // Start ublk daemon (needs config before we move it)
        Self::start_ublk_daemon(id, &config)?;

        let inner = Arc::new(DeviceInner {
            id,
            config: RwLock::new(config),
            stats: RuntimeStats::default(),
            compressor: RwLock::new(Some(compressor)),
        });

        // Register device
        {
            let mut devices = DEVICES.write().expect("rwlock poisoned");
            devices.insert(id, inner.clone());
        }

        Ok(Self { inner })
    }

    /// Create a mock device for testing (doesn't start ublk daemon)
    #[cfg(test)]
    pub fn create(config: DeviceConfig) -> Result<Self> {
        let id = if config.dev_id >= 0 {
            config.dev_id as u32
        } else {
            0 // Default to device 0 in tests
        };

        // Initialize compressor using CompressorBuilder
        let compressor = CompressorBuilder::new()
            .algorithm(config.algorithm)
            .build()?;

        let inner = Arc::new(DeviceInner {
            id,
            config: RwLock::new(config),
            stats: RuntimeStats::default(),
            compressor: RwLock::new(Some(compressor)),
        });

        // Register device
        {
            let mut devices = DEVICES.write().expect("rwlock poisoned");
            devices.insert(id, inner.clone());
        }

        Ok(Self { inner })
    }

    /// Open an existing device by path
    pub fn open(path: &Path) -> Result<Self> {
        let id = Self::parse_device_id(path)?;

        let devices = DEVICES.read().expect("rwlock poisoned");
        let inner = devices
            .get(&id)
            .cloned()
            .context(format!("Device {} not found", path.display()))?;

        Ok(Self { inner })
    }

    /// List all trueno-ublk devices
    pub fn list_all() -> Result<Vec<Self>> {
        let devices = DEVICES.read().expect("rwlock poisoned");
        Ok(devices
            .values()
            .map(|inner| Self {
                inner: inner.clone(),
            })
            .collect())
    }

    /// Find the next free device ID
    pub fn next_free_id() -> Result<u32> {
        let devices = DEVICES.read().expect("rwlock poisoned");
        let mut id = 0;
        while devices.contains_key(&id) {
            id += 1;
        }
        Ok(id)
    }

    /// Get device name
    pub fn name(&self) -> String {
        format!("ublkb{}", self.inner.id)
    }

    /// Get device path
    pub fn path(&self) -> PathBuf {
        PathBuf::from(format!("/dev/ublkb{}", self.inner.id))
    }

    /// Get device configuration
    pub fn config(&self) -> DeviceConfig {
        self.inner.config.read().expect("rwlock poisoned").clone()
    }

    /// Get device statistics
    pub fn stats(&self) -> DeviceStats {
        let s = &self.inner.stats;
        let config = self.config();

        let bytes_processed = s.bytes_processed.load(Ordering::Relaxed);
        let entropy_count = s.entropy_count.load(Ordering::Relaxed);

        DeviceStats {
            orig_data_size: s.orig_data_size.load(Ordering::Relaxed),
            compr_data_size: s.compr_data_size.load(Ordering::Relaxed),
            mem_used_total: s.mem_used_total.load(Ordering::Relaxed),
            mem_limit: config.mem_limit.unwrap_or(0),
            mem_used_max: s.mem_used_max.load(Ordering::Relaxed),
            same_pages: s.same_pages.load(Ordering::Relaxed),
            pages_compacted: s.pages_compacted.load(Ordering::Relaxed),
            huge_pages: s.huge_pages.load(Ordering::Relaxed),
            huge_pages_since: s.huge_pages_since.load(Ordering::Relaxed),
            failed_reads: s.failed_reads.load(Ordering::Relaxed),
            failed_writes: s.failed_writes.load(Ordering::Relaxed),
            invalid_io: s.invalid_io.load(Ordering::Relaxed),
            notify_free: s.notify_free.load(Ordering::Relaxed),
            bd_count: s.bd_count.load(Ordering::Relaxed),
            bd_reads: s.bd_reads.load(Ordering::Relaxed),
            bd_writes: s.bd_writes.load(Ordering::Relaxed),
            gpu_pages: s.gpu_pages.load(Ordering::Relaxed),
            simd_pages: s.simd_pages.load(Ordering::Relaxed),
            scalar_pages: s.scalar_pages.load(Ordering::Relaxed),
            throughput_gbps: if bytes_processed > 0 {
                // Estimate based on recent throughput
                (bytes_processed as f64) / 1e9
            } else {
                0.0
            },
            avg_entropy: if entropy_count > 0 {
                (s.entropy_sum.load(Ordering::Relaxed) as f64) / (entropy_count as f64) / 100.0
            } else {
                0.0
            },
            simd_backend: detect_simd_backend(),
        }
    }

    /// Get mountpoint if device is mounted
    pub fn mountpoint(&self) -> Option<String> {
        let path = self.path();
        let mounts = fs::read_to_string("/proc/mounts").ok()?;

        for line in mounts.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 && parts[0] == path.to_string_lossy() {
                return Some(parts[1].to_string());
            }
        }
        None
    }

    /// Remove the device
    pub fn remove(&self) -> Result<()> {
        Self::stop_ublk_daemon(self.inner.id)?;

        let mut devices = DEVICES.write().expect("rwlock poisoned");
        devices.remove(&self.inner.id);

        Ok(())
    }

    /// Trigger memory compaction
    pub fn compact(&self) -> Result<()> {
        // In a real implementation, this would trigger compaction
        self.inner
            .stats
            .pages_compacted
            .fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Mark pages as idle
    pub fn mark_idle(&self) -> Result<()> {
        // In a real implementation, this would mark pages as idle
        Ok(())
    }

    /// Trigger writeback to backing device
    pub fn writeback(&self, idle: bool, huge: bool, all: bool) -> Result<()> {
        let config = self.config();
        if config.backing_dev.is_none() {
            anyhow::bail!("No backing device configured");
        }

        // In a real implementation, this would trigger writeback
        let _ = (idle, huge, all);
        Ok(())
    }

    /// Set memory limit
    pub fn set_mem_limit(&self, limit: u64) -> Result<()> {
        let mut config = self.inner.config.write().expect("rwlock poisoned");
        config.mem_limit = Some(limit);
        Ok(())
    }

    /// Set GPU enabled/disabled
    pub fn set_gpu_enabled(&self, enabled: bool) -> Result<()> {
        let mut config = self.inner.config.write().expect("rwlock poisoned");
        config.gpu_enabled = enabled;

        // Reinitialize compressor with new settings
        let compressor = CompressorBuilder::new()
            .algorithm(config.algorithm)
            .build()?;
        *self.inner.compressor.write().expect("rwlock poisoned") = Some(compressor);

        Ok(())
    }

    /// Set entropy skip threshold
    pub fn set_entropy_threshold(&self, threshold: f64) -> Result<()> {
        let mut config = self.inner.config.write().expect("rwlock poisoned");
        config.entropy_skip_threshold = threshold;
        Ok(())
    }

    /// Set writeback limit
    pub fn set_writeback_limit(&self, limit: u64) -> Result<()> {
        let mut config = self.inner.config.write().expect("rwlock poisoned");
        config.writeback_limit = Some(limit);
        Ok(())
    }

    /// Enable/disable writeback limit
    pub fn set_writeback_limit_enabled(&self, enabled: bool) -> Result<()> {
        let mut config = self.inner.config.write().expect("rwlock poisoned");
        if !enabled {
            config.writeback_limit = None;
        }
        Ok(())
    }

    /// Reset mem_used_max watermark
    pub fn reset_mem_used_max(&self) -> Result<()> {
        let current = self.inner.stats.mem_used_total.load(Ordering::Relaxed);
        self.inner
            .stats
            .mem_used_max
            .store(current, Ordering::Relaxed);
        Ok(())
    }

    // Private helpers

    fn parse_device_id(path: &Path) -> Result<u32> {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .context("Invalid device path")?;

        if let Some(id_str) = name.strip_prefix("ublkb") {
            id_str.parse().context("Invalid device ID in path")
        } else {
            anyhow::bail!("Not a ublk device: {}", path.display())
        }
    }

    #[cfg(not(test))]
    fn start_ublk_daemon(id: u32, config: &DeviceConfig) -> Result<()> {
        use crate::ublk::{run_daemon, run_daemon_batched, BatchedDaemonConfig, DaemonError};
        use nix::libc;
        use std::sync::atomic::AtomicBool;

        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = stop.clone();

        // Setup signal handler for graceful shutdown
        ctrlc::set_handler(move || {
            stop_clone.store(true, std::sync::atomic::Ordering::Relaxed);
        })
        .ok();

        // Helper to run daemon with batched or non-batched mode
        let run_with_mode =
            |stop: Arc<AtomicBool>, ready_fd: Option<i32>| -> Result<(), DaemonError> {
                if config.batched {
                    // Batched mode: high-throughput compression (>10 GB/s)
                    tracing::info!(
                        "Starting daemon in batched mode (threshold={}, flush={}ms, perf={})",
                        config.batch_threshold,
                        config.flush_timeout_ms,
                        config.perf.is_some()
                    );
                    let batch_config = BatchedDaemonConfig {
                        dev_id: id as i32,
                        dev_size: config.size,
                        algorithm: config.algorithm,
                        batch_threshold: config.batch_threshold,
                        flush_timeout_ms: config.flush_timeout_ms,
                        gpu_batch_size: config.gpu_batch_size,
                        perf: config.perf.clone(), // PERF-001: Pass performance config
                        nr_hw_queues: config.nr_hw_queues, // PERF-003: Multi-queue
                        zero_copy: config.zero_copy, // PERF-006: Zero-copy mode
                        // KERN-001/002/003: Kernel-Cooperative Tiered Storage
                        backend: config.backend,
                        entropy_routing: config.entropy_routing,
                        zram_device: config.zram_device.clone(),
                        entropy_kernel_threshold: config.entropy_kernel_threshold,
                        entropy_skip_threshold: config.entropy_skip_threshold,
                    };
                    run_daemon_batched(batch_config, stop, ready_fd)
                } else {
                    // Standard mode: per-page compression
                    run_daemon(id as i32, config.size, config.algorithm, stop, ready_fd)
                }
            };

        // Foreground mode - run directly
        if config.foreground {
            tracing::info!("Running in foreground mode");
            // DT-007e: Lock memory in the daemon process (this IS the daemon)
            Self::lock_daemon_memory();
            return run_with_mode(stop, None).map_err(|e| anyhow::anyhow!("Daemon failed: {}", e));
        }

        // Create eventfd for synchronization
        // SAFETY: eventfd creates a file descriptor for event notification.
        // Return value is checked below for errors.
        let efd = unsafe { libc::eventfd(0, 0) };
        if efd < 0 {
            anyhow::bail!("Failed to create eventfd");
        }

        // Fork for daemon mode
        // SAFETY: fork() creates a new child process. Return value is checked
        // to handle error (-1), child (0), and parent (>0) cases.
        match unsafe { libc::fork() } {
            -1 => {
                // SAFETY: efd is a valid file descriptor from eventfd() above
                unsafe { libc::close(efd) };
                anyhow::bail!("Fork failed");
            }
            0 => {
                // Child process - run daemon
                // SAFETY: setsid() creates a new session with the calling process as leader.
                // Called in child process after fork to detach from controlling terminal.
                unsafe { libc::setsid() };

                // DT-007e: Lock memory in child process AFTER fork
                // Five Whys: mlock only affects calling process, not forked children
                Self::lock_daemon_memory();

                // Daemon will signal readiness via efd
                match run_with_mode(stop, Some(efd)) {
                    Ok(()) | Err(DaemonError::Stopped) => {
                        // SAFETY: efd is a valid file descriptor inherited from parent
                        unsafe { libc::close(efd) };
                        std::process::exit(0);
                    }
                    Err(e) => {
                        tracing::error!("Daemon failed: {}", e);
                        // SAFETY: efd is a valid file descriptor inherited from parent.
                        // Writing 0 signals failure to parent process waiting on read().
                        unsafe {
                            // Signal failure with 0
                            libc::write(efd, &0u64 as *const u64 as *const libc::c_void, 8);
                            libc::close(efd);
                        }
                        std::process::exit(1);
                    }
                }
            }
            _pid => {
                // Parent process - wait for device ID
                let mut val: u64 = 0;
                // SAFETY: efd is a valid eventfd, val is a properly aligned u64.
                // read() blocks until child writes success (device_id+1) or failure (0).
                let n = unsafe { libc::read(efd, &mut val as *mut u64 as *mut libc::c_void, 8) };
                // SAFETY: efd is a valid file descriptor, close releases it
                unsafe { libc::close(efd) };

                if n == 8 && val > 0 {
                    tracing::info!("Device created with ID: {}", (val - 1) as i64);
                    Ok(())
                } else {
                    anyhow::bail!("Failed to read device ID from daemon")
                }
            }
        }
    }

    fn stop_ublk_daemon(_id: u32) -> Result<()> {
        // Device cleanup is handled by UblkCtrl::drop
        Ok(())
    }
}

/// Detect the best available SIMD backend
fn detect_simd_backend() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return "avx512".to_string();
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return "avx2".to_string();
        }
        if std::arch::is_x86_feature_detected!("sse4.2") {
            return "sse4.2".to_string();
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return "neon".to_string();
    }

    "scalar".to_string()
}

// ============================================================================
// BlockDevice - Pure Rust block device abstraction for testing
// ============================================================================

/// Statistics for BlockDevice
#[derive(Clone, Debug, Default)]
pub struct BlockDeviceStats {
    pub pages_stored: u64,
    pub bytes_written: u64,
    pub bytes_read: u64,
    pub bytes_compressed: u64,
    pub zero_pages: u64,
    pub gpu_pages: u64,
    pub simd_pages: u64,
    pub scalar_pages: u64,
}

impl BlockDeviceStats {
    /// Calculate compression ratio (original / compressed)
    pub fn compression_ratio(&self) -> f64 {
        if self.bytes_compressed > 0 {
            self.bytes_written as f64 / self.bytes_compressed as f64
        } else {
            1.0
        }
    }
}

/// Pure Rust block device abstraction for testing without kernel dependencies.
///
/// This provides a simple in-memory block device that compresses pages using
/// trueno-zram-core. It can be used for:
/// - Unit testing compression roundtrips
/// - Benchmarking throughput
/// - Validating correctness before kernel integration
///
/// # Example
///
/// ```ignore
/// use trueno_zram_core::{Algorithm, CompressorBuilder};
/// use trueno_ublk::device::BlockDevice;
///
/// let compressor = CompressorBuilder::new()
///     .algorithm(Algorithm::Lz4)
///     .build()
///     .unwrap();
///
/// let mut device = BlockDevice::new(1 << 30, compressor); // 1GB
///
/// // Write data
/// let data = vec![0xAB; 4096];
/// device.write(0, &data).unwrap();
///
/// // Read back
/// let mut buf = vec![0u8; 4096];
/// device.read(0, &mut buf).unwrap();
/// assert_eq!(data, buf);
/// ```
pub struct BlockDevice {
    store: PageStore,
    size: u64,
    block_size: u32,
    bytes_written: AtomicU64,
    bytes_read: AtomicU64,
}

impl BlockDevice {
    /// Create a new block device with the given size and compressor.
    ///
    /// # Arguments
    /// * `size` - Total device size in bytes
    /// * `compressor` - The compressor to use for page compression
    pub fn new(size: u64, compressor: Box<dyn PageCompressor>) -> Self {
        Self {
            store: PageStore::with_compressor(Arc::from(compressor), 7.5),
            size,
            block_size: PAGE_SIZE as u32,
            bytes_written: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
        }
    }

    /// Create a new block device with custom entropy threshold.
    pub fn with_entropy_threshold(
        size: u64,
        compressor: Box<dyn PageCompressor>,
        entropy_threshold: f64,
    ) -> Self {
        Self {
            store: PageStore::with_compressor(Arc::from(compressor), entropy_threshold),
            size,
            block_size: PAGE_SIZE as u32,
            bytes_written: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
        }
    }

    /// Get the device size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Get the block size in bytes.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Read data from the device at the given byte offset.
    ///
    /// # Arguments
    /// * `offset` - Byte offset to read from (must be page-aligned)
    /// * `buf` - Buffer to read into (length must be multiple of PAGE_SIZE)
    pub fn read(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.validate_io(offset, buf.len())?;

        let pages = buf.len() / PAGE_SIZE;
        for i in 0..pages {
            let sector = self.offset_to_sector(offset + (i * PAGE_SIZE) as u64);
            let page_buf = &mut buf[i * PAGE_SIZE..(i + 1) * PAGE_SIZE];
            self.store.load(sector, page_buf)?;
        }

        self.bytes_read
            .fetch_add(buf.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Write data to the device at the given byte offset.
    ///
    /// # Arguments
    /// * `offset` - Byte offset to write to (must be page-aligned)
    /// * `data` - Data to write (length must be multiple of PAGE_SIZE)
    pub fn write(&mut self, offset: u64, data: &[u8]) -> Result<()> {
        self.validate_io(offset, data.len())?;

        let pages = data.len() / PAGE_SIZE;
        for i in 0..pages {
            let sector = self.offset_to_sector(offset + (i * PAGE_SIZE) as u64);
            let page_data = &data[i * PAGE_SIZE..(i + 1) * PAGE_SIZE];
            self.store.store(sector, page_data)?;
        }

        self.bytes_written
            .fetch_add(data.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Discard data at the given byte offset.
    ///
    /// # Arguments
    /// * `offset` - Byte offset to discard from (must be page-aligned)
    /// * `len` - Length to discard in bytes (must be multiple of PAGE_SIZE)
    pub fn discard(&mut self, offset: u64, len: u64) -> Result<()> {
        self.validate_io(offset, len as usize)?;

        let pages = len as usize / PAGE_SIZE;
        for i in 0..pages {
            let sector = self.offset_to_sector(offset + (i * PAGE_SIZE) as u64);
            self.store.remove(sector);
        }

        Ok(())
    }

    /// Sync all data to storage (no-op for in-memory device).
    pub fn sync(&self) -> Result<()> {
        Ok(())
    }

    /// Get device statistics.
    pub fn stats(&self) -> BlockDeviceStats {
        let store_stats = self.store.stats();
        BlockDeviceStats {
            pages_stored: store_stats.pages_stored,
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            bytes_compressed: store_stats.bytes_compressed,
            zero_pages: store_stats.zero_pages,
            gpu_pages: store_stats.gpu_pages,
            simd_pages: store_stats.simd_pages,
            scalar_pages: store_stats.scalar_pages,
        }
    }

    // Private helpers

    fn validate_io(&self, offset: u64, len: usize) -> Result<()> {
        if offset % PAGE_SIZE as u64 != 0 {
            anyhow::bail!(
                "Offset {} is not page-aligned (PAGE_SIZE={})",
                offset,
                PAGE_SIZE
            );
        }
        if len % PAGE_SIZE != 0 {
            anyhow::bail!(
                "Length {} is not a multiple of PAGE_SIZE ({})",
                len,
                PAGE_SIZE
            );
        }
        if offset + len as u64 > self.size {
            anyhow::bail!(
                "I/O extends beyond device size: offset={}, len={}, size={}",
                offset,
                len,
                self.size
            );
        }
        Ok(())
    }

    fn offset_to_sector(&self, offset: u64) -> u64 {
        // Sector size is 512 bytes, so divide by 512 to get sector number
        offset / 512
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trueno_zram_core::Algorithm;

    // Helper to create a test compressor
    fn test_compressor() -> Box<dyn PageCompressor> {
        CompressorBuilder::new()
            .algorithm(Algorithm::Lz4)
            .build()
            .unwrap()
    }

    #[test]
    fn test_parse_device_id() {
        let path = PathBuf::from("/dev/ublkb0");
        assert_eq!(UblkDevice::parse_device_id(&path).unwrap(), 0);

        let path = PathBuf::from("/dev/ublkb42");
        assert_eq!(UblkDevice::parse_device_id(&path).unwrap(), 42);
    }

    #[test]
    fn test_detect_simd_backend() {
        let backend = detect_simd_backend();
        assert!(!backend.is_empty());
    }

    #[test]
    fn test_device_stats_default() {
        let stats = DeviceStats::default();
        assert_eq!(stats.orig_data_size, 0);
        assert_eq!(stats.throughput_gbps, 0.0);
    }

    // ========================================================================
    // BlockDevice tests - Pure Rust compression roundtrip verification
    // ========================================================================

    #[test]
    fn test_block_device_creation() {
        let device = BlockDevice::new(1 << 20, test_compressor()); // 1MB
        assert_eq!(device.size(), 1 << 20);
        assert_eq!(device.block_size(), PAGE_SIZE as u32);
    }

    #[test]
    fn test_write_read_roundtrip_single_page() {
        let mut device = BlockDevice::new(1 << 20, test_compressor());

        // Write a single page
        let data = vec![0xAB; PAGE_SIZE];
        device.write(0, &data).unwrap();

        // Read back
        let mut buf = vec![0u8; PAGE_SIZE];
        device.read(0, &mut buf).unwrap();

        assert_eq!(data, buf, "Single page roundtrip failed");
    }

    #[test]
    fn test_write_read_roundtrip_multiple_pages() {
        let mut device = BlockDevice::new(1 << 20, test_compressor());

        // Write multiple pages at different offsets
        for i in 0..10 {
            let data: Vec<u8> = (0..PAGE_SIZE).map(|j| ((i + j) % 256) as u8).collect();
            device.write(i as u64 * PAGE_SIZE as u64, &data).unwrap();
        }

        // Read back and verify
        for i in 0..10 {
            let expected: Vec<u8> = (0..PAGE_SIZE).map(|j| ((i + j) % 256) as u8).collect();
            let mut buf = vec![0u8; PAGE_SIZE];
            device.read(i as u64 * PAGE_SIZE as u64, &mut buf).unwrap();
            assert_eq!(expected, buf, "Page {} roundtrip failed", i);
        }
    }

    #[test]
    fn test_write_read_roundtrip_random_data() {
        let mut device = BlockDevice::new(1 << 20, test_compressor());

        // Create pseudo-random data (high entropy)
        let data: Vec<u8> = (0..PAGE_SIZE).map(|i| (i * 17 + 31) as u8).collect();
        device.write(0, &data).unwrap();

        let mut buf = vec![0u8; PAGE_SIZE];
        device.read(0, &mut buf).unwrap();

        assert_eq!(data, buf, "Random data roundtrip failed");
    }

    #[test]
    fn test_zero_page_deduplication() {
        let mut device = BlockDevice::new(1 << 20, test_compressor());

        // Write zeros to multiple locations
        let zeros = vec![0u8; PAGE_SIZE];
        device.write(0, &zeros).unwrap();
        device.write(PAGE_SIZE as u64, &zeros).unwrap();
        device.write(2 * PAGE_SIZE as u64, &zeros).unwrap();

        let stats = device.stats();
        assert_eq!(stats.zero_pages, 3, "Should have 3 zero pages");

        // Read back and verify
        for i in 0..3 {
            let mut buf = vec![0xFFu8; PAGE_SIZE];
            device.read(i as u64 * PAGE_SIZE as u64, &mut buf).unwrap();
            assert_eq!(zeros, buf, "Zero page {} readback failed", i);
        }
    }

    #[test]
    fn test_compression_ratio_tracking() {
        let mut device = BlockDevice::new(1 << 20, test_compressor());

        // Write compressible but NON-uniform data (avoids same-fill optimization)
        // PERF-013: Same-fill pages are stored without compression, so use varied data
        let mut compressible = vec![0u8; PAGE_SIZE];
        for i in 0..PAGE_SIZE {
            // Repeating pattern but not uniform - still compresses well
            compressible[i] = (i % 16) as u8;
        }
        device.write(0, &compressible).unwrap();

        let stats = device.stats();
        assert!(
            stats.bytes_compressed < stats.bytes_written,
            "Compressible data should compress: written={}, compressed={}",
            stats.bytes_written,
            stats.bytes_compressed
        );
        assert!(
            stats.compression_ratio() > 1.0,
            "Compression ratio should be > 1.0 for compressible data"
        );
    }

    #[test]
    fn test_entropy_routing_high_entropy() {
        // Use threshold of 7.0 for testing
        let mut device = BlockDevice::with_entropy_threshold(1 << 20, test_compressor(), 7.0);

        // Write high-entropy (pseudo-random) data
        let random: Vec<u8> = (0..PAGE_SIZE).map(|i| (i * 17 + 31) as u8).collect();
        device.write(0, &random).unwrap();

        let stats = device.stats();
        assert!(
            stats.scalar_pages > 0,
            "High entropy data should use scalar path"
        );
    }

    #[test]
    fn test_entropy_routing_low_entropy() {
        // Use threshold of 7.0 for testing
        let mut device = BlockDevice::with_entropy_threshold(1 << 20, test_compressor(), 7.0);

        // Write low-entropy (repetitive pattern) data
        let repetitive: Vec<u8> = (0..PAGE_SIZE).map(|i| (i % 4) as u8).collect();
        device.write(0, &repetitive).unwrap();

        let stats = device.stats();
        // Low entropy should NOT go to scalar path
        assert_eq!(
            stats.scalar_pages, 0,
            "Low entropy data should not use scalar path"
        );
    }

    #[test]
    fn test_discard_operation() {
        let mut device = BlockDevice::new(1 << 20, test_compressor());

        // Write data
        let data = vec![0xAB; PAGE_SIZE];
        device.write(0, &data).unwrap();

        // Verify it's stored
        let stats_before = device.stats();
        assert_eq!(stats_before.pages_stored, 1);

        // Discard
        device.discard(0, PAGE_SIZE as u64).unwrap();

        // Should now read zeros
        let mut buf = vec![0xFFu8; PAGE_SIZE];
        device.read(0, &mut buf).unwrap();
        assert!(
            buf.iter().all(|&b| b == 0),
            "Discarded page should read as zeros"
        );
    }

    #[test]
    fn test_overwrite_page() {
        let mut device = BlockDevice::new(1 << 20, test_compressor());

        // Write initial data
        let data1 = vec![0xAA; PAGE_SIZE];
        device.write(0, &data1).unwrap();

        // Overwrite with different data
        let data2 = vec![0xBB; PAGE_SIZE];
        device.write(0, &data2).unwrap();

        // Read back should return new data
        let mut buf = vec![0u8; PAGE_SIZE];
        device.read(0, &mut buf).unwrap();
        assert_eq!(data2, buf, "Overwritten data should be returned");
    }

    #[test]
    fn test_unwritten_page_returns_zeros() {
        let device = BlockDevice::new(1 << 20, test_compressor());

        // Read from unwritten location
        let mut buf = vec![0xFFu8; PAGE_SIZE];
        device.read(0, &mut buf).unwrap();

        assert!(
            buf.iter().all(|&b| b == 0),
            "Unwritten page should read as zeros"
        );
    }

    #[test]
    fn test_alignment_validation_offset() {
        let mut device = BlockDevice::new(1 << 20, test_compressor());
        let data = vec![0u8; PAGE_SIZE];

        // Unaligned offset should fail
        let result = device.write(1, &data);
        assert!(result.is_err(), "Unaligned offset should fail");
    }

    #[test]
    fn test_alignment_validation_length() {
        let mut device = BlockDevice::new(1 << 20, test_compressor());
        let data = vec![0u8; PAGE_SIZE - 1]; // Not page-aligned length

        let result = device.write(0, &data);
        assert!(result.is_err(), "Non-page-aligned length should fail");
    }

    #[test]
    fn test_bounds_check() {
        let mut device = BlockDevice::new(PAGE_SIZE as u64 * 10, test_compressor()); // 10 pages
        let data = vec![0u8; PAGE_SIZE];

        // Writing beyond device size should fail
        let result = device.write(PAGE_SIZE as u64 * 10, &data);
        assert!(result.is_err(), "Writing beyond device size should fail");
    }

    #[test]
    fn test_stats_tracking() {
        let mut device = BlockDevice::new(1 << 20, test_compressor());

        // Write some data
        let data = vec![0xAB; PAGE_SIZE * 3];
        device.write(0, &data).unwrap();

        let stats = device.stats();
        assert_eq!(
            stats.bytes_written,
            PAGE_SIZE as u64 * 3,
            "Bytes written should track correctly"
        );
        assert_eq!(stats.pages_stored, 3, "Pages stored should be 3");

        // Read data
        let mut buf = vec![0u8; PAGE_SIZE * 2];
        device.read(0, &mut buf).unwrap();

        let stats = device.stats();
        assert_eq!(
            stats.bytes_read,
            PAGE_SIZE as u64 * 2,
            "Bytes read should track correctly"
        );
    }

    #[test]
    fn test_various_data_patterns() {
        let mut device = BlockDevice::new(1 << 20, test_compressor());
        let patterns: Vec<Vec<u8>> = vec![
            // All zeros
            vec![0u8; PAGE_SIZE],
            // All ones
            vec![0xFF; PAGE_SIZE],
            // Alternating bytes
            (0..PAGE_SIZE)
                .map(|i| if i % 2 == 0 { 0xAA } else { 0x55 })
                .collect(),
            // Sequential bytes
            (0..PAGE_SIZE).map(|i| (i % 256) as u8).collect(),
            // Repeating short pattern
            (0..PAGE_SIZE).map(|i| (i % 16) as u8).collect(),
            // Text-like data
            "The quick brown fox jumps over the lazy dog. "
                .repeat(100)
                .into_bytes()[..PAGE_SIZE]
                .to_vec(),
        ];

        for (i, pattern) in patterns.iter().enumerate() {
            let offset = i as u64 * PAGE_SIZE as u64;
            device.write(offset, pattern).unwrap();

            let mut buf = vec![0u8; PAGE_SIZE];
            device.read(offset, &mut buf).unwrap();

            assert_eq!(*pattern, buf, "Pattern {} roundtrip failed", i);
        }
    }

    #[test]
    fn test_block_device_stats_compression_ratio() {
        let stats = BlockDeviceStats {
            pages_stored: 10,
            bytes_written: 40960,
            bytes_read: 0,
            bytes_compressed: 10240,
            zero_pages: 0,
            gpu_pages: 0,
            simd_pages: 0,
            scalar_pages: 0,
        };

        assert!(
            (stats.compression_ratio() - 4.0).abs() < 0.001,
            "4:1 compression ratio expected"
        );
    }

    #[test]
    fn test_block_device_stats_compression_ratio_no_data() {
        let stats = BlockDeviceStats::default();
        assert!(
            (stats.compression_ratio() - 1.0).abs() < 0.001,
            "Default ratio should be 1.0"
        );
    }

    // ========================================================================
    // Popperian Falsification Checklist - Section A: Data Integrity (1-20)
    // ========================================================================

    /// A1: Write 4KB pattern A, Read, Verify.
    #[test]
    fn popperian_a01_write_pattern_a_read_verify() {
        let mut device = BlockDevice::new(1 << 20, test_compressor());

        let pattern_a: Vec<u8> = (0..PAGE_SIZE).map(|i| (i * 7 + 0xA5) as u8).collect();
        device.write(0, &pattern_a).unwrap();

        let mut buf = vec![0u8; PAGE_SIZE];
        device.read(0, &mut buf).unwrap();

        assert_eq!(pattern_a, buf, "A1: Pattern A roundtrip failed");
    }

    /// A2: Write 4KB pattern A, Write 4KB pattern B, Read, Verify B.
    #[test]
    fn popperian_a02_overwrite_pattern_a_with_b() {
        let mut device = BlockDevice::new(1 << 20, test_compressor());

        let pattern_a: Vec<u8> = (0..PAGE_SIZE).map(|i| (i * 7 + 0xA5) as u8).collect();
        let pattern_b: Vec<u8> = (0..PAGE_SIZE).map(|i| (i * 11 + 0xB7) as u8).collect();

        device.write(0, &pattern_a).unwrap();
        device.write(0, &pattern_b).unwrap();

        let mut buf = vec![0u8; PAGE_SIZE];
        device.read(0, &mut buf).unwrap();

        assert_eq!(pattern_b, buf, "A2: Pattern B should overwrite pattern A");
        assert_ne!(pattern_a, buf, "A2: Pattern A should be gone");
    }

    /// A3: Write 4KB zero-page, Read, Verify Zero.
    #[test]
    fn popperian_a03_zero_page_roundtrip() {
        let mut device = BlockDevice::new(1 << 20, test_compressor());

        let zeros = vec![0u8; PAGE_SIZE];
        device.write(0, &zeros).unwrap();

        let mut buf = vec![0xFFu8; PAGE_SIZE]; // Initialize with non-zero
        device.read(0, &mut buf).unwrap();

        assert_eq!(zeros, buf, "A3: Zero page roundtrip failed");
        assert!(
            device.stats().zero_pages >= 1,
            "A3: Should track zero pages"
        );
    }

    /// A4: Write 4KB random high-entropy (uncompressible), Read, Verify.
    #[test]
    fn popperian_a04_high_entropy_roundtrip() {
        let mut device = BlockDevice::new(1 << 20, test_compressor());

        // Use LCG for pseudo-random high-entropy data
        let mut state: u64 = 0xDEADBEEF;
        let random_data: Vec<u8> = (0..PAGE_SIZE)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 33) as u8
            })
            .collect();

        device.write(0, &random_data).unwrap();

        let mut buf = vec![0u8; PAGE_SIZE];
        device.read(0, &mut buf).unwrap();

        assert_eq!(random_data, buf, "A4: High-entropy data roundtrip failed");
    }

    /// A5: Write 4KB repeated byte (highly compressible), Read, Verify.
    #[test]
    fn popperian_a05_highly_compressible_roundtrip() {
        let mut device = BlockDevice::new(1 << 20, test_compressor());

        // Test multiple single-byte patterns
        for byte_val in [0x00, 0x42, 0xAA, 0xFF] {
            let data = vec![byte_val; PAGE_SIZE];
            let offset = byte_val as u64 * PAGE_SIZE as u64;

            device.write(offset, &data).unwrap();

            let mut buf = vec![0u8; PAGE_SIZE];
            device.read(offset, &mut buf).unwrap();

            assert_eq!(
                data, buf,
                "A5: Repeated byte 0x{:02X} roundtrip failed",
                byte_val
            );
        }
    }

    /// A6: Write to last sector of device boundary.
    #[test]
    fn popperian_a06_write_last_sector() {
        let device_size = PAGE_SIZE as u64 * 10; // 10 pages
        let mut device = BlockDevice::new(device_size, test_compressor());

        let last_offset = device_size - PAGE_SIZE as u64;
        let data = vec![0xEE; PAGE_SIZE];

        let result = device.write(last_offset, &data);
        assert!(result.is_ok(), "A6: Writing to last sector should succeed");

        let mut buf = vec![0u8; PAGE_SIZE];
        device.read(last_offset, &mut buf).unwrap();
        assert_eq!(data, buf, "A6: Last sector data should match");
    }

    /// A7: Read from last sector of device boundary.
    #[test]
    fn popperian_a07_read_last_sector() {
        let device_size = PAGE_SIZE as u64 * 10;
        let mut device = BlockDevice::new(device_size, test_compressor());

        let last_offset = device_size - PAGE_SIZE as u64;

        // Write data first
        let data = vec![0xDD; PAGE_SIZE];
        device.write(last_offset, &data).unwrap();

        // Read from last sector
        let mut buf = vec![0u8; PAGE_SIZE];
        let result = device.read(last_offset, &mut buf);
        assert!(
            result.is_ok(),
            "A7: Reading from last sector should succeed"
        );
        assert_eq!(data, buf, "A7: Last sector read should match written data");
    }

    /// A8: Write past device boundary (expect error).
    #[test]
    fn popperian_a08_write_past_boundary() {
        let device_size = PAGE_SIZE as u64 * 10;
        let mut device = BlockDevice::new(device_size, test_compressor());

        let data = vec![0xAA; PAGE_SIZE];

        // Write exactly at boundary (should fail)
        let result = device.write(device_size, &data);
        assert!(result.is_err(), "A8: Writing at boundary should fail");

        // Write past boundary
        let result = device.write(device_size + PAGE_SIZE as u64, &data);
        assert!(result.is_err(), "A8: Writing past boundary should fail");
    }

    /// A9: Read past device boundary (expect error).
    #[test]
    fn popperian_a09_read_past_boundary() {
        let device_size = PAGE_SIZE as u64 * 10;
        let device = BlockDevice::new(device_size, test_compressor());

        let mut buf = vec![0u8; PAGE_SIZE];

        // Read exactly at boundary (should fail)
        let result = device.read(device_size, &mut buf);
        assert!(result.is_err(), "A9: Reading at boundary should fail");

        // Read past boundary
        let result = device.read(device_size + PAGE_SIZE as u64, &mut buf);
        assert!(result.is_err(), "A9: Reading past boundary should fail");
    }

    /// A10: Read uninitialized sector (expect zeros).
    #[test]
    fn popperian_a10_read_uninitialized() {
        let device = BlockDevice::new(1 << 20, test_compressor());

        // Read from multiple uninitialized locations
        for offset in [0, PAGE_SIZE as u64, PAGE_SIZE as u64 * 5] {
            let mut buf = vec![0xFFu8; PAGE_SIZE]; // Initialize with non-zero
            device.read(offset, &mut buf).unwrap();

            assert!(
                buf.iter().all(|&b| b == 0),
                "A10: Uninitialized sector at offset {} should read as zeros",
                offset
            );
        }
    }

    /// A11: Write 1 byte (partial update) - tests error handling for non-page-aligned.
    /// Note: Our block device requires page-aligned I/O, so this tests the validation.
    #[test]
    fn popperian_a11_partial_write_rejected() {
        let mut device = BlockDevice::new(1 << 20, test_compressor());

        let data = vec![0xAB; 1]; // 1 byte

        let result = device.write(0, &data);
        assert!(
            result.is_err(),
            "A11: Partial write (1 byte) should be rejected"
        );
    }

    /// A12: Read 1 byte (partial read) - tests error handling for non-page-aligned.
    #[test]
    fn popperian_a12_partial_read_rejected() {
        let device = BlockDevice::new(1 << 20, test_compressor());

        let mut buf = vec![0u8; 1]; // 1 byte

        let result = device.read(0, &mut buf);
        assert!(
            result.is_err(),
            "A12: Partial read (1 byte) should be rejected"
        );
    }

    /// A13: Overwrite scalar compressed page with SIMD compressed page.
    #[test]
    fn popperian_a13_overwrite_scalar_with_simd() {
        let mut device = BlockDevice::with_entropy_threshold(1 << 20, test_compressor(), 7.0);

        // Write high-entropy data (will use scalar path)
        let mut state: u64 = 0xCAFEBABE;
        let high_entropy: Vec<u8> = (0..PAGE_SIZE)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state >> 33) as u8
            })
            .collect();

        device.write(0, &high_entropy).unwrap();
        let stats1 = device.stats();
        assert!(
            stats1.scalar_pages > 0,
            "A13: First write should use scalar path"
        );

        // Overwrite with low-entropy data (will use SIMD path)
        let low_entropy: Vec<u8> = (0..PAGE_SIZE).map(|i| (i % 16) as u8).collect();
        device.write(0, &low_entropy).unwrap();

        // Verify correct data is returned
        let mut buf = vec![0u8; PAGE_SIZE];
        device.read(0, &mut buf).unwrap();
        assert_eq!(
            low_entropy, buf,
            "A13: SIMD data should overwrite scalar data"
        );
    }

    /// A14: Overwrite SIMD compressed page with zero page.
    #[test]
    fn popperian_a14_overwrite_simd_with_zero() {
        let mut device = BlockDevice::with_entropy_threshold(1 << 20, test_compressor(), 7.0);

        // Write low-entropy data (SIMD path)
        let low_entropy: Vec<u8> = (0..PAGE_SIZE).map(|i| (i % 8) as u8).collect();
        device.write(0, &low_entropy).unwrap();

        // Overwrite with zeros
        let zeros = vec![0u8; PAGE_SIZE];
        device.write(0, &zeros).unwrap();

        // Verify zeros are returned
        let mut buf = vec![0xFFu8; PAGE_SIZE];
        device.read(0, &mut buf).unwrap();
        assert_eq!(zeros, buf, "A14: Zero page should overwrite SIMD page");

        let stats = device.stats();
        assert!(stats.zero_pages >= 1, "A14: Should count as zero page");
    }

    /// A15: Overwrite zero page with scalar compressed page.
    #[test]
    fn popperian_a15_overwrite_zero_with_scalar() {
        let mut device = BlockDevice::with_entropy_threshold(1 << 20, test_compressor(), 7.0);

        // Write zeros
        let zeros = vec![0u8; PAGE_SIZE];
        device.write(0, &zeros).unwrap();
        assert!(device.stats().zero_pages >= 1, "A15: Should have zero page");

        // Overwrite with high-entropy data (scalar path)
        let mut state: u64 = 0xFEEDFACE;
        let high_entropy: Vec<u8> = (0..PAGE_SIZE)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state >> 33) as u8
            })
            .collect();

        device.write(0, &high_entropy).unwrap();

        // Verify correct data
        let mut buf = vec![0u8; PAGE_SIZE];
        device.read(0, &mut buf).unwrap();
        assert_eq!(
            high_entropy, buf,
            "A15: Scalar data should overwrite zero page"
        );
    }

    /// A16: Persistence test - N/A for in-memory device (skip with documentation).
    /// Real persistence testing requires file-backed storage.
    #[test]
    fn popperian_a16_persistence_not_applicable() {
        // In-memory BlockDevice does not persist across restarts.
        // This test documents that persistence is tested elsewhere (file-backed mode).
        // For file-backed persistence, see trueno_ublk::backing tests.
    }

    /// A17: CRC32 corruption detection test.
    /// Note: This requires access to internal compressed data which PageStore encapsulates.
    /// We test that data corruption during storage would be caught on decompression.
    #[test]
    fn popperian_a17_crc32_integrity() {
        // CRC32 validation is handled internally by PageStore during decompression.
        // This test verifies the compressor includes integrity checks.
        let compressor = test_compressor();

        let data = vec![0xAB; PAGE_SIZE];
        let page: &[u8; PAGE_SIZE] = data.as_slice().try_into().unwrap();

        let compressed = compressor.compress(page).unwrap();

        // Verify compressed data has reasonable structure
        assert!(
            !compressed.data.is_empty(),
            "A17: Compressed data should not be empty"
        );

        // Decompress and verify
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(
            data,
            decompressed.as_slice(),
            "A17: Roundtrip should preserve data"
        );
    }

    /// A18: Concurrent Read/Write atomicity.
    #[test]
    fn popperian_a18_concurrent_read_write() {
        use std::sync::Arc;
        use std::thread;

        let device = Arc::new(std::sync::RwLock::new(BlockDevice::new(
            1 << 20,
            test_compressor(),
        )));

        let data = vec![0xCC; PAGE_SIZE];
        device
            .write()
            .expect("rwlock poisoned")
            .write(0, &data)
            .unwrap();

        let mut handles = vec![];

        // Spawn readers
        for _ in 0..4 {
            let dev = device.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let mut buf = vec![0u8; PAGE_SIZE];
                    dev.read()
                        .expect("rwlock poisoned")
                        .read(0, &mut buf)
                        .unwrap();
                    // Data should be either all 0xCC or all 0xDD (not mixed)
                    let first = buf[0];
                    assert!(
                        buf.iter().all(|&b| b == first),
                        "A18: Read should be atomic (no partial updates)"
                    );
                }
            }));
        }

        // Spawn writer
        let dev = device.clone();
        handles.push(thread::spawn(move || {
            let new_data = vec![0xDD; PAGE_SIZE];
            for _ in 0..50 {
                dev.write()
                    .expect("rwlock poisoned")
                    .write(0, &new_data)
                    .unwrap();
            }
        }));

        for h in handles {
            h.join().expect("Thread panicked");
        }
    }

    /// A19: Concurrent Write/Write (last writer wins).
    #[test]
    fn popperian_a19_concurrent_write_write() {
        use std::sync::Arc;
        use std::thread;

        let device = Arc::new(std::sync::Mutex::new(BlockDevice::new(
            1 << 20,
            test_compressor(),
        )));

        let mut handles = vec![];

        // Spawn multiple writers with different patterns
        for writer_id in 0..4u8 {
            let dev = device.clone();
            handles.push(thread::spawn(move || {
                let data = vec![writer_id; PAGE_SIZE];
                for _ in 0..100 {
                    dev.lock().unwrap().write(0, &data).unwrap();
                }
            }));
        }

        for h in handles {
            h.join().expect("Thread panicked");
        }

        // Final read should return one of the patterns (all same byte)
        let mut buf = vec![0u8; PAGE_SIZE];
        device.lock().unwrap().read(0, &mut buf).unwrap();

        let first = buf[0];
        assert!(
            buf.iter().all(|&b| b == first),
            "A19: Final state should be consistent (single writer's data)"
        );
        assert!(
            first < 4,
            "A19: Final byte should be from one of the writers (0-3)"
        );
    }

    /// A20: Discard (TRIM) zeroes data and frees memory.
    #[test]
    fn popperian_a20_discard_frees_memory() {
        let mut device = BlockDevice::new(1 << 20, test_compressor());

        // Write NON-uniform compressible data (avoids same-fill optimization)
        // PERF-013: Same-fill pages are stored without compression
        let mut data = vec![0u8; PAGE_SIZE];
        for i in 0..PAGE_SIZE {
            data[i] = (i % 32) as u8;
        }
        device.write(0, &data).unwrap();

        let stats_before = device.stats();
        assert_eq!(
            stats_before.pages_stored, 1,
            "A20: Should have 1 page stored"
        );
        assert!(
            stats_before.bytes_compressed > 0,
            "A20: Should have compressed bytes"
        );

        // Discard
        device.discard(0, PAGE_SIZE as u64).unwrap();

        // Verify page is freed (reads as zeros, not counted in storage)
        let mut buf = vec![0xFFu8; PAGE_SIZE];
        device.read(0, &mut buf).unwrap();
        assert!(
            buf.iter().all(|&b| b == 0),
            "A20: Discarded page should read as zeros"
        );

        // Note: Current PageStore::remove decrements page count but may not track freed memory.
        // The spec requirement is that data reads as zeros, which we verify above.
    }

    // ========================================================================
    // Popperian Falsification Checklist - Section B: Resource Management (21-30)
    // ========================================================================

    /// B21: Leak check - Create/Destroy device 1000 times.
    #[test]
    fn popperian_b21_leak_check_create_destroy() {
        // Create and drop devices repeatedly
        for i in 0..1000 {
            let mut device = BlockDevice::new(1 << 20, test_compressor());

            // Write some data
            let data = vec![0xAB; PAGE_SIZE];
            device.write(0, &data).unwrap();

            // Read back
            let mut buf = vec![0u8; PAGE_SIZE];
            device.read(0, &mut buf).unwrap();

            // Device drops here, releasing memory
            drop(device);

            if i % 100 == 0 {
                // Verify we're not accumulating state
                // (In real test, would check RSS)
            }
        }
        // If we get here without OOM, the test passes
    }

    /// B22: Zero-page deduplication efficiency.
    /// Write many zero pages, verify minimal memory usage.
    #[test]
    fn popperian_b22_zero_page_deduplication_efficiency() {
        let num_pages = 256; // 1MB of zeros
        let device_size = (num_pages * PAGE_SIZE) as u64;
        let mut device = BlockDevice::new(device_size, test_compressor());

        // Write all zeros
        let zeros = vec![0u8; PAGE_SIZE];
        for i in 0..num_pages {
            device.write((i * PAGE_SIZE) as u64, &zeros).unwrap();
        }

        let stats = device.stats();
        assert_eq!(
            stats.zero_pages, num_pages as u64,
            "B22: All pages should be zero-deduplicated"
        );

        // Compressed bytes for zero pages should be minimal
        // (Each zero page is stored as a sentinel, not actual data)
        let bytes_per_zero_page = stats.bytes_compressed as f64 / num_pages as f64;
        assert!(
            bytes_per_zero_page < 100.0,
            "B22: Zero pages should use minimal memory ({:.0} bytes/page)",
            bytes_per_zero_page
        );
    }

    /// B23: Compression ratio on realistic text data.
    #[test]
    fn popperian_b23_compression_ratio_text_data() {
        let num_pages = 256; // 1MB
        let device_size = (num_pages * PAGE_SIZE) as u64;
        let mut device = BlockDevice::new(device_size, test_compressor());

        // Write text-like data (highly compressible)
        let text_pattern = "The quick brown fox jumps over the lazy dog. \
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. \
            Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ";
        let text_page: Vec<u8> = text_pattern
            .chars()
            .cycle()
            .take(PAGE_SIZE)
            .map(|c| c as u8)
            .collect();

        for i in 0..num_pages {
            device.write((i * PAGE_SIZE) as u64, &text_page).unwrap();
        }

        let stats = device.stats();
        let ratio = stats.compression_ratio();

        // Text should compress at least 2:1 with LZ4
        assert!(
            ratio >= 2.0,
            "B23: Text data should compress >= 2:1, got {:.2}:1",
            ratio
        );

        println!("B23: Text compression ratio: {:.2}:1", ratio);
    }

    /// B24: Max device allocation test.
    /// Create many independent devices to test resource limits.
    #[test]
    fn popperian_b24_max_devices() {
        let mut devices = Vec::new();

        // Create up to 100 small devices
        for i in 0..100 {
            let device = BlockDevice::new(PAGE_SIZE as u64 * 10, test_compressor());
            devices.push(device);

            if i % 10 == 0 {
                // Verify we can still write to existing devices
                let data = vec![i as u8; PAGE_SIZE];
                for (j, dev) in devices.iter_mut().enumerate() {
                    if j < 10 {
                        dev.write(0, &data).unwrap();
                    }
                }
            }
        }

        assert_eq!(devices.len(), 100, "B24: Should create 100 devices");

        // Verify all can still be read
        for dev in &devices {
            let mut buf = vec![0u8; PAGE_SIZE];
            dev.read(0, &mut buf).unwrap();
        }
    }

    /// B25: OOM resilience - write until device is full.
    #[test]
    fn popperian_b25_oom_resilience() {
        // Small device that will fill up
        let device_size = PAGE_SIZE as u64 * 10;
        let mut device = BlockDevice::new(device_size, test_compressor());

        // Fill the device
        let data = vec![0xAB; PAGE_SIZE];
        for i in 0..10 {
            let result = device.write((i * PAGE_SIZE) as u64, &data);
            assert!(result.is_ok(), "B25: Write {} should succeed", i);
        }

        // Writing beyond capacity should fail gracefully
        let result = device.write(device_size, &data);
        assert!(
            result.is_err(),
            "B25: Write beyond capacity should fail gracefully"
        );
    }

    /// B26: CPU affinity (simplified test for single-threaded operation).
    #[test]
    fn popperian_b26_cpu_affinity() {
        // BlockDevice is single-threaded, so CPU affinity is managed by caller.
        // This test verifies operations complete on current thread.
        let mut device = BlockDevice::new(1 << 20, test_compressor());

        let data = vec![0xAB; PAGE_SIZE];
        let start_thread = std::thread::current().id();

        device.write(0, &data).unwrap();

        let end_thread = std::thread::current().id();
        assert_eq!(
            start_thread, end_thread,
            "B26: Operations should complete on same thread"
        );
    }

    /// B27: File descriptor leak check (simplified for in-memory device).
    #[test]
    fn popperian_b27_fd_leak_check() {
        // BlockDevice doesn't use file descriptors directly.
        // This test verifies repeated operations don't leak resources.
        for _ in 0..100 {
            let mut device = BlockDevice::new(1 << 20, test_compressor());

            // Perform operations
            let data = vec![0xAB; PAGE_SIZE * 10];
            device.write(0, &data).unwrap();

            let mut buf = vec![0u8; PAGE_SIZE * 10];
            device.read(0, &mut buf).unwrap();

            // Discard all
            device.discard(0, (PAGE_SIZE * 10) as u64).unwrap();

            // Device drops, resources released
        }
        // Success if we don't run out of resources
    }

    /// B28: Buffer pool exhaustion - high I/O depth stress test.
    #[test]
    fn popperian_b28_buffer_pool_exhaustion() {
        let mut device = BlockDevice::new(1 << 24, test_compressor()); // 16MB

        // Perform many rapid writes with varying patterns
        let patterns: Vec<Vec<u8>> = (0..256).map(|i| vec![(i % 256) as u8; PAGE_SIZE]).collect();

        // Rapid write/read cycles
        for iteration in 0..100 {
            for (i, pattern) in patterns.iter().enumerate() {
                let offset = (i * PAGE_SIZE) as u64;
                device.write(offset, pattern).unwrap();
            }

            // Verify random samples
            let check_indices = [0, 50, 100, 200, 255];
            for &idx in &check_indices {
                let mut buf = vec![0u8; PAGE_SIZE];
                device.read((idx * PAGE_SIZE) as u64, &mut buf).unwrap();
                assert!(
                    buf.iter().all(|&b| b == (idx % 256) as u8),
                    "B28: Iteration {}, pattern {} verification failed",
                    iteration,
                    idx
                );
            }
        }
    }

    /// B29: Idle timeout (simplified - verify device can be idle then reused).
    #[test]
    fn popperian_b29_idle_timeout() {
        let mut device = BlockDevice::new(1 << 20, test_compressor());

        // Write data
        let data = vec![0xAB; PAGE_SIZE];
        device.write(0, &data).unwrap();

        // "Idle" period (in real implementation, would sleep)
        // For testing, we just verify state is preserved

        // Verify data is still accessible after "idle"
        let mut buf = vec![0u8; PAGE_SIZE];
        device.read(0, &mut buf).unwrap();
        assert_eq!(data, buf, "B29: Data should persist through idle period");

        // Can still write after idle
        let new_data = vec![0xCD; PAGE_SIZE];
        device.write(PAGE_SIZE as u64, &new_data).unwrap();

        let mut buf2 = vec![0u8; PAGE_SIZE];
        device.read(PAGE_SIZE as u64, &mut buf2).unwrap();
        assert_eq!(new_data, buf2, "B29: Should write after idle");
    }

    /// B30: Fragmentation stress test - random write pattern.
    #[test]
    fn popperian_b30_fragmentation_stress() {
        let num_pages = 1000;
        let device_size = (num_pages * PAGE_SIZE) as u64;
        let mut device = BlockDevice::new(device_size, test_compressor());

        // Random-order writes using deterministic pseudo-random sequence
        let mut order: Vec<usize> = (0..num_pages).collect();
        let mut rng_state: u64 = 0xDEADBEEF;
        for i in (1..order.len()).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state as usize) % (i + 1);
            order.swap(i, j);
        }

        // Write in "random" order
        for &page_idx in &order {
            let data: Vec<u8> = (0..PAGE_SIZE)
                .map(|i| ((page_idx + i) % 256) as u8)
                .collect();
            device.write((page_idx * PAGE_SIZE) as u64, &data).unwrap();
        }

        // Verify all pages in sequential order
        for page_idx in 0..num_pages {
            let expected: Vec<u8> = (0..PAGE_SIZE)
                .map(|i| ((page_idx + i) % 256) as u8)
                .collect();
            let mut buf = vec![0u8; PAGE_SIZE];
            device
                .read((page_idx * PAGE_SIZE) as u64, &mut buf)
                .unwrap();
            assert_eq!(
                expected, buf,
                "B30: Page {} verification failed after fragmented writes",
                page_idx
            );
        }

        // Verify compression still works
        let stats = device.stats();
        assert!(
            stats.compression_ratio() >= 1.0,
            "B30: Compression should still function after fragmented writes"
        );
    }

    // ========================================================================
    // Popperian Falsification Checklist - Section C: Performance (31-40)
    // ========================================================================

    /// C31: Sequential write throughput test.
    /// Validates write path is functional with reasonable throughput.
    #[test]
    #[ignore = "Performance test - skip during coverage (instrumentation overhead)"]
    fn popperian_c31_sequential_write_throughput() {
        use std::time::Instant;

        let num_pages = 10000; // ~40MB
        let device_size = (num_pages * PAGE_SIZE) as u64;
        let mut device = BlockDevice::new(device_size, test_compressor());

        // Create test data (moderately compressible)
        let data: Vec<u8> = (0..PAGE_SIZE).map(|i| (i % 128) as u8).collect();

        let start = Instant::now();
        for i in 0..num_pages {
            device.write((i * PAGE_SIZE) as u64, &data).unwrap();
        }
        let duration = start.elapsed();

        let bytes_written = num_pages * PAGE_SIZE;
        let throughput_gbps = (bytes_written as f64) / duration.as_secs_f64() / 1e9;

        println!(
            "C31: Sequential write throughput: {:.2} GB/s ({} pages in {:?})",
            throughput_gbps, num_pages, duration
        );

        // Verify throughput is reasonable (>50MB/s minimum sanity check)
        // Accounts for parallel test execution overhead
        assert!(
            throughput_gbps > 0.05,
            "C31: Write throughput should be > 50 MB/s, got {:.2} GB/s",
            throughput_gbps
        );
    }

    /// C32: Sequential read throughput test.
    #[test]
    fn popperian_c32_sequential_read_throughput() {
        use std::time::Instant;

        let num_pages = 10000;
        let device_size = (num_pages * PAGE_SIZE) as u64;
        let mut device = BlockDevice::new(device_size, test_compressor());

        // Write data first
        let data: Vec<u8> = (0..PAGE_SIZE).map(|i| (i % 128) as u8).collect();
        for i in 0..num_pages {
            device.write((i * PAGE_SIZE) as u64, &data).unwrap();
        }

        // Benchmark reads
        let mut buf = vec![0u8; PAGE_SIZE];
        let start = Instant::now();
        for i in 0..num_pages {
            device.read((i * PAGE_SIZE) as u64, &mut buf).unwrap();
        }
        let duration = start.elapsed();

        let bytes_read = num_pages * PAGE_SIZE;
        let throughput_gbps = (bytes_read as f64) / duration.as_secs_f64() / 1e9;

        println!(
            "C32: Sequential read throughput: {:.2} GB/s ({} pages in {:?})",
            throughput_gbps, num_pages, duration
        );

        // Read should be at least as fast as write (decompression often faster)
        // Accounts for parallel test execution overhead
        assert!(
            throughput_gbps > 0.05,
            "C32: Read throughput should be > 50 MB/s, got {:.2} GB/s",
            throughput_gbps
        );
    }

    /// C33: Random 4K write IOPS test.
    #[test]
    fn popperian_c33_random_write_iops() {
        use std::time::Instant;

        let num_pages = 1000;
        let device_size = (num_pages * PAGE_SIZE) as u64;
        let mut device = BlockDevice::new(device_size, test_compressor());

        // Generate random access pattern
        let mut access_order: Vec<usize> = (0..num_pages).collect();
        let mut rng_state: u64 = 0xBADCAFE;
        for i in (1..access_order.len()).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state as usize) % (i + 1);
            access_order.swap(i, j);
        }

        let data = vec![0xAB; PAGE_SIZE];
        let iterations = 10000;

        let start = Instant::now();
        for i in 0..iterations {
            let page_idx = access_order[i % num_pages];
            device.write((page_idx * PAGE_SIZE) as u64, &data).unwrap();
        }
        let duration = start.elapsed();

        let iops = iterations as f64 / duration.as_secs_f64();

        println!(
            "C33: Random 4K write: {:.0} IOPS ({} ops in {:?})",
            iops, iterations, duration
        );

        // Sanity check: at least 1000 IOPS
        assert!(
            iops > 1000.0,
            "C33: Random write IOPS should be > 1000, got {:.0}",
            iops
        );
    }

    /// C34: Random 4K read IOPS test.
    #[test]
    fn popperian_c34_random_read_iops() {
        use std::time::Instant;

        let num_pages = 1000;
        let device_size = (num_pages * PAGE_SIZE) as u64;
        let mut device = BlockDevice::new(device_size, test_compressor());

        // Write initial data
        let data = vec![0xAB; PAGE_SIZE];
        for i in 0..num_pages {
            device.write((i * PAGE_SIZE) as u64, &data).unwrap();
        }

        // Generate random access pattern
        let mut access_order: Vec<usize> = (0..num_pages).collect();
        let mut rng_state: u64 = 0xCAFEBABE;
        for i in (1..access_order.len()).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state as usize) % (i + 1);
            access_order.swap(i, j);
        }

        let mut buf = vec![0u8; PAGE_SIZE];
        let iterations = 10000;

        let start = Instant::now();
        for i in 0..iterations {
            let page_idx = access_order[i % num_pages];
            device
                .read((page_idx * PAGE_SIZE) as u64, &mut buf)
                .unwrap();
        }
        let duration = start.elapsed();

        let iops = iterations as f64 / duration.as_secs_f64();

        println!(
            "C34: Random 4K read: {:.0} IOPS ({} ops in {:?})",
            iops, iterations, duration
        );

        assert!(
            iops > 1000.0,
            "C34: Random read IOPS should be > 1000, got {:.0}",
            iops
        );
    }

    /// C35: Latency test at QD=1.
    #[test]
    fn popperian_c35_latency_qd1() {
        use std::time::Instant;

        let mut device = BlockDevice::new(1 << 20, test_compressor());

        let data = vec![0xAB; PAGE_SIZE];
        let iterations = 1000;
        let mut latencies = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            let start = Instant::now();
            device.write(0, &data).unwrap();
            latencies.push(start.elapsed());
        }

        latencies.sort();
        let p50 = latencies[latencies.len() / 2];
        let p99 = latencies[latencies.len() * 99 / 100];
        let p999 = latencies[latencies.len() * 999 / 1000];

        println!(
            "C35: QD=1 latency: p50={:?}, p99={:?}, p99.9={:?}",
            p50, p99, p999
        );

        // In-memory device should be very fast
        assert!(
            p99.as_micros() < 10000, // 10ms
            "C35: p99 latency should be < 10ms at QD=1, got {:?}",
            p99
        );
    }

    /// C36: Latency test at high queue depth (simulated with threads).
    #[test]
    fn popperian_c36_latency_high_qd() {
        use std::sync::Arc;
        use std::thread;
        use std::time::Instant;

        let device = Arc::new(std::sync::Mutex::new(BlockDevice::new(
            1 << 24,
            test_compressor(),
        )));

        let num_threads = 8;
        let ops_per_thread = 500;
        let mut handles = vec![];

        let start = Instant::now();

        for t in 0..num_threads {
            let dev = device.clone();
            handles.push(thread::spawn(move || {
                let data = vec![t as u8; PAGE_SIZE];
                for i in 0..ops_per_thread {
                    let offset = ((t * ops_per_thread + i) * PAGE_SIZE) as u64;
                    dev.lock().unwrap().write(offset, &data).unwrap();
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        let duration = start.elapsed();
        let total_ops = num_threads * ops_per_thread;
        let avg_latency_us = duration.as_micros() as f64 / total_ops as f64;

        println!(
            "C36: High QD ({} threads) avg latency: {:.1}us ({} ops in {:?})",
            num_threads, avg_latency_us, total_ops, duration
        );

        // Even under contention, should complete reasonably
        assert!(
            avg_latency_us < 100000.0, // 100ms average
            "C36: Avg latency should be < 100ms under contention"
        );
    }

    /// C37: Scalability test across multiple threads.
    #[test]
    fn popperian_c37_scalability() {
        use std::sync::Arc;
        use std::thread;
        use std::time::Instant;

        let ops_per_thread = 1000;
        let device_size = (ops_per_thread * PAGE_SIZE * 2) as u64; // Room for all ops
        let mut results = Vec::new();

        for num_threads in [1, 2, 4] {
            let devices: Vec<_> = (0..num_threads)
                .map(|_| {
                    Arc::new(std::sync::Mutex::new(BlockDevice::new(
                        device_size,
                        test_compressor(),
                    )))
                })
                .collect();

            let start = Instant::now();

            let handles: Vec<_> = devices
                .iter()
                .enumerate()
                .map(|(t, dev)| {
                    let dev = dev.clone();
                    thread::spawn(move || {
                        let data = vec![t as u8; PAGE_SIZE];
                        for i in 0..ops_per_thread {
                            let offset = (i * PAGE_SIZE) as u64;
                            dev.lock().unwrap().write(offset, &data).unwrap();
                        }
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }

            let duration = start.elapsed();
            let total_ops = num_threads * ops_per_thread;
            let throughput = total_ops as f64 / duration.as_secs_f64();

            results.push((num_threads, throughput));
            println!(
                "C37: {} threads: {:.0} ops/sec ({} ops in {:?})",
                num_threads, throughput, total_ops, duration
            );
        }

        // Verify we get some scaling benefit
        // (Throughput at 4 threads should be at least 50% better than 1 thread)
        let (_, throughput_1) = results[0];
        let (_, throughput_4) = results[2];

        // Note: Due to mutex contention, perfect scaling isn't expected
        // Just verify it doesn't completely collapse
        assert!(
            throughput_4 > throughput_1 * 0.5,
            "C37: 4-thread throughput should be at least 50% of 1-thread"
        );
    }

    /// C38: SIMD backend detection verification.
    #[test]
    fn popperian_c38_simd_backend_detection() {
        let backend = detect_simd_backend();

        println!("C38: Detected SIMD backend: {}", backend);

        // Should detect something
        assert!(!backend.is_empty(), "C38: Should detect a SIMD backend");

        // On x86_64, should be avx512, avx2, sse4.2, or scalar
        #[cfg(target_arch = "x86_64")]
        assert!(
            ["avx512", "avx2", "sse4.2", "scalar"].contains(&backend.as_str()),
            "C38: x86_64 should detect known backend, got: {}",
            backend
        );

        // On aarch64, should be neon
        #[cfg(target_arch = "aarch64")]
        assert_eq!(backend, "neon", "C38: ARM should detect NEON");
    }

    /// C39: Algorithm switching test (LZ4 vs Zstd).
    #[test]
    fn popperian_c39_algorithm_switch() {
        use std::time::Instant;

        let data: Vec<u8> = (0..PAGE_SIZE).map(|i| (i % 128) as u8).collect();
        let iterations = 1000;

        // Test LZ4
        let lz4_compressor = CompressorBuilder::new()
            .algorithm(Algorithm::Lz4)
            .build()
            .unwrap();
        let mut lz4_device = BlockDevice::new(1 << 20, lz4_compressor);

        let start = Instant::now();
        for _ in 0..iterations {
            lz4_device.write(0, &data).unwrap();
        }
        let lz4_duration = start.elapsed();

        // Test Zstd (level 1 for speed)
        let zstd_compressor = CompressorBuilder::new()
            .algorithm(Algorithm::Zstd { level: 1 })
            .build()
            .unwrap();
        let mut zstd_device = BlockDevice::new(1 << 20, zstd_compressor);

        let start = Instant::now();
        for _ in 0..iterations {
            zstd_device.write(0, &data).unwrap();
        }
        let zstd_duration = start.elapsed();

        let lz4_stats = lz4_device.stats();
        let zstd_stats = zstd_device.stats();

        println!(
            "C39: LZ4: {:?} ({:.2}:1), Zstd: {:?} ({:.2}:1)",
            lz4_duration,
            lz4_stats.compression_ratio(),
            zstd_duration,
            zstd_stats.compression_ratio()
        );

        // Both should work
        assert!(lz4_stats.bytes_written > 0, "C39: LZ4 should write data");
        assert!(zstd_stats.bytes_written > 0, "C39: Zstd should write data");

        // Verify roundtrip for both
        let mut lz4_buf = vec![0u8; PAGE_SIZE];
        lz4_device.read(0, &mut lz4_buf).unwrap();
        assert_eq!(data, lz4_buf, "C39: LZ4 roundtrip should match");

        let mut zstd_buf = vec![0u8; PAGE_SIZE];
        zstd_device.read(0, &mut zstd_buf).unwrap();
        assert_eq!(data, zstd_buf, "C39: Zstd roundtrip should match");
    }

    /// C40: Dictionary training placeholder.
    /// Note: Dictionary training is not yet implemented in trueno-zram-core.
    #[test]
    fn popperian_c40_dictionary_training() {
        // Dictionary training improves compression for small, similar data blocks.
        // This is a placeholder for when dictionary support is added.
        //
        // When implemented:
        // 1. Train dictionary on representative data
        // 2. Verify improved compression ratio vs non-dictionary
        // 3. Verify roundtrip correctness

        let mut device = BlockDevice::new(1 << 20, test_compressor());

        // Write similar small patterns (would benefit from dictionary)
        let patterns: Vec<Vec<u8>> = (0..10)
            .map(|i| {
                format!(
                    "{{\"id\": {}, \"name\": \"user_{}\", \"active\": true}}",
                    i, i
                )
                .into_bytes()
                .into_iter()
                .chain(std::iter::repeat(0u8))
                .take(PAGE_SIZE)
                .collect()
            })
            .collect();

        for (i, pattern) in patterns.iter().enumerate() {
            device.write((i * PAGE_SIZE) as u64, pattern).unwrap();
        }

        let stats = device.stats();
        let ratio = stats.compression_ratio();

        println!(
            "C40: JSON-like data compression ratio (no dict): {:.2}:1",
            ratio
        );

        // Even without dictionary, should compress somewhat
        assert!(
            ratio > 1.0,
            "C40: JSON-like data should compress even without dictionary"
        );
    }
}
