//! Device module - ublk device management
//!
//! Provides abstraction over libublk for managing trueno-ublk devices.
//! Also provides a pure Rust `BlockDevice` for testing without kernel dependencies.

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
    pub size: u64,
    pub algorithm: trueno_zram_core::Algorithm,
    pub streams: usize,
    pub gpu_enabled: bool,
    pub mem_limit: Option<u64>,
    pub backing_dev: Option<PathBuf>,
    pub writeback_limit: Option<u64>,
    pub entropy_skip_threshold: f64,
    pub gpu_batch_size: usize,
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
    /// Create a new device with the given configuration
    pub fn create(config: DeviceConfig) -> Result<Self> {
        let id = Self::next_free_id()?;

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
            let mut devices = DEVICES.write().unwrap();
            devices.insert(id, inner.clone());
        }

        // Start ublk daemon in background
        Self::start_ublk_daemon(id)?;

        Ok(Self { inner })
    }

    /// Open an existing device by path
    pub fn open(path: &Path) -> Result<Self> {
        let id = Self::parse_device_id(path)?;

        let devices = DEVICES.read().unwrap();
        let inner = devices
            .get(&id)
            .cloned()
            .context(format!("Device {} not found", path.display()))?;

        Ok(Self { inner })
    }

    /// List all trueno-ublk devices
    pub fn list_all() -> Result<Vec<Self>> {
        let devices = DEVICES.read().unwrap();
        Ok(devices
            .values()
            .map(|inner| Self {
                inner: inner.clone(),
            })
            .collect())
    }

    /// Find the next free device ID
    pub fn next_free_id() -> Result<u32> {
        let devices = DEVICES.read().unwrap();
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
        self.inner.config.read().unwrap().clone()
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

        let mut devices = DEVICES.write().unwrap();
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
        let mut config = self.inner.config.write().unwrap();
        config.mem_limit = Some(limit);
        Ok(())
    }

    /// Set GPU enabled/disabled
    pub fn set_gpu_enabled(&self, enabled: bool) -> Result<()> {
        let mut config = self.inner.config.write().unwrap();
        config.gpu_enabled = enabled;

        // Reinitialize compressor with new settings
        let compressor = CompressorBuilder::new()
            .algorithm(config.algorithm)
            .build()?;
        *self.inner.compressor.write().unwrap() = Some(compressor);

        Ok(())
    }

    /// Set entropy skip threshold
    pub fn set_entropy_threshold(&self, threshold: f64) -> Result<()> {
        let mut config = self.inner.config.write().unwrap();
        config.entropy_skip_threshold = threshold;
        Ok(())
    }

    /// Set writeback limit
    pub fn set_writeback_limit(&self, limit: u64) -> Result<()> {
        let mut config = self.inner.config.write().unwrap();
        config.writeback_limit = Some(limit);
        Ok(())
    }

    /// Enable/disable writeback limit
    pub fn set_writeback_limit_enabled(&self, enabled: bool) -> Result<()> {
        let mut config = self.inner.config.write().unwrap();
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
            id_str
                .parse()
                .context("Invalid device ID in path")
        } else {
            anyhow::bail!("Not a ublk device: {}", path.display())
        }
    }

    fn start_ublk_daemon(_id: u32) -> Result<()> {
        // TODO: Actually start the ublk daemon using libublk
        // This would involve:
        // 1. Creating a ublk control device
        // 2. Setting up the block device parameters
        // 3. Starting IO processing threads
        Ok(())
    }

    fn stop_ublk_daemon(_id: u32) -> Result<()> {
        // TODO: Stop the ublk daemon
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
            store: PageStore::new(Arc::from(compressor), 7.5), // Default entropy threshold
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
            store: PageStore::new(Arc::from(compressor), entropy_threshold),
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

        // Write highly compressible data (all same value)
        let repetitive = vec![0xAA; PAGE_SIZE];
        device.write(0, &repetitive).unwrap();

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
        let mut device =
            BlockDevice::with_entropy_threshold(1 << 20, test_compressor(), 7.0);

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
        let mut device =
            BlockDevice::with_entropy_threshold(1 << 20, test_compressor(), 7.0);

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
        assert!(buf.iter().all(|&b| b == 0), "Discarded page should read as zeros");
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

        assert!(buf.iter().all(|&b| b == 0), "Unwritten page should read as zeros");
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
            (0..PAGE_SIZE).map(|i| if i % 2 == 0 { 0xAA } else { 0x55 }).collect(),
            // Sequential bytes
            (0..PAGE_SIZE).map(|i| (i % 256) as u8).collect(),
            // Repeating short pattern
            (0..PAGE_SIZE).map(|i| (i % 16) as u8).collect(),
            // Text-like data
            "The quick brown fox jumps over the lazy dog. ".repeat(100).into_bytes()[..PAGE_SIZE].to_vec(),
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

        assert!((stats.compression_ratio() - 4.0).abs() < 0.001, "4:1 compression ratio expected");
    }

    #[test]
    fn test_block_device_stats_compression_ratio_no_data() {
        let stats = BlockDeviceStats::default();
        assert!((stats.compression_ratio() - 1.0).abs() < 0.001, "Default ratio should be 1.0");
    }
}
