//! Device module - ublk device management
//!
//! Provides abstraction over libublk for managing trueno-ublk devices.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use trueno_zram_core::{CompressorBuilder, PageCompressor};

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
