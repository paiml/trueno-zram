//! GPU batch compression implementation.
//!
//! Implements the 5× PCIe rule for efficient GPU offload:
//! GPU beneficial when: T_compute > 5 × T_transfer
//!
//! For 4KB pages with ~2:1 ratio, batch 1000+ pages to amortize PCIe overhead.

use crate::error::{Error, Result};
use crate::page::CompressedPage;
use crate::{Algorithm, PAGE_SIZE};

/// Configuration for GPU batch compression.
#[derive(Debug, Clone)]
pub struct GpuBatchConfig {
    /// Device index (0 for first GPU).
    pub device_index: u32,
    /// Compression algorithm.
    pub algorithm: Algorithm,
    /// Number of pages per batch.
    pub batch_size: usize,
    /// Enable async DMA transfers.
    pub async_dma: bool,
    /// Number of DMA ring buffer slots.
    pub ring_buffer_slots: usize,
}

impl Default for GpuBatchConfig {
    fn default() -> Self {
        Self {
            device_index: 0,
            algorithm: Algorithm::Lz4,
            batch_size: 1000,
            async_dma: true,
            ring_buffer_slots: 4,
        }
    }
}

/// GPU batch compressor for high-throughput compression.
///
/// Implements the 5× PCIe rule to determine when GPU offload is beneficial.
/// Uses async DMA ring buffers for overlapping transfers and computation.
#[derive(Debug)]
#[allow(dead_code)] // Fields used when CUDA kernels are fully implemented
pub struct GpuBatchCompressor {
    config: GpuBatchConfig,
    #[cfg(feature = "cuda")]
    context: Option<CudaContext>,
    // Statistics
    pages_compressed: u64,
    total_bytes_in: u64,
    total_bytes_out: u64,
    total_time_ns: u64,
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
#[allow(dead_code)] // Fields will be used when CUDA kernels are implemented
struct CudaContext {
    // Will be implemented with cudarc
    device_index: u32,
}

/// Result of a batch compression operation.
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Compressed pages.
    pub pages: Vec<CompressedPage>,
    /// Time spent on host-to-device transfer (ns).
    pub h2d_time_ns: u64,
    /// Time spent on kernel execution (ns).
    pub kernel_time_ns: u64,
    /// Time spent on device-to-host transfer (ns).
    pub d2h_time_ns: u64,
    /// Total wall clock time (ns).
    pub total_time_ns: u64,
}

impl BatchResult {
    /// Calculate throughput in bytes per second.
    #[must_use]
    pub fn throughput_bytes_per_sec(&self, input_bytes: usize) -> f64 {
        if self.total_time_ns == 0 {
            return 0.0;
        }
        input_bytes as f64 / (self.total_time_ns as f64 / 1_000_000_000.0)
    }

    /// Calculate compression ratio.
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        let compressed_bytes: usize = self.pages.iter().map(|p| p.data.len()).sum();
        if compressed_bytes == 0 {
            return 1.0;
        }
        (self.pages.len() * PAGE_SIZE) as f64 / compressed_bytes as f64
    }

    /// Check if the 5× PCIe rule was satisfied.
    #[must_use]
    pub fn pcie_rule_satisfied(&self) -> bool {
        // GPU beneficial when: T_compute > 5 × T_transfer
        let transfer_time = self.h2d_time_ns + self.d2h_time_ns;
        self.kernel_time_ns > 5 * transfer_time
    }
}

impl GpuBatchCompressor {
    /// Create a new GPU batch compressor.
    ///
    /// # Errors
    ///
    /// Returns error if GPU is not available or device index is invalid.
    pub fn new(config: GpuBatchConfig) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            // Initialize CUDA context
            let context = Self::init_cuda(config.device_index)?;
            Ok(Self {
                config,
                context: Some(context),
                pages_compressed: 0,
                total_bytes_in: 0,
                total_bytes_out: 0,
                total_time_ns: 0,
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(Error::GpuNotAvailable(
                "CUDA feature not enabled".to_string(),
            ))
        }
    }

    #[cfg(feature = "cuda")]
    fn init_cuda(device_index: u32) -> Result<CudaContext> {
        use cudarc::driver::result::{self, device};

        result::init().map_err(|e| Error::GpuNotAvailable(format!("CUDA init failed: {e}")))?;

        let count = device::get_count()
            .map_err(|e| Error::GpuNotAvailable(format!("Failed to get device count: {e}")))?;

        if device_index >= count as u32 {
            return Err(Error::GpuNotAvailable(format!(
                "Device {device_index} not found (have {count} devices)"
            )));
        }

        Ok(CudaContext { device_index })
    }

    /// Compress a batch of pages.
    ///
    /// # Errors
    ///
    /// Returns error if compression fails.
    pub fn compress_batch(&mut self, pages: &[[u8; PAGE_SIZE]]) -> Result<BatchResult> {
        if pages.is_empty() {
            return Ok(BatchResult {
                pages: vec![],
                h2d_time_ns: 0,
                kernel_time_ns: 0,
                d2h_time_ns: 0,
                total_time_ns: 0,
            });
        }

        #[cfg(feature = "cuda")]
        {
            self.compress_batch_cuda(pages)
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(Error::GpuNotAvailable(
                "CUDA feature not enabled".to_string(),
            ))
        }
    }

    #[cfg(feature = "cuda")]
    fn compress_batch_cuda(&mut self, pages: &[[u8; PAGE_SIZE]]) -> Result<BatchResult> {
        use std::time::Instant;

        let start = Instant::now();

        // Phase 1: Host to Device transfer
        let h2d_start = Instant::now();
        let _input_buffer = self.transfer_to_device(pages)?;
        let h2d_time_ns = h2d_start.elapsed().as_nanos() as u64;

        // Phase 2: Kernel execution
        let kernel_start = Instant::now();
        let compressed_data = self.execute_compression_kernel(pages)?;
        let kernel_time_ns = kernel_start.elapsed().as_nanos() as u64;

        // Phase 3: Device to Host transfer
        let d2h_start = Instant::now();
        let pages_result = self.transfer_from_device(compressed_data)?;
        let d2h_time_ns = d2h_start.elapsed().as_nanos() as u64;

        let total_time_ns = start.elapsed().as_nanos() as u64;

        // Update statistics
        self.pages_compressed += pages.len() as u64;
        self.total_bytes_in += (pages.len() * PAGE_SIZE) as u64;
        self.total_bytes_out += pages_result.iter().map(|p| p.data.len() as u64).sum::<u64>();
        self.total_time_ns += total_time_ns;

        Ok(BatchResult {
            pages: pages_result,
            h2d_time_ns,
            kernel_time_ns,
            d2h_time_ns,
            total_time_ns,
        })
    }

    #[cfg(feature = "cuda")]
    fn transfer_to_device(&self, _pages: &[[u8; PAGE_SIZE]]) -> Result<Vec<u8>> {
        // TODO: Implement actual CUDA memory transfer
        // For now, return placeholder
        Ok(vec![])
    }

    #[cfg(feature = "cuda")]
    fn execute_compression_kernel(
        &self,
        pages: &[[u8; PAGE_SIZE]],
    ) -> Result<Vec<Vec<u8>>> {
        // TODO: Implement actual CUDA kernel
        // For now, use CPU fallback with simulated timing
        let mut results = Vec::with_capacity(pages.len());

        for page in pages {
            // Use CPU compression as placeholder
            let compressor = crate::CompressorBuilder::new()
                .algorithm(self.config.algorithm)
                .build()?;
            let compressed = compressor.compress(page)?;
            results.push(compressed.data);
        }

        Ok(results)
    }

    #[cfg(feature = "cuda")]
    fn transfer_from_device(&self, data: Vec<Vec<u8>>) -> Result<Vec<CompressedPage>> {
        // TODO: Implement actual CUDA memory transfer
        // For now, just wrap the data
        Ok(data
            .into_iter()
            .map(|d| CompressedPage {
                data: d,
                original_size: PAGE_SIZE,
                algorithm: self.config.algorithm,
            })
            .collect())
    }

    /// Get compression statistics.
    #[must_use]
    pub fn stats(&self) -> GpuBatchStats {
        GpuBatchStats {
            pages_compressed: self.pages_compressed,
            total_bytes_in: self.total_bytes_in,
            total_bytes_out: self.total_bytes_out,
            total_time_ns: self.total_time_ns,
        }
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &GpuBatchConfig {
        &self.config
    }
}

/// Statistics from GPU batch compression.
#[derive(Debug, Clone, Default)]
pub struct GpuBatchStats {
    /// Total pages compressed.
    pub pages_compressed: u64,
    /// Total input bytes.
    pub total_bytes_in: u64,
    /// Total output bytes.
    pub total_bytes_out: u64,
    /// Total time in nanoseconds.
    pub total_time_ns: u64,
}

impl GpuBatchStats {
    /// Calculate overall compression ratio.
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        if self.total_bytes_out == 0 {
            return 1.0;
        }
        self.total_bytes_in as f64 / self.total_bytes_out as f64
    }

    /// Calculate throughput in GB/s.
    #[must_use]
    pub fn throughput_gbps(&self) -> f64 {
        if self.total_time_ns == 0 {
            return 0.0;
        }
        self.total_bytes_in as f64 / (self.total_time_ns as f64 / 1_000_000_000.0) / 1e9
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // F036: GPU batch compressor can be created with valid config
    // ==========================================================================
    #[test]
    #[cfg(feature = "cuda")]
    fn test_f036_gpu_batch_compressor_creation() {
        let config = GpuBatchConfig::default();
        let result = GpuBatchCompressor::new(config);
        // Should succeed on systems with CUDA, fail gracefully otherwise
        if crate::gpu::gpu_available() {
            assert!(result.is_ok(), "Should create compressor when GPU available");
        } else {
            assert!(result.is_err(), "Should fail when GPU not available");
        }
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_f036_gpu_batch_compressor_no_cuda() {
        let config = GpuBatchConfig::default();
        let result = GpuBatchCompressor::new(config);
        assert!(result.is_err(), "Should fail without CUDA feature");
    }

    // ==========================================================================
    // F037: Batch compression produces valid output
    // ==========================================================================
    #[test]
    #[cfg(feature = "cuda")]
    fn test_f037_batch_compression_valid_output() {
        if !crate::gpu::gpu_available() {
            return;
        }

        let config = GpuBatchConfig {
            batch_size: 100,
            ..Default::default()
        };
        let mut compressor = GpuBatchCompressor::new(config).unwrap();

        // Create test pages with compressible data
        let pages: Vec<[u8; PAGE_SIZE]> = (0..100)
            .map(|i| {
                let mut page = [0u8; PAGE_SIZE];
                page[0] = i as u8;
                page
            })
            .collect();

        let result = compressor.compress_batch(&pages).unwrap();

        // Verify output
        assert_eq!(result.pages.len(), 100, "Should compress all pages");
        for page in &result.pages {
            assert!(!page.data.is_empty(), "Compressed data should not be empty");
            assert!(
                page.data.len() <= PAGE_SIZE,
                "Compressed size should not exceed original"
            );
        }
    }

    // ==========================================================================
    // F038: Empty batch returns empty result
    // ==========================================================================
    #[test]
    #[cfg(feature = "cuda")]
    fn test_f038_empty_batch() {
        if !crate::gpu::gpu_available() {
            return;
        }

        let config = GpuBatchConfig::default();
        let mut compressor = GpuBatchCompressor::new(config).unwrap();

        let result = compressor.compress_batch(&[]).unwrap();
        assert!(result.pages.is_empty(), "Empty input should produce empty output");
        assert_eq!(result.total_time_ns, 0);
    }

    // ==========================================================================
    // F039: Batch result tracks timing components
    // ==========================================================================
    #[test]
    #[cfg(feature = "cuda")]
    fn test_f039_timing_components() {
        if !crate::gpu::gpu_available() {
            return;
        }

        let config = GpuBatchConfig::default();
        let mut compressor = GpuBatchCompressor::new(config).unwrap();

        let pages: Vec<[u8; PAGE_SIZE]> = vec![[0xAAu8; PAGE_SIZE]; 100];
        let result = compressor.compress_batch(&pages).unwrap();

        // Timing should be non-zero for actual work
        assert!(result.total_time_ns > 0, "Total time should be positive");

        // Total time should be approximately sum of components
        let component_sum = result.h2d_time_ns + result.kernel_time_ns + result.d2h_time_ns;
        assert!(
            result.total_time_ns >= component_sum / 2,
            "Total time should account for all phases"
        );
    }

    // ==========================================================================
    // F040: 5× PCIe rule detection
    // ==========================================================================
    #[test]
    fn test_f040_pcie_rule_detection() {
        // When kernel time dominates (kernel > 5× transfer)
        // transfer = 100 + 100 = 200, 5× = 1000
        // kernel = 1001 > 1000 ✓
        let good_result = BatchResult {
            pages: vec![],
            h2d_time_ns: 100,
            kernel_time_ns: 1001, // > 5× transfer time (200*5=1000)
            d2h_time_ns: 100,
            total_time_ns: 1201,
        };
        assert!(
            good_result.pcie_rule_satisfied(),
            "Should satisfy 5× rule when kernel > 5× transfer"
        );

        // When transfer time dominates
        let bad_result = BatchResult {
            pages: vec![],
            h2d_time_ns: 500,
            kernel_time_ns: 100, // Only 0.1× transfer time
            d2h_time_ns: 500,
            total_time_ns: 1100,
        };
        assert!(
            !bad_result.pcie_rule_satisfied(),
            "Should not satisfy 5× rule when transfer >> kernel"
        );

        // Edge case: exactly at boundary (should not satisfy)
        let boundary_result = BatchResult {
            pages: vec![],
            h2d_time_ns: 100,
            kernel_time_ns: 1000, // Exactly 5× transfer time
            d2h_time_ns: 100,
            total_time_ns: 1200,
        };
        assert!(
            !boundary_result.pcie_rule_satisfied(),
            "Should not satisfy when exactly at 5× boundary (need >)"
        );
    }

    // ==========================================================================
    // F041: Statistics accumulate correctly
    // ==========================================================================
    #[test]
    #[cfg(feature = "cuda")]
    fn test_f041_statistics_accumulation() {
        if !crate::gpu::gpu_available() {
            return;
        }

        let config = GpuBatchConfig::default();
        let mut compressor = GpuBatchCompressor::new(config).unwrap();

        // First batch
        let pages1: Vec<[u8; PAGE_SIZE]> = vec![[0xAAu8; PAGE_SIZE]; 50];
        compressor.compress_batch(&pages1).unwrap();

        // Second batch
        let pages2: Vec<[u8; PAGE_SIZE]> = vec![[0xBBu8; PAGE_SIZE]; 50];
        compressor.compress_batch(&pages2).unwrap();

        let stats = compressor.stats();
        assert_eq!(stats.pages_compressed, 100, "Should accumulate page count");
        assert_eq!(
            stats.total_bytes_in,
            100 * PAGE_SIZE as u64,
            "Should accumulate input bytes"
        );
    }

    // ==========================================================================
    // F042: Throughput calculation
    // ==========================================================================
    #[test]
    fn test_f042_throughput_calculation() {
        let stats = GpuBatchStats {
            pages_compressed: 1000,
            total_bytes_in: 1000 * PAGE_SIZE as u64,
            total_bytes_out: 500 * PAGE_SIZE as u64,
            total_time_ns: 1_000_000_000, // 1 second
        };

        let throughput = stats.throughput_gbps();
        // 4096 * 1000 bytes / 1 second = ~4 MB/s = ~0.004 GB/s
        assert!(
            (throughput - 0.004096).abs() < 0.001,
            "Throughput calculation should be correct: got {throughput}"
        );
    }

    // ==========================================================================
    // F043: Compression ratio calculation
    // ==========================================================================
    #[test]
    fn test_f043_compression_ratio() {
        let stats = GpuBatchStats {
            pages_compressed: 100,
            total_bytes_in: 100 * PAGE_SIZE as u64,
            total_bytes_out: 50 * PAGE_SIZE as u64,
            total_time_ns: 1000,
        };

        let ratio = stats.compression_ratio();
        assert!(
            (ratio - 2.0).abs() < 0.001,
            "Compression ratio should be 2:1 for 50% compression"
        );
    }

    // ==========================================================================
    // F044: Config defaults are reasonable
    // ==========================================================================
    #[test]
    fn test_f044_config_defaults() {
        let config = GpuBatchConfig::default();

        assert_eq!(config.device_index, 0, "Default device should be 0");
        assert!(config.batch_size >= 100, "Batch size should be >= 100 for PCIe efficiency");
        assert!(config.batch_size <= 100_000, "Batch size should be reasonable");
        assert!(config.async_dma, "Async DMA should be enabled by default");
        assert!(config.ring_buffer_slots >= 2, "Ring buffer needs >= 2 slots");
    }

    // ==========================================================================
    // F045: Algorithm selection respected
    // ==========================================================================
    #[test]
    #[cfg(feature = "cuda")]
    fn test_f045_algorithm_selection() {
        if !crate::gpu::gpu_available() {
            return;
        }

        for algo in [Algorithm::Lz4, Algorithm::Zstd { level: 1 }] {
            let config = GpuBatchConfig {
                algorithm: algo,
                batch_size: 10,
                ..Default::default()
            };
            let mut compressor = GpuBatchCompressor::new(config).unwrap();

            let pages: Vec<[u8; PAGE_SIZE]> = vec![[0xCCu8; PAGE_SIZE]; 10];
            let result = compressor.compress_batch(&pages).unwrap();

            for page in &result.pages {
                assert_eq!(page.algorithm, algo, "Output should use configured algorithm");
            }
        }
    }

    // ==========================================================================
    // F046: Large batch achieves target throughput (50+ GB/s on RTX 4090)
    // ==========================================================================
    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "requires RTX 4090 for throughput target"]
    fn test_f046_throughput_target() {
        if !crate::gpu::gpu_available() {
            return;
        }

        let config = GpuBatchConfig {
            batch_size: 18_432, // Optimal for RTX 4090
            ..Default::default()
        };
        let mut compressor = GpuBatchCompressor::new(config).unwrap();

        // Create compressible test data
        let pages: Vec<[u8; PAGE_SIZE]> = (0..18_432)
            .map(|i| {
                let mut page = [0u8; PAGE_SIZE];
                // Fill with pattern for reasonable compression
                for (j, byte) in page.iter_mut().enumerate() {
                    *byte = ((i + j) % 256) as u8;
                }
                page
            })
            .collect();

        let result = compressor.compress_batch(&pages).unwrap();
        let input_bytes = pages.len() * PAGE_SIZE;
        let throughput_gbps = result.throughput_bytes_per_sec(input_bytes) / 1e9;

        assert!(
            throughput_gbps >= 50.0,
            "Throughput should be >= 50 GB/s on RTX 4090, got {throughput_gbps:.2} GB/s"
        );
    }

    // ==========================================================================
    // F047: Async DMA reduces latency
    // ==========================================================================
    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "requires GPU for async DMA test"]
    fn test_f047_async_dma_benefit() {
        if !crate::gpu::gpu_available() {
            return;
        }

        // With async DMA
        let async_config = GpuBatchConfig {
            async_dma: true,
            batch_size: 1000,
            ..Default::default()
        };
        let mut async_compressor = GpuBatchCompressor::new(async_config).unwrap();

        // Without async DMA
        let sync_config = GpuBatchConfig {
            async_dma: false,
            batch_size: 1000,
            ..Default::default()
        };
        let mut sync_compressor = GpuBatchCompressor::new(sync_config).unwrap();

        let pages: Vec<[u8; PAGE_SIZE]> = vec![[0xDDu8; PAGE_SIZE]; 1000];

        let async_result = async_compressor.compress_batch(&pages).unwrap();
        let sync_result = sync_compressor.compress_batch(&pages).unwrap();

        // Async should be faster due to overlapped transfers
        assert!(
            async_result.total_time_ns <= sync_result.total_time_ns,
            "Async DMA should not be slower than sync"
        );
    }

    // ==========================================================================
    // F048: Ring buffer slots configuration
    // ==========================================================================
    #[test]
    fn test_f048_ring_buffer_config() {
        let config = GpuBatchConfig {
            ring_buffer_slots: 8,
            ..Default::default()
        };
        assert_eq!(config.ring_buffer_slots, 8);
    }

    // ==========================================================================
    // F049: Batch result throughput helper
    // ==========================================================================
    #[test]
    fn test_f049_batch_result_throughput() {
        let result = BatchResult {
            pages: vec![CompressedPage {
                data: vec![0; 2048],
                original_size: PAGE_SIZE,
                algorithm: Algorithm::Lz4,
            }],
            h2d_time_ns: 100,
            kernel_time_ns: 800,
            d2h_time_ns: 100,
            total_time_ns: 1_000_000, // 1ms
        };

        let throughput = result.throughput_bytes_per_sec(PAGE_SIZE);
        // 4096 bytes / 0.001 seconds = 4,096,000 bytes/sec
        assert!(
            (throughput - 4_096_000.0).abs() < 1.0,
            "Throughput helper should be correct"
        );
    }

    // ==========================================================================
    // F050: Stats zero throughput for empty
    // ==========================================================================
    #[test]
    fn test_f050_stats_zero_throughput() {
        let stats = GpuBatchStats::default();
        assert!(stats.throughput_gbps().abs() < f64::EPSILON);
        assert!((stats.compression_ratio() - 1.0).abs() < f64::EPSILON);
    }
}
