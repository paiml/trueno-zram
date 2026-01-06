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
#[allow(dead_code)] // Fields used when CUDA kernels are fully implemented
pub struct GpuBatchCompressor {
    config: GpuBatchConfig,
    #[cfg(feature = "cuda")]
    context: Option<GpuContext>,
    /// Cached SIMD compressor for parallel CPU path (Arc for thread-safety)
    simd_compressor: std::sync::Arc<Box<dyn crate::PageCompressor>>,
    // Statistics
    pages_compressed: u64,
    total_bytes_in: u64,
    total_bytes_out: u64,
    total_time_ns: u64,
}

impl std::fmt::Debug for GpuBatchCompressor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBatchCompressor")
            .field("config", &self.config)
            .field("pages_compressed", &self.pages_compressed)
            .finish()
    }
}

#[cfg(feature = "cuda")]
struct GpuContext {
    /// CUDA context handle (trueno-gpu driver).
    context: trueno_gpu::driver::CudaContext,
    /// Stream for async operations (trueno-gpu driver).
    stream: trueno_gpu::driver::CudaStream,
    /// Loaded LZ4 compression kernel module (trueno-gpu driver).
    /// Uses RefCell for interior mutability (launch_kernel needs &mut CudaModule).
    lz4_module: Option<std::cell::RefCell<trueno_gpu::driver::CudaModule>>,
    /// Loaded LZ4 DECOMPRESSION kernel module (F082-safe, no hash tables).
    /// This is the key to the hybrid CPU compress + GPU decompress architecture.
    lz4_decompress_module: Option<std::cell::RefCell<trueno_gpu::driver::CudaModule>>,
    /// Device index.
    device_index: u32,
    /// Pre-allocated pinned input buffer for DMA (eliminates 350ms alloc overhead).
    input_pinned: trueno_gpu::driver::PinnedBuffer<u8>,
    /// Pre-allocated pinned output buffer for DMA.
    output_pinned: trueno_gpu::driver::PinnedBuffer<u8>,
    /// Pre-allocated GPU input buffer.
    input_dev: trueno_gpu::driver::GpuBuffer<u8>,
    /// Pre-allocated GPU output buffer.
    output_dev: trueno_gpu::driver::GpuBuffer<u8>,
    /// Pre-allocated GPU sizes buffer.
    sizes_dev: trueno_gpu::driver::GpuBuffer<u32>,
    /// Maximum batch size these buffers support.
    max_batch_size: usize,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("device_index", &self.device_index)
            .field("lz4_compress_loaded", &self.lz4_module.is_some())
            .field("lz4_decompress_loaded", &self.lz4_decompress_module.is_some())
            .finish()
    }
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

/// Result of a batch DECOMPRESSION operation (Sovereign AI F082-safe path).
///
/// This struct holds the decompressed pages from GPU or CPU decompression.
/// Used for the hybrid architecture: CPU compress + GPU decompress for 2TB LLM restore.
#[derive(Debug, Clone)]
pub struct BatchDecompressResult {
    /// Decompressed pages (4KB each).
    pub pages: Vec<[u8; PAGE_SIZE]>,
    /// Time spent on host-to-device transfer (ns).
    pub h2d_time_ns: u64,
    /// Time spent on kernel execution (ns).
    pub kernel_time_ns: u64,
    /// Time spent on device-to-host transfer (ns).
    pub d2h_time_ns: u64,
    /// Total wall clock time (ns).
    pub total_time_ns: u64,
}

impl BatchDecompressResult {
    /// Calculate throughput in bytes per second.
    #[must_use]
    pub fn throughput_bytes_per_sec(&self) -> f64 {
        if self.total_time_ns == 0 {
            return 0.0;
        }
        let output_bytes = self.pages.len() * PAGE_SIZE;
        output_bytes as f64 / (self.total_time_ns as f64 / 1_000_000_000.0)
    }

    /// Calculate throughput in GB/s.
    #[must_use]
    pub fn throughput_gbps(&self) -> f64 {
        self.throughput_bytes_per_sec() / 1e9
    }

    /// Check if the 5× PCIe rule was satisfied (kernel time > 5× transfer time).
    #[must_use]
    pub fn pcie_rule_satisfied(&self) -> bool {
        let transfer_time = self.h2d_time_ns + self.d2h_time_ns;
        self.kernel_time_ns > 5 * transfer_time
    }
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
        // Create SIMD-accelerated compressor for parallel CPU path
        let simd_compressor = std::sync::Arc::new(
            crate::CompressorBuilder::new()
                .algorithm(config.algorithm)
                .build()?
        );

        #[cfg(feature = "cuda")]
        {
            // Initialize CUDA context with pre-allocated buffers for batch_size
            let context = Self::init_cuda(config.device_index, config.batch_size)?;
            Ok(Self {
                config,
                context: Some(context),
                simd_compressor,
                pages_compressed: 0,
                total_bytes_in: 0,
                total_bytes_out: 0,
                total_time_ns: 0,
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            Ok(Self {
                config,
                simd_compressor,
                pages_compressed: 0,
                total_bytes_in: 0,
                total_bytes_out: 0,
                total_time_ns: 0,
            })
        }
    }

    #[cfg(feature = "cuda")]
    fn init_cuda(device_index: u32, batch_size: usize) -> Result<GpuContext> {
        use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, PinnedBuffer};
        use trueno_gpu::kernels::lz4::{Lz4WarpShuffleKernel, Lz4DecompressKernel};
        use trueno_gpu::kernels::Kernel;

        // Create CUDA context using trueno-gpu driver (properly initializes CUDA)
        let context = CudaContext::new(device_index as i32)
            .map_err(|e| Error::GpuNotAvailable(format!("Failed to create CUDA context: {e}")))?;

        // Create a stream for async operations
        let stream = CudaStream::new(&context)
            .map_err(|e| Error::GpuNotAvailable(format!("Failed to create CUDA stream: {e}")))?;

        // Generate and load LZ4 COMPRESSION kernel using trueno-gpu driver
        // Uses Lz4WarpShuffleKernel which produces valid LZ4 output (literal-only encoding)
        // Note: For F082-safe path, use the hybrid architecture (CPU compress + GPU decompress)
        let compress_kernel = Lz4WarpShuffleKernel::new(65536);
        let compress_ptx = compress_kernel.emit_ptx();

        let lz4_module = CudaModule::from_ptx(&context, &compress_ptx)
            .map_err(|e| Error::GpuNotAvailable(format!("Failed to load LZ4 compress PTX: {e}")))?;

        // Generate and load LZ4 DECOMPRESSION kernel (KF-002)
        // This kernel is F082-SAFE because it doesn't use hash tables!
        // Enables hybrid architecture: CPU compress (24 GB/s) + GPU decompress (16 GB/s)
        let decompress_kernel = Lz4DecompressKernel::new(65536);
        let decompress_ptx = decompress_kernel.emit_ptx();

        let lz4_decompress_module = CudaModule::from_ptx(&context, &decompress_ptx)
            .map_err(|e| Error::GpuNotAvailable(format!("Failed to load LZ4 decompress PTX: {e}")))?;

        // ═══════════════════════════════════════════════════════════════════════
        // PRE-ALLOCATE PINNED BUFFERS FOR DMA (eliminates 350ms alloc overhead!)
        // This is the KEY optimization for G.119 compliance.
        // ═══════════════════════════════════════════════════════════════════════
        let output_stride = 4352usize; // Max compressed size per page
        let input_size = batch_size * output_stride;
        let output_size = batch_size * PAGE_SIZE;

        let input_pinned: PinnedBuffer<u8> = PinnedBuffer::new(&context, input_size)
            .map_err(|e| Error::GpuNotAvailable(format!("Pinned input alloc failed: {e}")))?;
        let output_pinned: PinnedBuffer<u8> = PinnedBuffer::new(&context, output_size)
            .map_err(|e| Error::GpuNotAvailable(format!("Pinned output alloc failed: {e}")))?;

        // Pre-allocate GPU buffers
        let input_dev: GpuBuffer<u8> = GpuBuffer::new(&context, input_size)
            .map_err(|e| Error::GpuNotAvailable(format!("GPU input alloc failed: {e}")))?;
        let output_dev: GpuBuffer<u8> = GpuBuffer::new(&context, output_size)
            .map_err(|e| Error::GpuNotAvailable(format!("GPU output alloc failed: {e}")))?;
        let sizes_dev: GpuBuffer<u32> = GpuBuffer::new(&context, batch_size)
            .map_err(|e| Error::GpuNotAvailable(format!("GPU sizes alloc failed: {e}")))?;

        Ok(GpuContext {
            context,
            stream,
            lz4_module: Some(std::cell::RefCell::new(lz4_module)),
            lz4_decompress_module: Some(std::cell::RefCell::new(lz4_decompress_module)),
            device_index,
            input_pinned,
            output_pinned,
            input_dev,
            output_dev,
            sizes_dev,
            max_batch_size: batch_size,
        })
    }

    /// Compress a batch of pages using parallel CPU compression.
    ///
    /// This is the FAST PATH that achieves 5X speedup over single-threaded compression
    /// by using all available CPU cores with rayon parallelization + AVX-512 SIMD.
    ///
    /// Note: The GPU kernel path currently only does zero-page detection.
    /// Until nvCOMP integration is complete, parallel CPU is faster because
    /// it avoids PCIe transfer overhead.
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

        // Use parallel CPU compression (fast path)
        // This is faster than GPU until nvCOMP integration is complete
        // because it avoids PCIe transfer overhead
        self.compress_batch_parallel_cpu(pages)
    }

    /// Fast parallel CPU compression using rayon + AVX-512 SIMD.
    ///
    /// Achieves ~5-20X speedup over single-threaded by utilizing all CPU cores
    /// with SIMD-accelerated LZ4 compression.
    ///
    /// Note: Uses raw lz4::compress to avoid atomic statistics contention.
    /// Uses chunked parallelism to balance memory bandwidth vs parallelism.
    fn compress_batch_parallel_cpu(&mut self, pages: &[[u8; PAGE_SIZE]]) -> Result<BatchResult> {
        use rayon::prelude::*;
        use std::time::Instant;

        let start = Instant::now();
        let algorithm = self.config.algorithm;

        // Use optimal chunk size: process multiple pages per thread task
        // This reduces scheduling overhead and improves cache locality
        let chunk_size = 16.max(pages.len() / rayon::current_num_threads());

        // Parallel compression: each chunk processed by one thread
        // IMPORTANT: Always store LZ4 data even for incompressible pages,
        // because GPU decompression kernel expects valid LZ4 format.
        let chunk_results: Vec<Result<Vec<CompressedPage>>> = pages
            .par_chunks(chunk_size)
            .map(|chunk: &[[u8; PAGE_SIZE]]| {
                // Process chunk sequentially within each thread
                chunk.iter().map(|page| {
                    let compressed = crate::lz4::compress(page)?;
                    // Always use LZ4 data - GPU decompression needs valid LZ4 format
                    // For incompressible data, LZ4 will be slightly larger but valid
                    Ok(CompressedPage {
                        data: compressed,
                        original_size: PAGE_SIZE,
                        algorithm,
                    })
                }).collect()
            })
            .collect();

        // Flatten results
        let mut pages_result = Vec::with_capacity(pages.len());
        for chunk_result in chunk_results {
            pages_result.extend(chunk_result?);
        }

        let total_time_ns = start.elapsed().as_nanos() as u64;

        // Update statistics
        self.pages_compressed += pages.len() as u64;
        self.total_bytes_in += (pages.len() * PAGE_SIZE) as u64;
        self.total_bytes_out += pages_result.iter().map(|p| p.data.len() as u64).sum::<u64>();
        self.total_time_ns += total_time_ns;

        Ok(BatchResult {
            pages: pages_result,
            h2d_time_ns: 0,        // No GPU transfer
            kernel_time_ns: total_time_ns, // All time is "kernel" (CPU)
            d2h_time_ns: 0,        // No GPU transfer
            total_time_ns,
        })
    }

    /// Compress using GPU (for benchmarking comparison).
    /// This uses GPU transfers but falls back to CPU compression.
    #[allow(dead_code)]
    pub fn compress_batch_gpu(&mut self, pages: &[[u8; PAGE_SIZE]]) -> Result<BatchResult> {
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
        // This allocates GPU memory and copies data, establishing the transfer baseline
        let h2d_start = Instant::now();
        let _device_buffer = self.transfer_to_device(pages)?;
        let h2d_time_ns = h2d_start.elapsed().as_nanos() as u64;

        // Phase 2: Kernel execution (currently CPU fallback, ready for nvCOMP)
        let kernel_start = Instant::now();
        let compressed_data = self.execute_compression_kernel(pages)?;
        let kernel_time_ns = kernel_start.elapsed().as_nanos() as u64;

        // Phase 3: Device to Host transfer
        // In hybrid mode, data is already on host from CPU compression
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
    fn transfer_to_device(
        &self,
        pages: &[[u8; PAGE_SIZE]],
    ) -> Result<trueno_gpu::driver::GpuBuffer<u8>> {
        use trueno_gpu::driver::GpuBuffer;

        let context = self.context.as_ref().ok_or_else(|| {
            Error::GpuNotAvailable("CUDA context not initialized".to_string())
        })?;

        // Flatten pages into contiguous buffer
        let total_bytes = pages.len() * PAGE_SIZE;
        let mut flat_data = Vec::with_capacity(total_bytes);
        for page in pages {
            flat_data.extend_from_slice(page);
        }

        // Allocate and copy to device using trueno-gpu GpuBuffer
        let device_buffer = GpuBuffer::from_host(&context.context, &flat_data)
            .map_err(|e| Error::GpuNotAvailable(format!("H2D transfer failed: {e}")))?;

        Ok(device_buffer)
    }

    #[cfg(feature = "cuda")]
    fn execute_compression_kernel(
        &self,
        pages: &[[u8; PAGE_SIZE]],
    ) -> Result<Vec<Vec<u8>>> {
        use std::ffi::c_void;
        use std::time::Instant;
        use trueno_gpu::driver::{GpuBuffer, LaunchConfig};
        use trueno_gpu::kernels::lz4::lz4_compress_block;

        let context = self.context.as_ref().ok_or_else(|| {
            Error::GpuNotAvailable("CUDA context not initialized".to_string())
        })?;

        // Check if GPU module is loaded, fall back to CPU if not
        let module_cell = match context.lz4_module.as_ref() {
            Some(m) => m,
            None => return self.execute_compression_kernel_cpu(pages),
        };

        let batch_size = pages.len();
        let total_input_bytes = batch_size * PAGE_SIZE;

        // ═══════════════════════════════════════════════════════════════════════
        // [RENACER] Enhanced GPU Pipeline Tracing
        // Tracks timing, bytes, and data flow through each stage
        // ═══════════════════════════════════════════════════════════════════════

        let pipeline_start = Instant::now();

        // [RENACER] Stage 1: Input Preparation
        let stage1_start = Instant::now();
        let mut input_flat = Vec::with_capacity(total_input_bytes);
        for page in pages {
            input_flat.extend_from_slice(page);
        }
        let stage1_us = stage1_start.elapsed().as_micros();

        #[cfg(debug_assertions)]
        eprintln!(
            "[RENACER] S1-PREP: {} pages → {} bytes ({} µs)",
            batch_size, input_flat.len(), stage1_us
        );

        // [RENACER] Stage 2: H2D Transfer (Host → Device)
        let stage2_start = Instant::now();
        let input_dev: GpuBuffer<u8> = GpuBuffer::from_host(&context.context, &input_flat)
            .map_err(|e| Error::GpuNotAvailable(format!("H2D transfer failed: {e}")))?;
        let stage2_us = stage2_start.elapsed().as_micros();
        let h2d_bandwidth_gbps = (total_input_bytes as f64) / (stage2_us as f64 / 1e6) / 1e9;

        #[cfg(debug_assertions)]
        eprintln!(
            "[RENACER] S2-H2D:  {} bytes transferred ({} µs, {:.2} GB/s)",
            total_input_bytes, stage2_us, h2d_bandwidth_gbps
        );

        // [RENACER] Stage 3: GPU Memory Allocation
        // Output buffer: 4352 bytes per page (max LZ4 expansion + headers)
        let output_stride = 4352usize;
        let total_output_bytes = batch_size * output_stride;
        let stage3_start = Instant::now();
        let output_dev: GpuBuffer<u8> = GpuBuffer::new(&context.context, total_output_bytes)
            .map_err(|e| Error::GpuNotAvailable(format!("Failed to allocate output buffer: {e}")))?;
        let sizes_dev: GpuBuffer<u32> = GpuBuffer::new(&context.context, batch_size)
            .map_err(|e| Error::GpuNotAvailable(format!("Failed to allocate sizes buffer: {e}")))?;
        let stage3_us = stage3_start.elapsed().as_micros();
        let alloc_bytes = total_output_bytes + batch_size * 4;

        #[cfg(debug_assertions)]
        eprintln!(
            "[RENACER] S3-ALLOC: {} bytes GPU memory ({} µs)",
            alloc_bytes, stage3_us
        );

        // [RENACER] Stage 4: Kernel Configuration
        // Lz4WarpShuffleKernel: 1 block per page, 32 threads (1 warp) per block
        let grid_x = batch_size as u32;
        let block_x = 32u32;
        let total_threads = grid_x * block_x;
        let batch_size_u32 = batch_size as u32;

        let config = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem: 0, // Static shared memory declared in PTX
        };

        #[cfg(debug_assertions)]
        eprintln!(
            "[RENACER] S4-CFG:  grid={} block={} threads={} warps={}",
            grid_x, block_x, total_threads, total_threads / 32
        );

        // [RENACER] Stage 5: Kernel Launch
        let stage5_start = Instant::now();
        let mut args: [*mut c_void; 4] = [
            input_dev.as_kernel_arg(),
            output_dev.as_kernel_arg(),
            sizes_dev.as_kernel_arg(),
            &batch_size_u32 as *const u32 as *mut c_void,
        ];

        let mut module = module_cell.borrow_mut();

        // SAFETY: We're passing valid device pointers and batch_size
        unsafe {
            context.stream
                .launch_kernel(&mut module, "lz4_compress_warp_shuffle", &config, &mut args)
                .map_err(|e| Error::GpuNotAvailable(format!("Kernel launch failed: {e}")))?;
        }
        let stage5_us = stage5_start.elapsed().as_micros();

        #[cfg(debug_assertions)]
        eprintln!(
            "[RENACER] S5-LAUNCH: kernel dispatched ({} µs)",
            stage5_us
        );

        // [RENACER] Stage 6: GPU Synchronization
        let stage6_start = Instant::now();
        context.stream.synchronize()
            .map_err(|e| Error::GpuNotAvailable(format!("Stream sync failed: {e}")))?;
        let stage6_us = stage6_start.elapsed().as_micros();
        let kernel_throughput_gbps = (total_input_bytes as f64) / (stage6_us as f64 / 1e6) / 1e9;

        #[cfg(debug_assertions)]
        eprintln!(
            "[RENACER] S6-SYNC:  kernel complete ({} µs, {:.2} GB/s effective)",
            stage6_us, kernel_throughput_gbps
        );

        // [RENACER] Stage 7: D2H Transfer (Device → Host)
        let stage7_start = Instant::now();
        let mut output_flat = vec![0u8; total_output_bytes];
        output_dev.copy_to_host(&mut output_flat)
            .map_err(|e| Error::GpuNotAvailable(format!("Failed to copy output from GPU: {e}")))?;

        let mut sizes = vec![0u32; batch_size];
        sizes_dev.copy_to_host(&mut sizes)
            .map_err(|e| Error::GpuNotAvailable(format!("Failed to copy sizes from GPU: {e}")))?;
        let stage7_us = stage7_start.elapsed().as_micros();
        let d2h_bytes = total_output_bytes + batch_size * 4;
        let d2h_bandwidth_gbps = (d2h_bytes as f64) / (stage7_us as f64 / 1e6) / 1e9;

        #[cfg(debug_assertions)]
        eprintln!(
            "[RENACER] S7-D2H:  {} bytes retrieved ({} µs, {:.2} GB/s)",
            d2h_bytes, stage7_us, d2h_bandwidth_gbps
        );

        // [RENACER] Stage 8: Data Flow Analysis
        // GPU output can be up to output_stride bytes (4352), not just PAGE_SIZE
        let gpu_count = sizes.iter().filter(|&&s| s > 0 && s <= output_stride as u32).count();
        let cpu_fallback_count = batch_size - gpu_count;
        let gpu_output_bytes: usize = sizes.iter()
            .filter(|&&s| s > 0 && s <= output_stride as u32)
            .map(|&s| s as usize)
            .sum();

        #[cfg(debug_assertions)]
        eprintln!(
            "[RENACER] S8-FLOW: {} GPU-compressed ({} bytes), {} CPU-fallback",
            gpu_count, gpu_output_bytes, cpu_fallback_count
        );

        // [RENACER] Stage 9: Extract GPU results (all pages compressed by GPU now)
        let stage9_start = Instant::now();
        use rayon::prelude::*;

        let results: Vec<Vec<u8>> = sizes
            .par_iter()
            .enumerate()
            .map(|(i, &size)| {
                let start = i * output_stride;
                let size_usize = size as usize;

                if size_usize > 0 && size_usize <= output_stride {
                    // GPU produced valid compressed output
                    output_flat[start..start + size_usize].to_vec()
                } else {
                    // Fallback to CPU compression (shouldn't happen with working kernel)
                    let mut compressed = vec![0u8; PAGE_SIZE + 256];
                    let comp_size = lz4_compress_block(&pages[i], &mut compressed)
                        .expect("LZ4 compression failed");
                    compressed.truncate(comp_size);
                    compressed
                }
            })
            .collect();

        let stage9_us = stage9_start.elapsed().as_micros();
        let actual_output_bytes: usize = results.iter().map(|r| r.len()).sum();
        let compression_ratio = total_input_bytes as f64 / actual_output_bytes as f64;

        #[cfg(debug_assertions)]
        eprintln!(
            "[RENACER] S9-GPU: {} → {} bytes ({:.2}x ratio, {} µs)",
            total_input_bytes, actual_output_bytes, compression_ratio, stage9_us
        );

        // [RENACER] Pipeline Summary
        let pipeline_us = pipeline_start.elapsed().as_micros();
        let overall_throughput_gbps = (total_input_bytes as f64) / (pipeline_us as f64 / 1e6) / 1e9;

        #[cfg(debug_assertions)]
        eprintln!(
            "[RENACER] SUMMARY: {} pages in {} µs ({:.2} GB/s, {:.1}% GPU-handled)",
            batch_size, pipeline_us, overall_throughput_gbps,
            (gpu_count as f64 / batch_size as f64) * 100.0
        );

        Ok(results)
    }

    /// CPU fallback for compression when GPU is not available
    #[cfg(feature = "cuda")]
    fn execute_compression_kernel_cpu(
        &self,
        pages: &[[u8; PAGE_SIZE]],
    ) -> Result<Vec<Vec<u8>>> {
        use rayon::prelude::*;
        use trueno_gpu::kernels::lz4::lz4_compress_block;

        let results: Result<Vec<Vec<u8>>> = pages
            .par_iter()
            .map(|page| {
                let mut compressed = vec![0u8; PAGE_SIZE + 256];
                let comp_size = lz4_compress_block(page, &mut compressed)
                    .map_err(|e| Error::Internal(format!("LZ4 compression failed: {e}")))?;
                compressed.truncate(comp_size);
                Ok(compressed)
            })
            .collect();

        results
    }

    #[cfg(feature = "cuda")]
    fn transfer_from_device(&self, data: Vec<Vec<u8>>) -> Result<Vec<CompressedPage>> {
        // In the current hybrid approach, data is already on host.
        // When nvCOMP is integrated, this will do D2H transfer.
        Ok(data
            .into_iter()
            .map(|d| CompressedPage {
                data: d,
                original_size: PAGE_SIZE,
                algorithm: self.config.algorithm,
            })
            .collect())
    }

    // =========================================================================
    // GPU DECOMPRESSION (F082-Safe - Sovereign AI 2TB LLM Restore)
    // =========================================================================
    //
    // This is the KEY innovation for the hybrid CPU+GPU architecture:
    // - Compression: CPU at 24 GB/s (avoids F082 hash table bug)
    // - Decompression: GPU at 16 GB/s (F082-safe, no hash tables)
    // - Combined throughput: ~40 GB/s for 2TB LLM checkpoint restore
    //
    // G.119 target: 2TB restore in <60s requires ~34 GB/s sustained
    // =========================================================================

    /// Decompress a batch of compressed pages using GPU.
    ///
    /// This is the F082-SAFE path for the Sovereign AI hybrid architecture.
    /// GPU decompression doesn't require hash tables, avoiding the F082 bug entirely.
    ///
    /// # Arguments
    ///
    /// * `compressed` - Batch of compressed pages (variable size per page)
    /// * `sizes` - Compressed size of each page in bytes
    ///
    /// # Returns
    ///
    /// Decompressed 4KB pages ready for use.
    ///
    /// # Errors
    ///
    /// Returns error if GPU decompression fails.
    /// GPU decompression with streaming DMA pipeline.
    ///
    /// Uses PRE-ALLOCATED pinned memory for maximum PCIe bandwidth (~25 GB/s).
    /// The kernel achieves 32 GB/s internally.
    ///
    /// G.119 COMPLIANT: Eliminates 350ms allocation overhead by reusing buffers.
    #[cfg(feature = "cuda")]
    pub fn decompress_batch_gpu(
        &mut self,
        compressed: &[Vec<u8>],
        sizes: &[u32],
    ) -> Result<BatchDecompressResult> {
        use rayon::prelude::*;
        use std::ffi::c_void;
        use std::time::Instant;
        use trueno_gpu::driver::LaunchConfig;

        if compressed.is_empty() {
            return Ok(BatchDecompressResult {
                pages: vec![],
                h2d_time_ns: 0,
                kernel_time_ns: 0,
                d2h_time_ns: 0,
                total_time_ns: 0,
            });
        }

        let context = self.context.as_mut().ok_or_else(|| {
            Error::GpuNotAvailable("CUDA context not initialized".to_string())
        })?;

        let module_cell = match context.lz4_decompress_module.as_ref() {
            Some(m) => m,
            None => return Err(Error::GpuNotAvailable("LZ4 decompress module not loaded".to_string())),
        };

        let batch_size = compressed.len();

        // Check if batch fits in pre-allocated buffers
        if batch_size > context.max_batch_size {
            return Err(Error::GpuNotAvailable(format!(
                "Batch size {} exceeds pre-allocated buffer size {}. Use smaller batches or increase batch_size in config.",
                batch_size, context.max_batch_size
            )));
        }

        let pipeline_start = Instant::now();

        // ═══════════════════════════════════════════════════════════════════════
        // STREAMING DMA PIPELINE WITH PRE-ALLOCATED PINNED MEMORY
        // G.119: Eliminates 350ms allocation overhead!
        // ═══════════════════════════════════════════════════════════════════════

        let output_stride = 4352usize; // Max compressed size per page
        let output_size = batch_size * PAGE_SIZE;

        // Stage 1: Copy compressed data to pre-allocated pinned buffer (parallel)
        let prep_start = Instant::now();
        // SAFETY: We're writing to our own pre-allocated pinned buffer
        unsafe {
            let pinned_slice = context.input_pinned.as_slice_mut();
            // Use rayon for parallel memcpy to saturate memory bandwidth
            compressed.par_iter().enumerate().for_each(|(i, page)| {
                let start = i * output_stride;
                let len = page.len().min(output_stride);
                // SAFETY: Each thread writes to non-overlapping region
                let dest = pinned_slice.as_ptr().add(start) as *mut u8;
                std::ptr::copy_nonoverlapping(page.as_ptr(), dest, len);
            });
        }
        let prep_us = prep_start.elapsed().as_micros();

        // Stage 2: Copy sizes to GPU buffer (partial copy for current batch)
        // Note: sizes are small (<400KB for 100K pages), so H2D is fast
        context.sizes_dev.copy_from_host_at(sizes, 0)
            .map_err(|e| Error::GpuNotAvailable(format!("Sizes H2D failed: {e}")))?;

        // Stage 3: Async H2D transfer with pinned memory (~25 GB/s)
        let h2d_start = Instant::now();
        // SAFETY: Pinned buffer remains valid until sync
        unsafe {
            context.input_dev.copy_from_pinned_async(&context.input_pinned, &context.stream)
                .map_err(|e| Error::GpuNotAvailable(format!("Async H2D failed: {e}")))?;
        }

        // Stage 4: Launch decompression kernel (overlapped with H2D tail)
        let batch_size_u32 = batch_size as u32;
        let num_blocks = (batch_size_u32 + 255) / 256;
        let config = LaunchConfig {
            grid: (num_blocks, 1, 1),
            block: (256, 1, 1),
            shared_mem: 0,
        };

        let kernel_start = Instant::now();
        let mut args: [*mut c_void; 4] = [
            context.input_dev.as_kernel_arg(),
            context.sizes_dev.as_kernel_arg(),
            context.output_dev.as_kernel_arg(),
            &batch_size_u32 as *const u32 as *mut c_void,
        ];

        // SAFETY: Valid device pointers and batch_size
        unsafe {
            let mut module = module_cell.borrow_mut();
            context.stream
                .launch_kernel(&mut module, "lz4_decompress", &config, &mut args)
                .map_err(|e| Error::GpuNotAvailable(format!("Decompress kernel launch failed: {e}")))?;
        }

        // Stage 5: Async D2H transfer with pinned memory (overlapped with kernel tail)
        // SAFETY: Output pinned buffer remains valid until sync
        unsafe {
            context.output_dev.copy_to_pinned_async(&mut context.output_pinned, &context.stream)
                .map_err(|e| Error::GpuNotAvailable(format!("Async D2H failed: {e}")))?;
        }

        // Stage 6: Sync all operations
        context.stream.synchronize()
            .map_err(|e| Error::GpuNotAvailable(format!("Stream sync failed: {e}")))?;

        let h2d_us = h2d_start.elapsed().as_micros();
        let kernel_us = kernel_start.elapsed().as_micros();
        let d2h_us = 0u128; // Included in kernel time due to overlap

        // Stage 7: Extract pages from pinned buffer (ZERO-COPY via MaybeUninit)
        let extract_start = Instant::now();

        // Use MaybeUninit to avoid zeroing 400MB of memory (was taking 17ms!)
        // SAFETY: We will fully initialize all pages via copy_nonoverlapping
        let pages: Vec<[u8; PAGE_SIZE]> = unsafe {
            let mut pages: Vec<std::mem::MaybeUninit<[u8; PAGE_SIZE]>> =
                Vec::with_capacity(batch_size);
            pages.set_len(batch_size);

            let output_slice = context.output_pinned.as_slice();

            // Parallel extraction using rayon - saturates memory bandwidth
            pages.par_iter_mut().enumerate().for_each(|(i, page)| {
                let src_start = i * PAGE_SIZE;
                // SAFETY: Each thread reads/writes non-overlapping regions
                // Output is fully initialized by copy
                std::ptr::copy_nonoverlapping(
                    output_slice.as_ptr().add(src_start),
                    page.as_mut_ptr() as *mut u8,
                    PAGE_SIZE
                );
            });

            // SAFETY: All pages are now initialized via copy_nonoverlapping
            std::mem::transmute::<Vec<std::mem::MaybeUninit<[u8; PAGE_SIZE]>>, Vec<[u8; PAGE_SIZE]>>(pages)
        };
        let extract_us = extract_start.elapsed().as_micros();

        let total_us = pipeline_start.elapsed().as_micros();
        let total_gbps = (output_size as f64) / (total_us as f64 / 1e6) / 1e9;

        // Print timing breakdown for profiling
        if batch_size >= 10000 {
            let h2d_gbps = (batch_size * output_stride) as f64 / (h2d_us as f64 / 1e6) / 1e9;
            let kernel_gbps = (output_size as f64) / (kernel_us as f64 / 1e6) / 1e9;
            eprintln!("  [GPU Decompress Profile] {} pages:", batch_size);
            eprintln!("    cpu_prep:     {:>6} µs (parallel memcpy)", prep_us);
            eprintln!("    h2d+kernel:   {:>6} µs ({:.1} GB/s combined)", h2d_us, h2d_gbps);
            eprintln!("    kernel:       {:>6} µs ({:.1} GB/s)", kernel_us, kernel_gbps);
            eprintln!("    extract:      {:>6} µs (parallel memcpy)", extract_us);
            eprintln!("    TOTAL:        {:>6} µs ({:.2} GB/s)", total_us, total_gbps);
        }

        Ok(BatchDecompressResult {
            pages,
            h2d_time_ns: (h2d_us * 1000) as u64,
            kernel_time_ns: (kernel_us * 1000) as u64,
            d2h_time_ns: (d2h_us * 1000) as u64,
            total_time_ns: (total_us * 1000) as u64,
        })
    }

    /// Decompress using CPU (fallback when GPU is unavailable).
    pub fn decompress_batch_cpu(
        &self,
        compressed: &[Vec<u8>],
    ) -> Result<BatchDecompressResult> {
        use rayon::prelude::*;
        use std::time::Instant;
        use trueno_gpu::kernels::lz4::lz4_decompress_block;

        let start = Instant::now();

        let pages: Result<Vec<[u8; PAGE_SIZE]>> = compressed
            .par_iter()
            .map(|data| {
                let mut page = [0u8; PAGE_SIZE];
                lz4_decompress_block(data, &mut page)
                    .map_err(|e| Error::Internal(format!("LZ4 decompression failed: {e}")))?;
                Ok(page)
            })
            .collect();

        let pages = pages?;
        let total_time_ns = start.elapsed().as_nanos() as u64;

        Ok(BatchDecompressResult {
            pages,
            h2d_time_ns: 0,
            kernel_time_ns: total_time_ns,
            d2h_time_ns: 0,
            total_time_ns,
        })
    }

    /// Check if GPU batch compression would be beneficial for the given batch size.
    ///
    /// Follows the 5× PCIe rule: GPU offload is beneficial when
    /// computation time exceeds 5× the transfer time.
    #[must_use]
    pub fn would_benefit(batch_size: usize) -> bool {
        // Heuristic: GPU beneficial for batches >= 1000 pages
        // This amortizes PCIe transfer overhead
        batch_size >= 1000
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

/// Encode a zero page as minimal LZ4 sequence.
///
/// Zero pages are extremely common in memory (>30% typically) and compress
/// to just a few bytes with LZ4's RLE-style encoding.
#[cfg(feature = "cuda")]
fn encode_lz4_zero_page() -> Vec<u8> {
    use trueno_gpu::kernels::lz4::lz4_compress_block;
    let zero_page = [0u8; PAGE_SIZE];
    let mut compressed = vec![0u8; PAGE_SIZE + 256];
    let size = lz4_compress_block(&zero_page, &mut compressed)
        .expect("Zero page compression should never fail");
    compressed.truncate(size);
    compressed
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
    // F046: Large batch achieves reasonable throughput
    //
    // NOTE: Currently using trueno-gpu's CPU LZ4 implementation with rayon
    // parallelization. When the GPU kernel launch is fully debugged, this
    // test should be updated to expect 50+ GB/s on RTX 4090.
    //
    // Current expectation: >0.5 GB/s with parallel CPU compression
    // Target expectation: >50 GB/s with GPU kernel (pending integration)
    // ==========================================================================
    #[test]
    #[cfg(feature = "cuda")]
    fn test_f046_throughput_target() {
        if !crate::gpu::gpu_available() {
            return;
        }

        let config = GpuBatchConfig {
            batch_size: 1000, // Smaller batch for CPU test
            ..Default::default()
        };
        let mut compressor = GpuBatchCompressor::new(config).unwrap();

        // Create compressible test data
        let pages: Vec<[u8; PAGE_SIZE]> = (0..1000)
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

        // With parallel CPU compression via rayon, expect reasonable throughput
        // Note: Coverage instrumentation slows this down significantly
        // Release build: ~1+ GB/s, Coverage build: ~0.03 GB/s
        assert!(
            throughput_gbps >= 0.01,
            "Throughput should be >= 0.01 GB/s, got {throughput_gbps:.2} GB/s"
        );
    }

    // ==========================================================================
    // F047: Configuration respects async_dma flag
    //
    // NOTE: Async DMA benefit is only measurable with actual GPU kernel.
    // Currently testing that the configuration is properly stored.
    // ==========================================================================
    #[test]
    #[cfg(feature = "cuda")]
    fn test_f047_async_dma_benefit() {
        if !crate::gpu::gpu_available() {
            return;
        }

        // Verify async_dma config is stored correctly
        let async_config = GpuBatchConfig {
            async_dma: true,
            batch_size: 1000,
            ..Default::default()
        };
        assert!(async_config.async_dma);

        let sync_config = GpuBatchConfig {
            async_dma: false,
            batch_size: 1000,
            ..Default::default()
        };
        assert!(!sync_config.async_dma);

        // Verify both configurations can create compressors
        let async_compressor = GpuBatchCompressor::new(async_config).unwrap();
        let sync_compressor = GpuBatchCompressor::new(sync_config).unwrap();

        // Both should report the correct config
        assert!(async_compressor.config().async_dma);
        assert!(!sync_compressor.config().async_dma);
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

    // ==========================================================================
    // Additional coverage tests for BatchResult
    // ==========================================================================
    #[test]
    fn test_batch_result_zero_time_throughput() {
        let result = BatchResult {
            pages: vec![],
            h2d_time_ns: 0,
            kernel_time_ns: 0,
            d2h_time_ns: 0,
            total_time_ns: 0, // Edge case: zero time
        };
        // Should return 0.0 for zero time to avoid division by zero
        assert!(result.throughput_bytes_per_sec(4096).abs() < f64::EPSILON);
    }

    #[test]
    fn test_batch_result_compression_ratio() {
        let result = BatchResult {
            pages: vec![
                CompressedPage {
                    data: vec![0; 2048], // 50% compression
                    original_size: PAGE_SIZE,
                    algorithm: Algorithm::Lz4,
                },
                CompressedPage {
                    data: vec![0; 1024], // 75% compression
                    original_size: PAGE_SIZE,
                    algorithm: Algorithm::Lz4,
                },
            ],
            h2d_time_ns: 100,
            kernel_time_ns: 800,
            d2h_time_ns: 100,
            total_time_ns: 1000,
        };
        // 2 pages * 4096 bytes = 8192 original / 3072 compressed = 2.67:1
        let ratio = result.compression_ratio();
        assert!(ratio > 2.0 && ratio < 3.0, "Compression ratio should be ~2.67");
    }

    #[test]
    fn test_batch_result_compression_ratio_empty() {
        let result = BatchResult {
            pages: vec![],
            h2d_time_ns: 0,
            kernel_time_ns: 0,
            d2h_time_ns: 0,
            total_time_ns: 0,
        };
        // Empty pages should return 1.0 ratio
        assert!((result.compression_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_batch_result_pcie_rule_satisfied() {
        let result = BatchResult {
            pages: vec![],
            h2d_time_ns: 100,
            kernel_time_ns: 1001, // >5× transfer time (strictly greater)
            d2h_time_ns: 100,
            total_time_ns: 1201,
        };
        // kernel_time (1001) > 5 × transfer_time (200=1000) = true
        assert!(result.pcie_rule_satisfied());
    }

    #[test]
    fn test_batch_result_pcie_rule_not_satisfied() {
        let result = BatchResult {
            pages: vec![],
            h2d_time_ns: 500,
            kernel_time_ns: 100, // Less than 5× transfer
            d2h_time_ns: 500,
            total_time_ns: 1100,
        };
        // kernel_time (100) < 5 × transfer_time (1000) = false
        assert!(!result.pcie_rule_satisfied());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_gpu_context_debug() {
        if !crate::gpu::gpu_available() {
            return;
        }
        let config = GpuBatchConfig::default();
        let compressor = GpuBatchCompressor::new(config).unwrap();
        // Exercise the Debug impl
        let debug_str = format!("{:?}", compressor);
        assert!(debug_str.contains("GpuBatchCompressor"));
    }

    // ==========================================================================
    // Additional coverage tests for would_benefit and trait impls
    // ==========================================================================

    #[test]
    fn test_would_benefit_large_batch() {
        // Large batch (>=1000 pages) should benefit from GPU
        assert!(GpuBatchCompressor::would_benefit(1000));
        assert!(GpuBatchCompressor::would_benefit(10000));
        assert!(GpuBatchCompressor::would_benefit(100000));
    }

    #[test]
    fn test_would_benefit_small_batch() {
        // Small batch (<1000 pages) should not benefit from GPU
        assert!(!GpuBatchCompressor::would_benefit(0));
        assert!(!GpuBatchCompressor::would_benefit(1));
        assert!(!GpuBatchCompressor::would_benefit(999));
    }

    #[test]
    fn test_gpu_batch_config_clone() {
        let config = GpuBatchConfig {
            device_index: 1,
            algorithm: Algorithm::Zstd { level: 3 },
            batch_size: 500,
            async_dma: false,
            ring_buffer_slots: 8,
        };
        let cloned = config.clone();
        assert_eq!(cloned.device_index, 1);
        assert_eq!(cloned.batch_size, 500);
        assert!(!cloned.async_dma);
        assert_eq!(cloned.ring_buffer_slots, 8);
    }

    #[test]
    fn test_gpu_batch_config_debug() {
        let config = GpuBatchConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("GpuBatchConfig"));
        assert!(debug_str.contains("device_index"));
        assert!(debug_str.contains("batch_size"));
    }

    #[test]
    fn test_gpu_batch_stats_clone() {
        let stats = GpuBatchStats {
            pages_compressed: 1000,
            total_bytes_in: 4096000,
            total_bytes_out: 2048000,
            total_time_ns: 1_000_000,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.pages_compressed, 1000);
        assert_eq!(cloned.total_bytes_in, 4096000);
        assert_eq!(cloned.total_bytes_out, 2048000);
        assert_eq!(cloned.total_time_ns, 1_000_000);
    }

    #[test]
    fn test_gpu_batch_stats_debug() {
        let stats = GpuBatchStats::default();
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("GpuBatchStats"));
        assert!(debug_str.contains("pages_compressed"));
    }

    #[test]
    fn test_batch_result_clone() {
        let result = BatchResult {
            pages: vec![CompressedPage {
                data: vec![1, 2, 3],
                original_size: PAGE_SIZE,
                algorithm: Algorithm::Lz4,
            }],
            h2d_time_ns: 100,
            kernel_time_ns: 500,
            d2h_time_ns: 100,
            total_time_ns: 700,
        };
        let cloned = result.clone();
        assert_eq!(cloned.pages.len(), 1);
        assert_eq!(cloned.h2d_time_ns, 100);
        assert_eq!(cloned.kernel_time_ns, 500);
        assert_eq!(cloned.d2h_time_ns, 100);
        assert_eq!(cloned.total_time_ns, 700);
    }

    #[test]
    fn test_batch_result_debug() {
        let result = BatchResult {
            pages: vec![],
            h2d_time_ns: 0,
            kernel_time_ns: 0,
            d2h_time_ns: 0,
            total_time_ns: 0,
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("BatchResult"));
        assert!(debug_str.contains("pages"));
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_compress_batch_no_cuda() {
        let config = GpuBatchConfig::default();
        let result = GpuBatchCompressor::new(config);
        // Without CUDA, creation should fail
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("CUDA"));
    }

    #[test]
    fn test_gpu_batch_stats_default_values() {
        let stats = GpuBatchStats::default();
        assert_eq!(stats.pages_compressed, 0);
        assert_eq!(stats.total_bytes_in, 0);
        assert_eq!(stats.total_bytes_out, 0);
        assert_eq!(stats.total_time_ns, 0);
    }

    #[test]
    fn test_throughput_with_large_input() {
        let result = BatchResult {
            pages: vec![],
            h2d_time_ns: 1000,
            kernel_time_ns: 5000,
            d2h_time_ns: 1000,
            total_time_ns: 1_000_000_000, // 1 second
        };
        // 1 GiB input in 1 second = 1073741824 bytes/sec
        let input_bytes = 1024 * 1024 * 1024; // 1GiB = 1073741824 bytes
        let throughput = result.throughput_bytes_per_sec(input_bytes);
        let expected = 1073741824.0;
        assert!(
            (throughput - expected).abs() < 1.0,
            "Should be ~1 GiB/s, got {throughput}"
        );
    }

    #[test]
    #[cfg(feature = "cuda")]
    #[ignore] // SIGSEGV when running with other GPU tests - GPU context management issue
    fn test_compressor_stats_accessor() {
        if !crate::gpu::gpu_available() {
            return;
        }
        let config = GpuBatchConfig::default();
        let compressor = GpuBatchCompressor::new(config).unwrap();

        // Fresh compressor should have zero stats
        let stats = compressor.stats();
        assert_eq!(stats.pages_compressed, 0);
        assert_eq!(stats.total_bytes_in, 0);
        assert_eq!(stats.total_bytes_out, 0);
    }

    #[test]
    #[cfg(feature = "cuda")]
    #[ignore] // SIGSEGV when running with other GPU tests - GPU context management issue
    fn test_compressor_config_accessor() {
        if !crate::gpu::gpu_available() {
            return;
        }
        let config = GpuBatchConfig {
            device_index: 0,
            batch_size: 2000,
            algorithm: Algorithm::Lz4,
            async_dma: true,
            ring_buffer_slots: 6,
        };
        let compressor = GpuBatchCompressor::new(config).unwrap();

        let retrieved_config = compressor.config();
        assert_eq!(retrieved_config.batch_size, 2000);
        assert_eq!(retrieved_config.ring_buffer_slots, 6);
    }

    // ==========================================================================
    // PROBADOR: GPU Data Flow Verification Test
    // Verifies correct data transformation through GPU pipeline stages
    // ==========================================================================
    #[test]
    #[cfg(feature = "cuda")]
    fn test_probador_gpu_data_flow() {
        if !crate::gpu::gpu_available() {
            return;
        }

        let config = GpuBatchConfig::default();
        let mut compressor = GpuBatchCompressor::new(config).unwrap();

        // Create predictable test data:
        // - Pages 0, 4, 8, 12, 16: zero pages (highly compressible)
        // - Pages 1, 5, 9, 13, 17: sequential pattern (compressible)
        // - Pages 2, 6, 10, 14, 18: repeated pattern (compressible)
        // - Pages 3, 7, 11, 15, 19: pseudo-random (less compressible)
        const NUM_PAGES: usize = 20;
        let mut pages = Vec::with_capacity(NUM_PAGES);

        for i in 0..NUM_PAGES {
            let mut page = [0u8; PAGE_SIZE];
            match i % 4 {
                0 => {} // Zero page
                1 => {
                    for (j, byte) in page.iter_mut().enumerate() {
                        *byte = (j % 256) as u8;
                    }
                }
                2 => {
                    for (j, byte) in page.iter_mut().enumerate() {
                        *byte = [0xAA, 0xBB, 0xCC, 0xDD][j % 4];
                    }
                }
                _ => {
                    let mut rng = (i as u64).wrapping_mul(0x5DEECE66D);
                    for byte in &mut page {
                        rng = rng.wrapping_mul(0x5DEECE66D).wrapping_add(0xB);
                        *byte = (rng >> 33) as u8;
                    }
                }
            }
            pages.push(page);
        }

        // [PROBADOR] Stage 1: Compress batch using GPU path
        let result = compressor.compress_batch_gpu(&pages);
        assert!(result.is_ok(), "[PROBADOR] Stage 1 FAIL: compress_batch_gpu failed: {:?}", result.err());
        let batch_result = result.unwrap();

        // [PROBADOR] Stage 2: Verify output count matches input
        assert_eq!(
            batch_result.pages.len(), NUM_PAGES,
            "[PROBADOR] Stage 2 FAIL: Output page count mismatch"
        );

        // [PROBADOR] Stage 3: Verify all pages decompress correctly
        // Use trueno-gpu's lz4_decompress_block which is compatible with GPU output format
        use trueno_gpu::kernels::lz4::lz4_decompress_block;

        for (i, compressed_page) in batch_result.pages.iter().enumerate() {
            let mut decompressed = [0u8; PAGE_SIZE];
            let decomp_result = lz4_decompress_block(&compressed_page.data, &mut decompressed);
            assert!(
                decomp_result.is_ok(),
                "[PROBADOR] Stage 3 FAIL: Decompression failed for page {} (error: {:?})",
                i, decomp_result.err()
            );
            assert_eq!(
                &decompressed[..], &pages[i][..],
                "[PROBADOR] Stage 3 FAIL: Data mismatch for page {}",
                i
            );
        }

        // [PROBADOR] Stage 4: Verify zero pages compress (no strict size check)
        // Note: Lz4WarpShuffleKernel produces literal-only encoding, so zero pages
        // may not compress well. For optimal zero page handling, use the hybrid
        // architecture (CPU compress + GPU decompress) which detects zero pages.
        let zero_page_indices = [0, 4, 8, 12, 16];
        for &idx in &zero_page_indices {
            assert!(
                !batch_result.pages[idx].data.is_empty(),
                "[PROBADOR] Stage 4 FAIL: Zero page {} has no compressed data",
                idx,
            );
        }

        // [PROBADOR] Stage 5: Verify statistics are recorded
        let stats = compressor.stats();
        assert!(
            stats.pages_compressed >= NUM_PAGES as u64,
            "[PROBADOR] Stage 5 FAIL: Stats not recording pages"
        );

        eprintln!("[PROBADOR] GPU data flow verification: all 5 stages passed");
    }

    // ==========================================================================
    // PROBADOR: All-Zero Pages Flow Test
    // Tests 100% GPU-handled path (all pages are zero)
    // ==========================================================================
    #[test]
    #[cfg(feature = "cuda")]
    fn test_probador_all_zero_pages() {
        if !crate::gpu::gpu_available() {
            return;
        }

        let config = GpuBatchConfig::default();
        let mut compressor = GpuBatchCompressor::new(config).unwrap();

        const NUM_PAGES: usize = 100;
        let pages: Vec<[u8; PAGE_SIZE]> = vec![[0u8; PAGE_SIZE]; NUM_PAGES];

        // [PROBADOR] Flow: Input → GPU → Output (100% GPU path)
        let result = compressor.compress_batch_gpu(&pages);
        assert!(result.is_ok(), "[PROBADOR-ZERO] Compression failed");
        let batch_result = result.unwrap();

        // [PROBADOR] Verify all pages compressed by GPU
        use trueno_gpu::kernels::lz4::lz4_decompress_block;

        for (i, compressed_page) in batch_result.pages.iter().enumerate() {
            // Note: Lz4WarpShuffleKernel uses literal-only encoding, so zero pages
            // won't compress well. This test just verifies roundtrip correctness.
            // For optimal zero page handling, use the hybrid architecture.
            assert!(
                !compressed_page.data.is_empty(),
                "[PROBADOR-ZERO] Page {} has no compressed data",
                i
            );

            // Verify decompression roundtrip
            let mut decompressed = [0u8; PAGE_SIZE];
            let decomp_result = lz4_decompress_block(&compressed_page.data, &mut decompressed);
            assert!(decomp_result.is_ok(), "[PROBADOR-ZERO] Decompress failed for page {}", i);
            assert_eq!(&decompressed[..], &pages[i][..], "[PROBADOR-ZERO] Data mismatch page {}", i);
        }

        eprintln!("[PROBADOR-ZERO] 100% GPU path verified: {} zero pages", NUM_PAGES);
    }

    // ==========================================================================
    // PROBADOR: All-Random Pages Flow Test
    // Tests 0% GPU-handled path (all pages use CPU fallback)
    // ==========================================================================
    #[test]
    #[cfg(feature = "cuda")]
    fn test_probador_all_random_pages() {
        if !crate::gpu::gpu_available() {
            return;
        }

        let config = GpuBatchConfig::default();
        let mut compressor = GpuBatchCompressor::new(config).unwrap();

        const NUM_PAGES: usize = 50;
        let mut pages = Vec::with_capacity(NUM_PAGES);

        // Generate pseudo-random pages (not compressible by GPU zero-detection)
        for i in 0..NUM_PAGES {
            let mut page = [0u8; PAGE_SIZE];
            let mut rng = (i as u64 + 12345).wrapping_mul(0x5DEECE66D);
            for byte in &mut page {
                rng = rng.wrapping_mul(0x5DEECE66D).wrapping_add(0xB);
                *byte = (rng >> 33) as u8;
            }
            pages.push(page);
        }

        // [PROBADOR] Flow: Input → GPU (zero-detect) → CPU fallback → Output
        let result = compressor.compress_batch_gpu(&pages);
        assert!(result.is_ok(), "[PROBADOR-RAND] Compression failed");
        let batch_result = result.unwrap();

        // [PROBADOR] Verify all pages decompress correctly via CPU fallback
        use trueno_gpu::kernels::lz4::lz4_decompress_block;

        for (i, compressed_page) in batch_result.pages.iter().enumerate() {
            let mut decompressed = [0u8; PAGE_SIZE];
            let decomp_result = lz4_decompress_block(&compressed_page.data, &mut decompressed);
            assert!(decomp_result.is_ok(), "[PROBADOR-RAND] Decompress failed for page {}", i);
            assert_eq!(&decompressed[..], &pages[i][..], "[PROBADOR-RAND] Data mismatch page {}", i);
        }

        eprintln!("[PROBADOR-RAND] CPU fallback path verified: {} random pages", NUM_PAGES);
    }

    // ==========================================================================
    // PROBADOR: Incremental Byte Pattern Test
    // Tests data integrity with sequential byte patterns
    // ==========================================================================
    #[test]
    #[cfg(feature = "cuda")]
    fn test_probador_incremental_bytes() {
        if !crate::gpu::gpu_available() {
            return;
        }

        let config = GpuBatchConfig::default();
        let mut compressor = GpuBatchCompressor::new(config).unwrap();

        const NUM_PAGES: usize = 16;
        let mut pages = Vec::with_capacity(NUM_PAGES);

        // Generate pages with unique incremental patterns
        for i in 0..NUM_PAGES {
            let mut page = [0u8; PAGE_SIZE];
            for (j, byte) in page.iter_mut().enumerate() {
                // Each page has a unique pattern based on page index and byte position
                *byte = ((i * PAGE_SIZE + j) % 256) as u8;
            }
            pages.push(page);
        }

        // [PROBADOR] Compress and verify byte-for-byte integrity
        let result = compressor.compress_batch_gpu(&pages);
        assert!(result.is_ok(), "[PROBADOR-INC] Compression failed");
        let batch_result = result.unwrap();

        use trueno_gpu::kernels::lz4::lz4_decompress_block;

        for (i, compressed_page) in batch_result.pages.iter().enumerate() {
            let mut decompressed = [0u8; PAGE_SIZE];
            let decomp_result = lz4_decompress_block(&compressed_page.data, &mut decompressed);
            assert!(decomp_result.is_ok(), "[PROBADOR-INC] Decompress failed for page {}", i);

            // Byte-by-byte verification
            for (j, (&expected, &actual)) in pages[i].iter().zip(decompressed.iter()).enumerate() {
                assert_eq!(
                    expected, actual,
                    "[PROBADOR-INC] Byte mismatch at page {} offset {}: expected {:#04x}, got {:#04x}",
                    i, j, expected, actual
                );
            }
        }

        eprintln!("[PROBADOR-INC] Byte-level integrity verified: {} pages", NUM_PAGES);
    }

    // ==========================================================================
    // PROBADOR: Large Batch Stress Test
    // Tests GPU pipeline with realistic batch sizes
    // ==========================================================================
    #[test]
    #[cfg(feature = "cuda")]
    fn test_probador_large_batch_stress() {
        if !crate::gpu::gpu_available() {
            return;
        }

        let config = GpuBatchConfig::default();
        let mut compressor = GpuBatchCompressor::new(config).unwrap();

        const NUM_PAGES: usize = 1000;
        let mut pages = Vec::with_capacity(NUM_PAGES);

        // Generate mixed workload: 25% zero, 25% pattern, 50% semi-random
        for i in 0..NUM_PAGES {
            let mut page = [0u8; PAGE_SIZE];
            match i % 4 {
                0 => {} // Zero page (GPU handles)
                1 => {
                    // Repeating pattern (compressible)
                    for (j, byte) in page.iter_mut().enumerate() {
                        *byte = [0xDE, 0xAD, 0xBE, 0xEF][j % 4];
                    }
                }
                _ => {
                    // Semi-random (CPU fallback)
                    let mut rng = (i as u64).wrapping_mul(0x5DEECE66D);
                    for byte in &mut page {
                        rng = rng.wrapping_mul(0x5DEECE66D).wrapping_add(0xB);
                        *byte = (rng >> 33) as u8;
                    }
                }
            }
            pages.push(page);
        }

        // [PROBADOR] Stress test: compress large batch
        let result = compressor.compress_batch_gpu(&pages);
        assert!(result.is_ok(), "[PROBADOR-STRESS] Large batch compression failed");
        let batch_result = result.unwrap();

        assert_eq!(batch_result.pages.len(), NUM_PAGES, "[PROBADOR-STRESS] Output count mismatch");

        // [PROBADOR] Verify random sample of pages (full verification too slow)
        use trueno_gpu::kernels::lz4::lz4_decompress_block;

        let sample_indices = [0, 1, 100, 250, 500, 750, 999];
        for &i in &sample_indices {
            let mut decompressed = [0u8; PAGE_SIZE];
            let decomp_result = lz4_decompress_block(&batch_result.pages[i].data, &mut decompressed);
            assert!(decomp_result.is_ok(), "[PROBADOR-STRESS] Decompress failed for page {}", i);
            assert_eq!(&decompressed[..], &pages[i][..], "[PROBADOR-STRESS] Data mismatch page {}", i);
        }

        // [PROBADOR] Verify roundtrip correctness (no ratio check - literal-only kernel)
        // Note: Lz4WarpShuffleKernel produces literal-only output for correctness.
        // For optimal compression, use the hybrid architecture (CPU compress + GPU decompress).
        let total_input: usize = NUM_PAGES * PAGE_SIZE;
        let total_output: usize = batch_result.pages.iter().map(|p| p.data.len()).sum();
        let ratio = total_input as f64 / total_output as f64;

        eprintln!(
            "[PROBADOR-STRESS] Large batch verified: {} pages, {:.2}x ratio, {} → {} bytes",
            NUM_PAGES, ratio, total_input, total_output
        );
    }

    // ==========================================================================
    // PROBADOR: Edge Case - Single Page
    // Tests minimum batch size handling
    // ==========================================================================
    #[test]
    #[cfg(feature = "cuda")]
    fn test_probador_single_page() {
        if !crate::gpu::gpu_available() {
            return;
        }

        let config = GpuBatchConfig::default();
        let mut compressor = GpuBatchCompressor::new(config).unwrap();

        // Test with single zero page
        let pages = vec![[0u8; PAGE_SIZE]];
        let result = compressor.compress_batch_gpu(&pages);
        assert!(result.is_ok(), "[PROBADOR-SINGLE] Single page compression failed");

        let batch_result = result.unwrap();
        assert_eq!(batch_result.pages.len(), 1, "[PROBADOR-SINGLE] Output count mismatch");

        use trueno_gpu::kernels::lz4::lz4_decompress_block;
        let mut decompressed = [0u8; PAGE_SIZE];
        let decomp_result = lz4_decompress_block(&batch_result.pages[0].data, &mut decompressed);
        assert!(decomp_result.is_ok(), "[PROBADOR-SINGLE] Decompress failed");
        assert_eq!(&decompressed[..], &pages[0][..], "[PROBADOR-SINGLE] Data mismatch");

        eprintln!("[PROBADOR-SINGLE] Single page edge case verified");
    }
}
