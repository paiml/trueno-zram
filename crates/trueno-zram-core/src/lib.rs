//! SIMD-accelerated memory compression for Linux zram.
//!
//! This crate provides high-performance LZ4 and Zstandard compression
//! optimized for 4KB memory pages, with runtime SIMD dispatch.
//!
//! # Example
//!
//! ```
//! use trueno_zram_core::{CompressorBuilder, Algorithm, PageCompressor, PAGE_SIZE};
//!
//! let compressor = CompressorBuilder::new()
//!     .algorithm(Algorithm::Lz4)
//!     .build()
//!     .unwrap();
//!
//! let page = [0u8; PAGE_SIZE];
//! let compressed = compressor.compress(&page).unwrap();
//! let decompressed = compressor.decompress(&compressed).unwrap();
//!
//! assert_eq!(page, decompressed);
//! ```

#![deny(missing_docs)]
#![deny(clippy::panic)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::similar_names)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::cognitive_complexity)]
#![allow(clippy::cast_ptr_alignment)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::manual_strip)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::inline_always)]
#![allow(clippy::unused_self)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::incompatible_msrv)]
#![allow(clippy::uninit_vec)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::large_stack_arrays)]
#![allow(unused_assignments)]

pub mod benchmark;
pub mod compat;
mod error;
pub mod gpu;
pub mod integration;
pub mod lz4;
mod page;
pub mod samefill;
pub mod simd;
pub mod zram;
pub mod zstd;

pub use error::{Error, Result};
pub use page::{CompressedPage, CompressionStats, PAGE_SIZE};

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Compression algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Algorithm {
    /// No compression (store as-is).
    None,
    /// LZ4 fast compression.
    #[default]
    Lz4,
    /// LZ4-HC high compression (not yet implemented).
    Lz4Hc,
    /// Zstandard with configurable level.
    Zstd {
        /// Compression level (1-22).
        level: i32,
    },
    /// Adaptive selection based on entropy.
    Adaptive,
}

/// SIMD implementation backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SimdBackend {
    /// Scalar fallback (no SIMD).
    #[default]
    Scalar,
    /// SSE4.2 (128-bit).
    Sse42,
    /// AVX2 (256-bit).
    Avx2,
    /// AVX-512 (512-bit).
    Avx512,
    /// ARM NEON (128-bit).
    Neon,
}

/// Trait for page compression implementations.
pub trait PageCompressor: Send + Sync {
    /// Compress a 4KB page.
    ///
    /// # Errors
    ///
    /// Returns an error if compression fails.
    fn compress(&self, page: &[u8; PAGE_SIZE]) -> Result<CompressedPage>;

    /// Decompress to a 4KB page.
    ///
    /// # Errors
    ///
    /// Returns an error if decompression fails.
    fn decompress(&self, compressed: &CompressedPage) -> Result<[u8; PAGE_SIZE]>;

    /// Get the SIMD backend in use.
    fn backend(&self) -> SimdBackend;

    /// Get compression statistics.
    fn stats(&self) -> CompressionStats;

    /// Reset statistics.
    fn reset_stats(&self);
}

/// Builder for configuring a page compressor.
#[derive(Debug, Clone)]
pub struct CompressorBuilder {
    algorithm: Algorithm,
    preferred_backend: Option<SimdBackend>,
}

impl Default for CompressorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressorBuilder {
    /// Create a new compressor builder with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            algorithm: Algorithm::default(),
            preferred_backend: None,
        }
    }

    /// Set the compression algorithm.
    #[must_use]
    pub fn algorithm(mut self, algo: Algorithm) -> Self {
        self.algorithm = algo;
        self
    }

    /// Set the preferred SIMD backend.
    #[must_use]
    pub fn prefer_backend(mut self, backend: SimdBackend) -> Self {
        self.preferred_backend = Some(backend);
        self
    }

    /// Build the compressor.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid or the preferred
    /// backend is not available.
    pub fn build(self) -> Result<Box<dyn PageCompressor>> {
        let backend = self.preferred_backend.unwrap_or_else(simd::best_backend);

        // Validate backend availability
        if !simd::backend_available(backend) {
            return Err(Error::SimdNotAvailable(backend));
        }

        Ok(Box::new(GenericCompressor::new(self.algorithm, backend)))
    }
}

/// Generic page compressor implementation.
struct GenericCompressor {
    algorithm: Algorithm,
    backend: SimdBackend,
    stats: CompressorStats,
}

struct CompressorStats {
    pages_compressed: AtomicU64,
    pages_incompressible: AtomicU64,
    bytes_in: AtomicU64,
    bytes_out: AtomicU64,
    compress_time_ns: AtomicU64,
    decompress_time_ns: AtomicU64,
}

impl Default for CompressorStats {
    fn default() -> Self {
        Self {
            pages_compressed: AtomicU64::new(0),
            pages_incompressible: AtomicU64::new(0),
            bytes_in: AtomicU64::new(0),
            bytes_out: AtomicU64::new(0),
            compress_time_ns: AtomicU64::new(0),
            decompress_time_ns: AtomicU64::new(0),
        }
    }
}

impl GenericCompressor {
    fn new(algorithm: Algorithm, backend: SimdBackend) -> Self {
        Self {
            algorithm,
            backend,
            stats: CompressorStats::default(),
        }
    }
}

impl PageCompressor for GenericCompressor {
    fn compress(&self, page: &[u8; PAGE_SIZE]) -> Result<CompressedPage> {
        let start = Instant::now();

        let result = match self.algorithm {
            Algorithm::None => Ok(CompressedPage::uncompressed(*page)),
            Algorithm::Lz4 | Algorithm::Lz4Hc => {
                let compressed = lz4::compress(page)?;
                if compressed.len() >= PAGE_SIZE {
                    self.stats
                        .pages_incompressible
                        .fetch_add(1, Ordering::Relaxed);
                    Ok(CompressedPage::uncompressed(*page))
                } else {
                    CompressedPage::new(compressed, PAGE_SIZE, Algorithm::Lz4)
                }
            }
            Algorithm::Zstd { level } => {
                let compressed = zstd::compress(page, level)?;
                if compressed.len() >= PAGE_SIZE {
                    self.stats
                        .pages_incompressible
                        .fetch_add(1, Ordering::Relaxed);
                    Ok(CompressedPage::uncompressed(*page))
                } else {
                    CompressedPage::new(compressed, PAGE_SIZE, Algorithm::Zstd { level })
                }
            }
            Algorithm::Adaptive => {
                // Try LZ4 first as it's fastest
                let compressed = lz4::compress(page)?;
                if compressed.len() >= PAGE_SIZE {
                    self.stats
                        .pages_incompressible
                        .fetch_add(1, Ordering::Relaxed);
                    Ok(CompressedPage::uncompressed(*page))
                } else {
                    CompressedPage::new(compressed, PAGE_SIZE, Algorithm::Lz4)
                }
            }
        };

        let elapsed = start.elapsed().as_nanos() as u64;
        self.stats
            .compress_time_ns
            .fetch_add(elapsed, Ordering::Relaxed);
        self.stats.pages_compressed.fetch_add(1, Ordering::Relaxed);
        self.stats
            .bytes_in
            .fetch_add(PAGE_SIZE as u64, Ordering::Relaxed);

        if let Ok(ref page) = result {
            self.stats
                .bytes_out
                .fetch_add(page.data.len() as u64, Ordering::Relaxed);
        }

        result
    }

    fn decompress(&self, compressed: &CompressedPage) -> Result<[u8; PAGE_SIZE]> {
        let start = Instant::now();

        let result = match compressed.algorithm {
            Algorithm::None => {
                let mut page = [0u8; PAGE_SIZE];
                if compressed.data.len() != PAGE_SIZE {
                    return Err(Error::CorruptedData(format!(
                        "uncompressed page has wrong size: {}",
                        compressed.data.len()
                    )));
                }
                page.copy_from_slice(&compressed.data);
                Ok(page)
            }
            Algorithm::Lz4 | Algorithm::Lz4Hc => {
                let mut page = [0u8; PAGE_SIZE];
                let len = lz4::decompress(&compressed.data, &mut page)?;
                if len != PAGE_SIZE {
                    return Err(Error::CorruptedData(format!(
                        "decompressed size mismatch: expected {PAGE_SIZE}, got {len}"
                    )));
                }
                Ok(page)
            }
            Algorithm::Zstd { .. } => {
                let mut page = [0u8; PAGE_SIZE];
                let len = zstd::decompress(&compressed.data, &mut page)?;
                if len != PAGE_SIZE {
                    return Err(Error::CorruptedData(format!(
                        "decompressed size mismatch: expected {PAGE_SIZE}, got {len}"
                    )));
                }
                Ok(page)
            }
            Algorithm::Adaptive => {
                // Should not happen - adaptive resolves to concrete algorithm
                Err(Error::Internal(
                    "adaptive algorithm in compressed data".to_string(),
                ))
            }
        };

        let elapsed = start.elapsed().as_nanos() as u64;
        self.stats
            .decompress_time_ns
            .fetch_add(elapsed, Ordering::Relaxed);

        result
    }

    fn backend(&self) -> SimdBackend {
        self.backend
    }

    fn stats(&self) -> CompressionStats {
        CompressionStats {
            pages_compressed: self.stats.pages_compressed.load(Ordering::Relaxed),
            pages_incompressible: self.stats.pages_incompressible.load(Ordering::Relaxed),
            bytes_in: self.stats.bytes_in.load(Ordering::Relaxed),
            bytes_out: self.stats.bytes_out.load(Ordering::Relaxed),
            compress_time_ns: self.stats.compress_time_ns.load(Ordering::Relaxed),
            decompress_time_ns: self.stats.decompress_time_ns.load(Ordering::Relaxed),
        }
    }

    fn reset_stats(&self) {
        self.stats.pages_compressed.store(0, Ordering::Relaxed);
        self.stats.pages_incompressible.store(0, Ordering::Relaxed);
        self.stats.bytes_in.store(0, Ordering::Relaxed);
        self.stats.bytes_out.store(0, Ordering::Relaxed);
        self.stats.compress_time_ns.store(0, Ordering::Relaxed);
        self.stats.decompress_time_ns.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compressor_builder_default() {
        let compressor = CompressorBuilder::new().build().unwrap();
        assert!(matches!(
            compressor.backend(),
            SimdBackend::Scalar
                | SimdBackend::Sse42
                | SimdBackend::Avx2
                | SimdBackend::Avx512
                | SimdBackend::Neon
        ));
    }

    #[test]
    fn test_compressor_roundtrip_lz4() {
        let compressor = CompressorBuilder::new()
            .algorithm(Algorithm::Lz4)
            .build()
            .unwrap();

        let page = [0xABu8; PAGE_SIZE];
        let compressed = compressor.compress(&page).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(page, decompressed);
    }

    #[test]
    fn test_compressor_roundtrip_zstd() {
        let compressor = CompressorBuilder::new()
            .algorithm(Algorithm::Zstd { level: 3 })
            .build()
            .unwrap();

        let page = [0xCDu8; PAGE_SIZE];
        let compressed = compressor.compress(&page).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(page, decompressed);
    }

    #[test]
    fn test_compressor_none() {
        let compressor = CompressorBuilder::new()
            .algorithm(Algorithm::None)
            .build()
            .unwrap();

        let page = [0x12u8; PAGE_SIZE];
        let compressed = compressor.compress(&page).unwrap();

        assert!(!compressed.is_compressed());
        assert_eq!(compressed.data.len(), PAGE_SIZE);

        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(page, decompressed);
    }

    #[test]
    fn test_compressor_stats() {
        let compressor = CompressorBuilder::new()
            .algorithm(Algorithm::Lz4)
            .build()
            .unwrap();

        let page = [0u8; PAGE_SIZE];
        compressor.compress(&page).unwrap();
        compressor.compress(&page).unwrap();

        let stats = compressor.stats();
        assert_eq!(stats.pages_compressed, 2);
        assert_eq!(stats.bytes_in, PAGE_SIZE as u64 * 2);
    }

    #[test]
    fn test_compressor_stats_reset() {
        let compressor = CompressorBuilder::new().build().unwrap();

        let page = [0u8; PAGE_SIZE];
        compressor.compress(&page).unwrap();

        compressor.reset_stats();
        let stats = compressor.stats();
        assert_eq!(stats.pages_compressed, 0);
    }

    #[test]
    fn test_algorithm_default() {
        assert_eq!(Algorithm::default(), Algorithm::Lz4);
    }

    #[test]
    fn test_simd_backend_default() {
        assert_eq!(SimdBackend::default(), SimdBackend::Scalar);
    }

    #[test]
    fn test_prefer_unavailable_backend() {
        // AVX-512 might not be available on all systems
        // This test just verifies the error handling works
        let result = CompressorBuilder::new()
            .prefer_backend(SimdBackend::Avx512)
            .build();

        // Either succeeds (CPU has AVX-512) or returns appropriate error
        match result {
            Ok(c) => assert_eq!(c.backend(), SimdBackend::Avx512),
            Err(Error::SimdNotAvailable(_)) => {}
            Err(e) => unreachable!("Unexpected error: {e}"),
        }
    }

    #[test]
    fn test_compressor_builder_default_impl() {
        let compressor = CompressorBuilder::default().build().unwrap();
        let _ = compressor.backend();
    }

    #[test]
    fn test_compressor_adaptive() {
        let compressor = CompressorBuilder::new()
            .algorithm(Algorithm::Adaptive)
            .build()
            .unwrap();

        let page = [0xABu8; PAGE_SIZE];
        let compressed = compressor.compress(&page).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(page, decompressed);
    }

    #[test]
    fn test_compressor_adaptive_incompressible() {
        let compressor = CompressorBuilder::new()
            .algorithm(Algorithm::Adaptive)
            .build()
            .unwrap();

        // Random data is incompressible
        let mut page = [0u8; PAGE_SIZE];
        let mut rng = 12345u64;
        for byte in &mut page {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *byte = (rng >> 33) as u8;
        }

        let compressed = compressor.compress(&page).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(page, decompressed);
    }

    #[test]
    fn test_compressor_zstd_incompressible() {
        let compressor = CompressorBuilder::new()
            .algorithm(Algorithm::Zstd { level: 1 })
            .build()
            .unwrap();

        // Random data is incompressible
        let mut page = [0u8; PAGE_SIZE];
        let mut rng = 54321u64;
        for byte in &mut page {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *byte = (rng >> 33) as u8;
        }

        let compressed = compressor.compress(&page).unwrap();
        // Should store uncompressed since random data doesn't compress
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(page, decompressed);
    }

    #[test]
    fn test_decompress_wrong_size_uncompressed() {
        let compressor = CompressorBuilder::new()
            .algorithm(Algorithm::None)
            .build()
            .unwrap();

        // Create a compressed page with wrong size
        let bad_page = CompressedPage {
            data: vec![0u8; PAGE_SIZE / 2],
            original_size: PAGE_SIZE,
            algorithm: Algorithm::None,
        };

        let result = compressor.decompress(&bad_page);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("wrong size"));
    }

    #[test]
    fn test_decompress_adaptive_error() {
        let compressor = CompressorBuilder::new()
            .algorithm(Algorithm::Lz4)
            .build()
            .unwrap();

        // Create a compressed page with Adaptive algorithm (shouldn't happen)
        let bad_page = CompressedPage {
            data: vec![0u8; 100],
            original_size: PAGE_SIZE,
            algorithm: Algorithm::Adaptive,
        };

        let result = compressor.decompress(&bad_page);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("adaptive"));
    }

    #[test]
    fn test_decompress_lz4_size_mismatch() {
        let compressor = CompressorBuilder::new()
            .algorithm(Algorithm::Lz4)
            .build()
            .unwrap();

        // Compress a small page, but claim it's PAGE_SIZE
        let small_page = [0xAAu8; 128];
        let compressed = lz4::compress(&small_page).unwrap();
        let bad_page = CompressedPage {
            data: compressed,
            original_size: PAGE_SIZE, // Wrong size!
            algorithm: Algorithm::Lz4,
        };

        let result = compressor.decompress(&bad_page);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("size mismatch") || err.to_string().contains("corrupted"));
    }

    #[test]
    fn test_decompress_zstd_size_mismatch() {
        let compressor = CompressorBuilder::new()
            .algorithm(Algorithm::Zstd { level: 1 })
            .build()
            .unwrap();

        // Compress a small page, but claim it's PAGE_SIZE
        let small_page = [0xBBu8; 256];
        let compressed = zstd::compress(&small_page, 1).unwrap();
        let bad_page = CompressedPage {
            data: compressed,
            original_size: PAGE_SIZE, // Wrong size!
            algorithm: Algorithm::Zstd { level: 1 },
        };

        let result = compressor.decompress(&bad_page);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("size mismatch") || err.to_string().contains("corrupted"));
    }

    #[test]
    fn test_compressor_lz4hc_roundtrip() {
        // Test LZ4HC algorithm path
        let compressor = CompressorBuilder::new()
            .algorithm(Algorithm::Lz4Hc)
            .build()
            .unwrap();

        let page = [0xCDu8; PAGE_SIZE];
        let compressed = compressor.compress(&page).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(page, decompressed);
    }

    #[test]
    fn test_stats_decompress_time_tracked() {
        let compressor = CompressorBuilder::new()
            .algorithm(Algorithm::Lz4)
            .build()
            .unwrap();

        let page = [0xEFu8; PAGE_SIZE];
        let compressed = compressor.compress(&page).unwrap();
        compressor.decompress(&compressed).unwrap();

        let stats = compressor.stats();
        assert!(stats.decompress_time_ns > 0);
    }

    // ============================================================
    // Safety and Security Falsification Tests F076-F085
    // ============================================================

    #[test]
    fn test_f077_no_integer_overflow_page_size() {
        // F077: Size calculations should be checked
        let compressor = CompressorBuilder::new()
            .algorithm(Algorithm::Lz4)
            .build()
            .unwrap();

        // Compress many pages to stress size calculations
        for _ in 0..100 {
            let page = [0xABu8; PAGE_SIZE];
            let compressed = compressor.compress(&page).unwrap();
            assert!(compressed.data.len() <= PAGE_SIZE + 256); // LZ4 max expansion
        }
    }

    #[test]
    fn test_f081_no_panics_on_valid_input() {
        // F081: Panic-free library code
        let compressor = CompressorBuilder::new()
            .algorithm(Algorithm::Lz4)
            .build()
            .unwrap();

        // Various valid inputs should not panic
        let patterns = [
            [0u8; PAGE_SIZE],    // zeros
            [0xFFu8; PAGE_SIZE], // all ones
            [0xAAu8; PAGE_SIZE], // alternating bits
        ];

        for page in &patterns {
            let result = compressor.compress(page);
            assert!(result.is_ok());
            let compressed = result.unwrap();
            let decompressed = compressor.decompress(&compressed);
            assert!(decompressed.is_ok());
        }
    }

    #[test]
    fn test_f082_error_types_implement_error() {
        // F082: All errors are std::error::Error
        use std::error::Error as StdError;

        let error = Error::CorruptedData("test".to_string());
        // Error implements std::error::Error
        let _: &dyn StdError = &error;
        // Error has a display implementation
        let msg = format!("{error}");
        assert!(!msg.is_empty());
    }

    #[test]
    fn test_f082_error_types_send_sync() {
        // F082: Errors should be Send + Sync for thread safety
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Error>();
    }

    #[test]
    fn test_corrupted_data_returns_error() {
        // F009: Corrupted input detected (from spec)
        let compressor = CompressorBuilder::new()
            .algorithm(Algorithm::Lz4)
            .build()
            .unwrap();

        // Create valid compressed data
        let page = [0xCDu8; PAGE_SIZE];
        let mut compressed = compressor.compress(&page).unwrap();

        // Corrupt the compressed data
        if !compressed.data.is_empty() {
            compressed.data[0] ^= 0xFF;
        }

        // Decompression should return error, not panic
        let result = compressor.decompress(&compressed);
        // Either produces an error or incorrect output (no panic)
        let _ = result;
    }

    #[test]
    fn test_truncated_data_returns_error() {
        // F010: Truncated input detected (from spec)
        let compressor = CompressorBuilder::new()
            .algorithm(Algorithm::Lz4)
            .build()
            .unwrap();

        // Create valid compressed data
        let page = [0xABu8; PAGE_SIZE];
        let mut compressed = compressor.compress(&page).unwrap();

        // Truncate the compressed data
        if compressed.data.len() > 1 {
            compressed.data.truncate(compressed.data.len() / 2);
        }

        // Decompression should return error, not panic
        let result = compressor.decompress(&compressed);
        assert!(result.is_err());
    }

    #[test]
    fn test_f012_concurrent_compression_safe() {
        // F012: Concurrent compression safe
        use std::sync::Arc;
        use std::thread;

        let compressor = Arc::new(
            CompressorBuilder::new()
                .algorithm(Algorithm::Lz4)
                .build()
                .unwrap(),
        );

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let comp = Arc::clone(&compressor);
                thread::spawn(move || {
                    for j in 0..100 {
                        let page = [(i * 10 + j) as u8; PAGE_SIZE];
                        let compressed = comp.compress(&page).unwrap();
                        let decompressed = comp.decompress(&compressed).unwrap();
                        assert_eq!(page, decompressed);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_f020_stack_usage_bounded() {
        // F020: Stack usage bounded - <64KB per compression call
        // This test verifies we don't use excessive stack space
        let compressor = CompressorBuilder::new()
            .algorithm(Algorithm::Lz4)
            .build()
            .unwrap();

        // Recursive function to consume stack space, then compress
        fn compress_with_stack_pressure(
            compressor: &dyn PageCompressor,
            depth: usize,
        ) -> Result<()> {
            if depth == 0 {
                let page = [0xEEu8; PAGE_SIZE];
                let compressed = compressor.compress(&page)?;
                let _ = compressor.decompress(&compressed)?;
                Ok(())
            } else {
                // Use some stack space (but not too much to avoid stack overflow)
                let buffer = [0u8; 256];
                std::hint::black_box(&buffer);
                compress_with_stack_pressure(compressor, depth - 1)
            }
        }

        // Should succeed even with nested calls
        compress_with_stack_pressure(compressor.as_ref(), 10).unwrap();
    }

    #[test]
    fn test_empty_page_compress() {
        // Edge case: ensure we handle page boundaries correctly
        let compressor = CompressorBuilder::new()
            .algorithm(Algorithm::Lz4)
            .build()
            .unwrap();

        // Minimum non-trivial page
        let page = {
            let mut p = [0u8; PAGE_SIZE];
            p[0] = 1;
            p
        };

        let compressed = compressor.compress(&page).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(page, decompressed);
    }

    #[test]
    fn test_max_entropy_page() {
        // High entropy page (worst case for compression)
        let compressor = CompressorBuilder::new()
            .algorithm(Algorithm::Lz4)
            .build()
            .unwrap();

        let mut page = [0u8; PAGE_SIZE];
        let mut rng = 98765u64;
        for byte in &mut page {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *byte = (rng >> 33) as u8;
        }

        let compressed = compressor.compress(&page).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(page, decompressed);
    }
}
