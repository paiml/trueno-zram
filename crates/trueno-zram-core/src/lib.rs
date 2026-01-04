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

mod error;
pub mod lz4;
mod page;
pub mod simd;
pub mod zram;
pub mod zstd;

pub use error::{Error, Result};
pub use page::{CompressedPage, CompressionStats, PAGE_SIZE};

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Compression algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
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
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }
}
