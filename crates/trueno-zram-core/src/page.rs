//! Page compression types and utilities.

use crate::{Algorithm, Error, Result};

/// Standard memory page size (4KB).
pub const PAGE_SIZE: usize = 4096;

/// A compressed memory page.
#[derive(Debug, Clone)]
pub struct CompressedPage {
    /// Compressed data bytes.
    pub data: Vec<u8>,
    /// Original uncompressed size (always PAGE_SIZE for valid pages).
    pub original_size: usize,
    /// Algorithm used for compression.
    pub algorithm: Algorithm,
}

impl CompressedPage {
    /// Create a new compressed page.
    ///
    /// # Errors
    ///
    /// Returns an error if the original size is not PAGE_SIZE.
    pub fn new(data: Vec<u8>, original_size: usize, algorithm: Algorithm) -> Result<Self> {
        if original_size != PAGE_SIZE {
            return Err(Error::InvalidInput(format!(
                "original_size must be {PAGE_SIZE}, got {original_size}"
            )));
        }

        Ok(Self {
            data,
            original_size,
            algorithm,
        })
    }

    /// Create a compressed page for incompressible data (stored uncompressed).
    #[must_use]
    pub fn uncompressed(data: [u8; PAGE_SIZE]) -> Self {
        Self {
            data: data.to_vec(),
            original_size: PAGE_SIZE,
            algorithm: Algorithm::None,
        }
    }

    /// Get the compression ratio (original / compressed).
    ///
    /// Returns 1.0 if compressed size >= original size.
    #[must_use]
    pub fn ratio(&self) -> f64 {
        if self.data.is_empty() {
            return 1.0;
        }
        self.original_size as f64 / self.data.len() as f64
    }

    /// Check if the data was actually compressed (ratio > 1.0).
    #[must_use]
    pub fn is_compressed(&self) -> bool {
        self.data.len() < self.original_size && !matches!(self.algorithm, Algorithm::None)
    }

    /// Get the space saved by compression in bytes.
    #[must_use]
    pub fn bytes_saved(&self) -> usize {
        self.original_size.saturating_sub(self.data.len())
    }
}

/// Statistics for compression operations.
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    /// Total pages compressed.
    pub pages_compressed: u64,
    /// Total pages that were incompressible.
    pub pages_incompressible: u64,
    /// Total bytes before compression.
    pub bytes_in: u64,
    /// Total bytes after compression.
    pub bytes_out: u64,
    /// Total compression time in nanoseconds.
    pub compress_time_ns: u64,
    /// Total decompression time in nanoseconds.
    pub decompress_time_ns: u64,
}

impl CompressionStats {
    /// Create new empty statistics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the overall compression ratio.
    #[must_use]
    pub fn overall_ratio(&self) -> f64 {
        if self.bytes_out == 0 {
            return 1.0;
        }
        self.bytes_in as f64 / self.bytes_out as f64
    }

    /// Get compression throughput in bytes per second.
    #[must_use]
    pub fn compress_throughput(&self) -> f64 {
        if self.compress_time_ns == 0 {
            return 0.0;
        }
        self.bytes_in as f64 / (self.compress_time_ns as f64 / 1e9)
    }

    /// Get decompression throughput in bytes per second.
    #[must_use]
    pub fn decompress_throughput(&self) -> f64 {
        if self.decompress_time_ns == 0 {
            return 0.0;
        }
        self.bytes_in as f64 / (self.decompress_time_ns as f64 / 1e9)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_size_constant() {
        assert_eq!(PAGE_SIZE, 4096);
    }

    #[test]
    fn test_compressed_page_new_valid() {
        let data = vec![0u8; 100];
        let page = CompressedPage::new(data.clone(), PAGE_SIZE, Algorithm::Lz4).unwrap();
        assert_eq!(page.data, data);
        assert_eq!(page.original_size, PAGE_SIZE);
    }

    #[test]
    fn test_compressed_page_new_invalid_size() {
        let data = vec![0u8; 100];
        let result = CompressedPage::new(data, 1024, Algorithm::Lz4);
        assert!(result.is_err());
    }

    #[test]
    fn test_compressed_page_ratio() {
        let data = vec![0u8; 1024]; // 4x compression
        let page = CompressedPage::new(data, PAGE_SIZE, Algorithm::Lz4).unwrap();
        assert!((page.ratio() - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_compressed_page_is_compressed() {
        let compressed = CompressedPage::new(vec![0u8; 1024], PAGE_SIZE, Algorithm::Lz4).unwrap();
        assert!(compressed.is_compressed());

        let uncompressed = CompressedPage::uncompressed([0u8; PAGE_SIZE]);
        assert!(!uncompressed.is_compressed());
    }

    #[test]
    fn test_compressed_page_bytes_saved() {
        let data = vec![0u8; 1024];
        let page = CompressedPage::new(data, PAGE_SIZE, Algorithm::Lz4).unwrap();
        assert_eq!(page.bytes_saved(), PAGE_SIZE - 1024);
    }

    #[test]
    fn test_compression_stats_default() {
        let stats = CompressionStats::new();
        assert_eq!(stats.pages_compressed, 0);
        assert_eq!(stats.overall_ratio(), 1.0);
    }

    #[test]
    fn test_compression_stats_ratio() {
        let stats = CompressionStats {
            bytes_in: 4096,
            bytes_out: 1024,
            ..Default::default()
        };
        assert!((stats.overall_ratio() - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_compression_stats_throughput() {
        let stats = CompressionStats {
            bytes_in: 1_000_000_000,         // 1GB
            compress_time_ns: 1_000_000_000, // 1 second
            ..Default::default()
        };
        assert!((stats.compress_throughput() - 1e9).abs() < 1.0);
    }
}
