//! LZ4 compression implementation.
//!
//! This module provides a pure Rust implementation of the LZ4 block format
//! as specified in <https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md>.

mod compress;
mod decompress;

#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "x86_64")]
pub mod avx512;
#[cfg(target_arch = "aarch64")]
pub mod neon;

pub use compress::compress;
pub use decompress::decompress;

use crate::{Result, PAGE_SIZE};

/// Decompress with automatic SIMD dispatch.
///
/// Selects the best available SIMD implementation at runtime:
/// - AVX-512 on `x86_64` with AVX-512 support
/// - AVX2 on `x86_64` with AVX2 support
/// - NEON on `AArch64`
/// - Scalar fallback on other platforms
pub fn decompress_simd(input: &[u8], output: &mut [u8; PAGE_SIZE]) -> Result<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        // Prefer AVX-512 if available (≥5 GB/s target)
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
        {
            // SAFETY: We just checked that AVX-512 is available
            return unsafe { avx512::decompress_avx512(input, output) };
        }
        // Fall back to AVX2 (≥4 GB/s target)
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: We just checked that AVX2 is available
            return unsafe { avx2::decompress_avx2(input, output) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Use NEON on ARM (≥4 GB/s target)
        // SAFETY: NEON is always available on AArch64
        return unsafe { neon::decompress_neon(input, output) };
    }

    // Fallback to scalar
    #[allow(unreachable_code)]
    decompress(input, output)
}

/// Compress with automatic SIMD dispatch.
///
/// Note: Compression is hash-table bound and benefits less from SIMD than decompression.
/// All backends currently use the optimized scalar implementation.
pub fn compress_simd(input: &[u8; PAGE_SIZE]) -> Result<Vec<u8>> {
    compress(input)
}

/// LZ4 block format constants.
pub mod constants {
    /// Minimum match length.
    pub const MIN_MATCH: usize = 4;
    /// Maximum match length that can be encoded in token.
    pub const ML_MASK: u8 = 0x0F;
    /// Literal length mask in token.
    pub const LL_MASK: u8 = 0xF0;
    /// Maximum literal run before extension bytes.
    pub const RUN_MASK: u8 = 15;
    /// Offset for last literals (must leave room at end).
    pub const LAST_LITERALS: usize = 5;
    /// Minimum length for last match (safety margin).
    pub const MF_LIMIT: usize = 12;
    /// Hash table size (power of 2).
    pub const HASH_SIZE_U32: usize = 1 << 14; // 16384 entries
    /// Maximum offset for matches.
    pub const MAX_DISTANCE: usize = 65535;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PAGE_SIZE;

    #[test]
    fn test_roundtrip_zeros() {
        let input = [0u8; PAGE_SIZE];
        let compressed = compress(&input).unwrap();
        let mut output = [0u8; PAGE_SIZE];
        let len = decompress(&compressed, &mut output).unwrap();
        assert_eq!(len, PAGE_SIZE);
        assert_eq!(input, output);
    }

    #[test]
    fn test_roundtrip_ones() {
        let input = [0xFFu8; PAGE_SIZE];
        let compressed = compress(&input).unwrap();
        let mut output = [0u8; PAGE_SIZE];
        let len = decompress(&compressed, &mut output).unwrap();
        assert_eq!(len, PAGE_SIZE);
        assert_eq!(input, output);
    }

    #[test]
    fn test_roundtrip_sequential() {
        let mut input = [0u8; PAGE_SIZE];
        for (i, byte) in input.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }
        let compressed = compress(&input).unwrap();
        let mut output = [0u8; PAGE_SIZE];
        let len = decompress(&compressed, &mut output).unwrap();
        assert_eq!(len, PAGE_SIZE);
        assert_eq!(input, output);
    }

    #[test]
    fn test_roundtrip_repeating_pattern() {
        let mut input = [0u8; PAGE_SIZE];
        let pattern = b"ABCD";
        for (i, byte) in input.iter_mut().enumerate() {
            *byte = pattern[i % pattern.len()];
        }
        let compressed = compress(&input).unwrap();
        let mut output = [0u8; PAGE_SIZE];
        let len = decompress(&compressed, &mut output).unwrap();
        assert_eq!(len, PAGE_SIZE);
        assert_eq!(input, output);
    }

    #[test]
    fn test_compression_ratio_zeros() {
        let input = [0u8; PAGE_SIZE];
        let compressed = compress(&input).unwrap();
        // Zero page should compress very well (>10x)
        let ratio = PAGE_SIZE as f64 / compressed.len() as f64;
        assert!(ratio > 10.0, "Expected ratio > 10x, got {ratio:.2}x");
    }

    #[test]
    fn test_compression_ratio_pattern() {
        let mut input = [0u8; PAGE_SIZE];
        let pattern = b"Hello World! ";
        for (i, byte) in input.iter_mut().enumerate() {
            *byte = pattern[i % pattern.len()];
        }
        let compressed = compress(&input).unwrap();
        // Repeating pattern should compress well (>5x)
        let ratio = PAGE_SIZE as f64 / compressed.len() as f64;
        assert!(ratio > 5.0, "Expected ratio > 5x, got {ratio:.2}x");
    }

    #[test]
    fn test_incompressible_data() {
        // Pseudo-random data that is hard to compress
        let mut input = [0u8; PAGE_SIZE];
        let mut state = 12345u64;
        for byte in &mut input {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *byte = (state >> 33) as u8;
        }

        let compressed = compress(&input).unwrap();
        let mut output = [0u8; PAGE_SIZE];
        let len = decompress(&compressed, &mut output).unwrap();
        assert_eq!(len, PAGE_SIZE);
        assert_eq!(input, output);
    }
}
