//! ARM NEON accelerated LZ4 implementation.
//!
//! This module provides NEON (128-bit SIMD) optimized LZ4 compression
//! and decompression for AArch64 CPUs.

use crate::{Error, Result, PAGE_SIZE};

/// NEON accelerated LZ4 compression.
///
/// # Safety
///
/// Caller must ensure NEON is available (always true on AArch64).
#[cfg(target_arch = "aarch64")]
pub unsafe fn compress_neon(input: &[u8; PAGE_SIZE]) -> Result<Vec<u8>> {
    // TODO: Implement NEON optimized compression
    super::compress::compress(input)
}

/// NEON accelerated LZ4 decompression.
///
/// # Safety
///
/// Caller must ensure NEON is available (always true on AArch64).
#[cfg(target_arch = "aarch64")]
pub unsafe fn decompress_neon(input: &[u8], output: &mut [u8; PAGE_SIZE]) -> Result<usize> {
    // TODO: Implement NEON optimized decompression
    super::decompress::decompress(input, output)
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_roundtrip() {
        use super::*;

        let input = [0xAAu8; PAGE_SIZE];
        let compressed = unsafe { compress_neon(&input) }.unwrap();
        let mut output = [0u8; PAGE_SIZE];
        let len = unsafe { decompress_neon(&compressed, &mut output) }.unwrap();

        assert_eq!(len, PAGE_SIZE);
        assert_eq!(input, output);
    }
}
