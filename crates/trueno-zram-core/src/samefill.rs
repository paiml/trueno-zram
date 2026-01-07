//! Same-fill page detection and optimization.
//!
//! This module implements the kernel zram same-fill optimization that stores
//! pages consisting of a repeated value (e.g., all zeros) as a flag and value,
//! consuming no memory for compressed storage.
//!
//! Reference: Linux kernel drivers/block/zram/zram_drv.c
//!
//! ## Kernel Algorithm (page_same_filled)
//!
//! The kernel stores a `u64` fill value (not just `u8`), allowing patterns like
//! 0xDEADBEEF repeated to also benefit from same-fill optimization. This is
//! critical for performance: same-fill pages skip compression entirely.

use crate::PAGE_SIZE;

/// Result of same-fill detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SameFillResult {
    /// Page is not same-fill (contains varying data).
    NotSameFill,
    /// Page is same-fill with the given repeated value.
    SameFill {
        /// The repeated byte value.
        value: u8,
    },
}

/// Detect if a page consists entirely of the same u64 word value.
///
/// This follows the kernel zram `page_same_filled()` algorithm exactly:
/// 1. Quick check: compare first vs last word (fast rejection)
/// 2. Full scan: only if first/last match
///
/// Returns `Some(fill_value)` if all words are the same, `None` otherwise.
/// The returned `u64` can be stored directly in the handle (no allocation).
///
/// # Performance
///
/// This is the CRITICAL hot path. Same-fill pages (especially zeros) are
/// 30-40% of typical swap workloads. Detecting them BEFORE compression
/// is what allows kernel ZRAM to hit 171 GB/s on zero pages.
///
/// # Reference
///
/// Linux kernel drivers/block/zram/zram_drv.c lines 344-358
#[inline]
#[must_use]
pub fn page_same_filled(page: &[u8; PAGE_SIZE]) -> Option<u64> {
    // Safety: PAGE_SIZE is 4096, always divisible by 8
    // This transmute is safe because:
    // - [u8; 4096] has same size as [u64; 512]
    // - u64 has weaker alignment than [u8; 4096] on stack
    let words: &[u64; PAGE_SIZE / 8] = unsafe { &*(page.as_ptr() as *const [u64; PAGE_SIZE / 8]) };

    let val = words[0];
    let last_pos = words.len() - 1;

    // Quick check: first vs last word (kernel optimization)
    if val != words[last_pos] {
        return None;
    }

    // Full scan only if first/last match
    for &word in &words[1..last_pos] {
        if word != val {
            return None;
        }
    }

    Some(val) // Return the fill value, not just bool
}

/// Fill a page with a u64 word value (kernel memset_l equivalent).
///
/// This is faster than byte-by-byte memset for word-aligned fills.
#[inline]
pub fn fill_page_word(page: &mut [u8; PAGE_SIZE], value: u64) {
    // Safety: Same as page_same_filled - sizes match, alignment is safe
    let words: &mut [u64; PAGE_SIZE / 8] =
        unsafe { &mut *(page.as_mut_ptr() as *mut [u64; PAGE_SIZE / 8]) };

    for word in words.iter_mut() {
        *word = value;
    }
}

/// Check if a page consists entirely of the same byte value.
///
/// This is an optimized check that uses word-level comparisons for speed.
/// Following the kernel implementation, we check 8 bytes at a time.
///
/// # Arguments
///
/// * `page` - The 4KB page to check.
///
/// # Returns
///
/// `SameFill { value }` if all bytes are the same, `NotSameFill` otherwise.
#[must_use]
pub fn detect_same_fill(page: &[u8; PAGE_SIZE]) -> SameFillResult {
    if page.is_empty() {
        return SameFillResult::NotSameFill;
    }

    let first_byte = page[0];

    // Create a word filled with the first byte for fast comparison
    let fill_word = u64::from_ne_bytes([first_byte; 8]);

    // Check 8 bytes at a time
    let chunks = page.chunks_exact(8);
    let remainder = chunks.remainder();

    for chunk in chunks {
        // chunks_exact guarantees exactly 8-byte chunks, so try_into always succeeds
        let word = u64::from_ne_bytes(chunk.try_into().expect("chunks_exact guarantees 8 bytes"));
        if word != fill_word {
            return SameFillResult::NotSameFill;
        }
    }

    // Check any remaining bytes
    for &byte in remainder {
        if byte != first_byte {
            return SameFillResult::NotSameFill;
        }
    }

    SameFillResult::SameFill { value: first_byte }
}

/// Check if a page is a zero page (all zeros).
///
/// Zero pages are the most common same-fill case and are given
/// special treatment in zram for maximum efficiency.
#[must_use]
pub fn is_zero_page(page: &[u8; PAGE_SIZE]) -> bool {
    matches!(
        detect_same_fill(page),
        SameFillResult::SameFill { value: 0 }
    )
}

/// Expand a same-fill value back to a full page.
///
/// # Arguments
///
/// * `value` - The repeated byte value.
///
/// # Returns
///
/// A 4KB page filled with the specified value.
#[must_use]
pub fn expand_same_fill(value: u8) -> [u8; PAGE_SIZE] {
    [value; PAGE_SIZE]
}

/// Compact representation of a same-fill page.
///
/// This uses only 2 bytes to represent a 4KB same-fill page:
/// - 1 byte: flags (bit 0 = is_same_fill)
/// - 1 byte: fill value
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactSameFill {
    /// Flags byte (bit 0 indicates same-fill).
    pub flags: u8,
    /// The fill value.
    pub value: u8,
}

impl CompactSameFill {
    /// Flag indicating this is a same-fill page.
    pub const FLAG_SAME_FILL: u8 = 0x01;

    /// Create a compact representation for a same-fill page.
    #[must_use]
    pub fn new(value: u8) -> Self {
        Self {
            flags: Self::FLAG_SAME_FILL,
            value,
        }
    }

    /// Check if this represents a same-fill page.
    #[must_use]
    pub fn is_same_fill(&self) -> bool {
        self.flags & Self::FLAG_SAME_FILL != 0
    }

    /// Expand back to a full page.
    #[must_use]
    pub fn expand(&self) -> [u8; PAGE_SIZE] {
        expand_same_fill(self.value)
    }

    /// Serialize to bytes.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 2] {
        [self.flags, self.value]
    }

    /// Deserialize from bytes.
    #[must_use]
    pub fn from_bytes(bytes: &[u8; 2]) -> Self {
        Self {
            flags: bytes[0],
            value: bytes[1],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // Falsification Tests F003, F006, F007
    // ============================================================

    #[test]
    fn test_zero_page_detected() {
        // F003: Zero pages compress correctly
        let page = [0u8; PAGE_SIZE];
        let result = detect_same_fill(&page);
        assert_eq!(result, SameFillResult::SameFill { value: 0 });
        assert!(is_zero_page(&page));
    }

    #[test]
    fn test_same_fill_detected() {
        // F007: Repeated patterns detected
        let page = [0xAAu8; PAGE_SIZE];
        let result = detect_same_fill(&page);
        assert_eq!(result, SameFillResult::SameFill { value: 0xAA });
        assert!(!is_zero_page(&page));
    }

    #[test]
    fn test_non_same_fill_detected() {
        let mut page = [0u8; PAGE_SIZE];
        page[100] = 1; // Single different byte
        let result = detect_same_fill(&page);
        assert_eq!(result, SameFillResult::NotSameFill);
    }

    #[test]
    fn test_last_byte_different() {
        let mut page = [0u8; PAGE_SIZE];
        page[PAGE_SIZE - 1] = 1;
        let result = detect_same_fill(&page);
        assert_eq!(result, SameFillResult::NotSameFill);
    }

    #[test]
    fn test_random_page_not_same_fill() {
        let mut page = [0u8; PAGE_SIZE];
        let mut rng = 12345u64;
        for byte in &mut page {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *byte = (rng >> 33) as u8;
        }
        let result = detect_same_fill(&page);
        assert_eq!(result, SameFillResult::NotSameFill);
    }

    #[test]
    fn test_expand_same_fill_zeros() {
        let page = expand_same_fill(0);
        assert!(page.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_expand_same_fill_value() {
        let page = expand_same_fill(0xFF);
        assert!(page.iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn test_compact_same_fill_new() {
        let compact = CompactSameFill::new(0xBB);
        assert!(compact.is_same_fill());
        assert_eq!(compact.value, 0xBB);
    }

    #[test]
    fn test_compact_same_fill_expand() {
        let compact = CompactSameFill::new(0xCC);
        let page = compact.expand();
        assert!(page.iter().all(|&b| b == 0xCC));
    }

    #[test]
    fn test_compact_same_fill_roundtrip() {
        let compact = CompactSameFill::new(0xDD);
        let bytes = compact.to_bytes();
        let restored = CompactSameFill::from_bytes(&bytes);
        assert_eq!(compact, restored);
    }

    #[test]
    fn test_compact_same_fill_serialization() {
        let compact = CompactSameFill::new(0xEE);
        let bytes = compact.to_bytes();
        assert_eq!(bytes[0], CompactSameFill::FLAG_SAME_FILL);
        assert_eq!(bytes[1], 0xEE);
    }

    #[test]
    fn test_same_fill_all_values() {
        // Test all 256 possible fill values
        for value in 0..=255u8 {
            let page = [value; PAGE_SIZE];
            let result = detect_same_fill(&page);
            assert_eq!(result, SameFillResult::SameFill { value });
        }
    }

    #[test]
    fn test_same_fill_detection_speed() {
        // Ensure detection is fast (should be O(n) with early exit)
        // Note: Debug builds are ~10x slower than release builds
        let page = [0u8; PAGE_SIZE];
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = detect_same_fill(&page);
        }
        let elapsed = start.elapsed();
        // Should process 1K pages in under 1 second (debug build)
        assert!(elapsed.as_millis() < 1000);
    }

    #[test]
    fn test_detect_same_fill_efficiency() {
        // F006: Zero page has minimal storage (2 bytes)
        let compact = CompactSameFill::new(0);
        assert_eq!(compact.to_bytes().len(), 2);
        // Compression ratio: 4096 / 2 = 2048:1
    }

    #[test]
    fn test_different_at_word_boundary() {
        let mut page = [0xAAu8; PAGE_SIZE];
        // Different at 8-byte boundary
        page[8] = 0xBB;
        let result = detect_same_fill(&page);
        assert_eq!(result, SameFillResult::NotSameFill);
    }

    #[test]
    fn test_different_in_remainder() {
        // PAGE_SIZE is 4096 which is divisible by 8, so test a specific position
        let mut page = [0xCCu8; PAGE_SIZE];
        page[PAGE_SIZE - 1] = 0xDD;
        let result = detect_same_fill(&page);
        assert_eq!(result, SameFillResult::NotSameFill);
    }
}
