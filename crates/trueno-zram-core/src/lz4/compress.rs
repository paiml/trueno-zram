//! LZ4 compression implementation.
//!
//! High-performance LZ4 block format compression using unsafe pointer arithmetic.

use super::constants::*;
use crate::{Error, Result};

/// Hash table size (64KB = 16384 entries of 4 bytes each).
const HASH_LOG: usize = 14;
const HASH_SIZE: usize = 1 << HASH_LOG;
const HASH_MASK: usize = HASH_SIZE - 1;

/// Acceleration factor for faster skipping.
const SKIP_TRIGGER: usize = 6;

/// Hash table for match finding.
struct HashTable {
    table: [u32; HASH_SIZE],
}

impl HashTable {
    #[inline]
    fn new() -> Self {
        Self {
            table: [0; HASH_SIZE],
        }
    }

    /// Fast hash using Knuth multiplicative method.
    #[inline(always)]
    fn hash(val: u32) -> usize {
        ((val.wrapping_mul(2654435761)) >> (32 - HASH_LOG)) as usize & HASH_MASK
    }
}

/// Read u32 from unaligned pointer.
#[inline(always)]
unsafe fn read_u32(ptr: *const u8) -> u32 {
    std::ptr::read_unaligned(ptr as *const u32)
}

/// Read u64 from unaligned pointer.
#[inline(always)]
unsafe fn read_u64(ptr: *const u8) -> u64 {
    std::ptr::read_unaligned(ptr as *const u64)
}

/// Write u16 to unaligned pointer.
#[inline(always)]
unsafe fn write_u16(ptr: *mut u8, val: u16) {
    std::ptr::write_unaligned(ptr as *mut u16, val);
}

/// Copy 8 bytes (wildcard copy).
#[inline(always)]
unsafe fn copy_8(dst: *mut u8, src: *const u8) {
    std::ptr::write_unaligned(dst as *mut u64, std::ptr::read_unaligned(src as *const u64));
}

/// Count matching bytes using 64-bit comparisons.
#[inline(always)]
unsafe fn count_match(mut p1: *const u8, mut p2: *const u8, p2_end: *const u8) -> usize {
    let start = p2;

    // Compare 8 bytes at a time
    while p2.add(8) <= p2_end {
        let diff = read_u64(p1) ^ read_u64(p2);
        if diff != 0 {
            // Found difference - count trailing zeros to find exact position
            return (p2 as usize - start as usize) + (diff.trailing_zeros() as usize >> 3);
        }
        p1 = p1.add(8);
        p2 = p2.add(8);
    }

    // Handle remaining bytes
    while p2 < p2_end && *p1 == *p2 {
        p1 = p1.add(1);
        p2 = p2.add(1);
    }

    p2 as usize - start as usize
}

/// Compress input data using LZ4 block format.
///
/// # Errors
///
/// Returns an error if compression fails.
pub fn compress(input: &[u8]) -> Result<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    // Worst case: incompressible data expands slightly
    let max_output = input.len() + (input.len() / 255) + 16;
    let mut output = Vec::with_capacity(max_output);

    // SAFETY: We carefully track bounds and ensure all pointer arithmetic stays within bounds.
    unsafe {
        compress_fast(input, &mut output);
    }

    Ok(output)
}

/// Fast compression using unsafe pointer arithmetic.
///
/// # Safety
///
/// Caller must ensure input is valid and output has sufficient capacity.
#[inline(never)]
unsafe fn compress_fast(input: &[u8], output: &mut Vec<u8>) {
    let input_len = input.len();

    if input_len < MF_LIMIT {
        // Too short for compression, emit as literals
        emit_last_literals(output, input.as_ptr(), input_len);
        return;
    }

    let mut hash_table = HashTable::new();

    let base = input.as_ptr();
    let mut ip = base; // Current input position
    let mut anchor = base; // Start of literals

    let input_end = base.add(input_len);
    let match_limit = input_end.sub(LAST_LITERALS);
    let mf_limit = input_end.sub(MF_LIMIT);

    // Start after first byte
    ip = ip.add(1);

    let mut acceleration = 1usize;

    loop {
        let mut match_ptr: *const u8;
        let mut token_pos: *const u8;

        // Find a match
        loop {
            let step = acceleration >> SKIP_TRIGGER;
            let next_ip = ip.add(step + 1);

            if next_ip > mf_limit {
                // Emit remaining literals
                emit_last_literals(output, anchor, input_end as usize - anchor as usize);
                return;
            }

            let sequence = read_u32(ip);
            let hash = HashTable::hash(sequence);
            match_ptr = base.add(hash_table.table[hash] as usize);
            hash_table.table[hash] = (ip as usize - base as usize) as u32;

            // Check if match is valid
            if read_u32(match_ptr) == sequence
                && ip as usize - match_ptr as usize <= MAX_DISTANCE
                && match_ptr >= base
            {
                token_pos = ip;
                break;
            }

            ip = next_ip;
            acceleration += 1;
        }

        // Found a match! Extend backward if possible
        while ip > anchor && match_ptr > base && *ip.sub(1) == *match_ptr.sub(1) {
            ip = ip.sub(1);
            match_ptr = match_ptr.sub(1);
        }

        // Calculate literal length
        let literal_len = ip as usize - anchor as usize;

        // Reserve space in output
        let out_start = output.len();
        output.reserve(literal_len + 16);

        // Emit token
        let token_idx = output.len();
        output.push(0); // Placeholder for token

        // Emit extended literal length
        if literal_len >= 15 {
            emit_length(output, literal_len - 15);
        }

        // Copy literals with wildcard copy
        let lit_out_start = output.len();
        output.set_len(output.len() + literal_len);
        let mut lit_src = anchor;
        let mut lit_dst = output.as_mut_ptr().add(lit_out_start);
        let lit_end = anchor.add(literal_len);

        while lit_src.add(8) <= lit_end {
            copy_8(lit_dst, lit_src);
            lit_src = lit_src.add(8);
            lit_dst = lit_dst.add(8);
        }
        while lit_src < lit_end {
            *lit_dst = *lit_src;
            lit_src = lit_src.add(1);
            lit_dst = lit_dst.add(1);
        }

        // Emit offset
        let offset = ip as usize - match_ptr as usize;
        output.reserve(2);
        let off_idx = output.len();
        output.set_len(output.len() + 2);
        write_u16(output.as_mut_ptr().add(off_idx), offset as u16);

        // Count match length
        ip = ip.add(MIN_MATCH);
        match_ptr = match_ptr.add(MIN_MATCH);
        let match_len = count_match(match_ptr, ip, match_limit);
        ip = ip.add(match_len);

        // Write token
        let ml_token = match_len.min(15) as u8;
        let lit_token = literal_len.min(15) as u8;
        output[token_idx] = (lit_token << 4) | ml_token;

        // Emit extended match length
        if match_len >= 15 {
            emit_length(output, match_len - 15);
        }

        // Update anchor
        anchor = ip;

        // Add intermediate positions to hash table
        if ip < mf_limit {
            hash_table.table[HashTable::hash(read_u32(ip.sub(2)))] =
                (ip.sub(2) as usize - base as usize) as u32;
        }

        // Reset acceleration
        acceleration = 1;
    }
}

/// Emit variable-length encoding.
#[inline(always)]
fn emit_length(output: &mut Vec<u8>, mut len: usize) {
    while len >= 255 {
        output.push(255);
        len -= 255;
    }
    output.push(len as u8);
}

/// Emit final literals (no match follows).
#[inline(always)]
unsafe fn emit_last_literals(output: &mut Vec<u8>, src: *const u8, len: usize) {
    // Token
    let token = (len.min(15) as u8) << 4;
    output.push(token);

    // Extended length
    if len >= 15 {
        emit_length(output, len - 15);
    }

    // Copy literals
    let start = output.len();
    output.reserve(len);
    output.set_len(output.len() + len);

    std::ptr::copy_nonoverlapping(src, output.as_mut_ptr().add(start), len);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_empty() {
        let result = compress(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_compress_small() {
        let input = b"Hello";
        let result = compress(input).unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_hash_distribution() {
        let h1 = HashTable::hash(0x12345678);
        let h2 = HashTable::hash(0x87654321);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_compress_repetitive() {
        let input = [0xAA; 4096];
        let result = compress(&input).unwrap();
        // Should compress well
        assert!(result.len() < input.len() / 10);
    }

    #[test]
    fn test_compress_incompressible() {
        // Random-ish data
        let input: Vec<u8> = (0..4096).map(|i| (i * 17 + i / 3) as u8).collect();
        let result = compress(&input).unwrap();
        // Should not crash, may be larger than input
        assert!(!result.is_empty());
    }
}
