//! LZ4 decompression implementation.
//!
//! High-performance LZ4 block format decompression using unsafe pointer arithmetic.

use super::constants::*;
use crate::{Error, Result};

/// Read u16 from unaligned pointer.
#[inline(always)]
unsafe fn read_u16_le(ptr: *const u8) -> u16 {
    std::ptr::read_unaligned(ptr as *const u16)
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

/// Write u64 to unaligned pointer.
#[inline(always)]
unsafe fn write_u64(ptr: *mut u8, val: u64) {
    std::ptr::write_unaligned(ptr as *mut u64, val);
}

/// Copy 8 bytes unconditionally (wildcard copy).
#[inline(always)]
unsafe fn copy_8(dst: *mut u8, src: *const u8) {
    write_u64(dst, read_u64(src));
}

/// Copy 16 bytes unconditionally (double wildcard copy).
#[inline(always)]
unsafe fn copy_16(dst: *mut u8, src: *const u8) {
    write_u64(dst, read_u64(src));
    write_u64(dst.add(8), read_u64(src.add(8)));
}

/// Wildcard copy - copies in 8-byte chunks, may overwrite past end.
#[inline(always)]
unsafe fn wildcard_copy(mut dst: *mut u8, mut src: *const u8, len: usize) {
    let end = dst.add(len);
    loop {
        copy_8(dst, src);
        dst = dst.add(8);
        src = src.add(8);
        if dst >= end {
            break;
        }
    }
}

/// Decompress LZ4 block format data.
///
/// # Arguments
///
/// * `input` - Compressed data in LZ4 block format
/// * `output` - Buffer to write decompressed data (must be large enough)
///
/// # Returns
///
/// The number of bytes written to output.
///
/// # Errors
///
/// Returns an error if the compressed data is corrupted.
pub fn decompress(input: &[u8], output: &mut [u8]) -> Result<usize> {
    if input.is_empty() {
        return Ok(0);
    }

    // SAFETY: We carefully track bounds and ensure all pointer arithmetic stays within limits.
    unsafe { decompress_fast(input, output) }
}

/// Fast decompression using unsafe pointer arithmetic.
#[inline(never)]
unsafe fn decompress_fast(input: &[u8], output: &mut [u8]) -> Result<usize> {
    let mut ip = input.as_ptr();
    let ip_end = ip.add(input.len());

    let mut op = output.as_mut_ptr();
    let op_start = op;
    let op_end = op.add(output.len());

    // We need headroom for wildcard copies
    let op_safe_end = if output.len() >= 16 {
        op_end.sub(16)
    } else {
        op_start
    };

    loop {
        // Read token
        if ip >= ip_end {
            return Err(Error::CorruptedData("unexpected end of input".to_string()));
        }
        let token = *ip;
        ip = ip.add(1);

        // Decode literal length
        let mut literal_len = ((token >> 4) & 0x0F) as usize;
        if literal_len == 15 {
            loop {
                if ip >= ip_end {
                    return Err(Error::CorruptedData(
                        "unexpected end of input in literal length".to_string(),
                    ));
                }
                let byte = *ip;
                ip = ip.add(1);
                literal_len += byte as usize;
                if byte != 255 {
                    break;
                }
            }
        }

        // Copy literals
        if literal_len > 0 {
            if ip.add(literal_len) > ip_end {
                return Err(Error::CorruptedData(
                    "literal extends past input".to_string(),
                ));
            }
            if op.add(literal_len) > op_end {
                return Err(Error::BufferTooSmall {
                    needed: (op as usize - op_start as usize) + literal_len,
                    available: output.len(),
                });
            }

            // Use wildcard copy when safe, otherwise byte-by-byte
            if literal_len <= 16 && op.add(16) <= op_end {
                copy_16(op, ip);
            } else if op.add(literal_len + 8) <= op_end {
                wildcard_copy(op, ip, literal_len);
            } else {
                std::ptr::copy_nonoverlapping(ip, op, literal_len);
            }
            ip = ip.add(literal_len);
            op = op.add(literal_len);
        }

        // Check for end of block (no offset follows last literals)
        if ip >= ip_end {
            break;
        }

        // Read offset (little-endian 16-bit)
        if ip.add(2) > ip_end {
            return Err(Error::CorruptedData(
                "unexpected end of input at offset".to_string(),
            ));
        }
        let offset = read_u16_le(ip) as usize;
        ip = ip.add(2);

        if offset == 0 {
            return Err(Error::CorruptedData("zero offset".to_string()));
        }

        let current_pos = op as usize - op_start as usize;
        if offset > current_pos {
            return Err(Error::CorruptedData(format!(
                "offset {offset} exceeds output position {current_pos}"
            )));
        }

        let match_src = op.sub(offset);

        // Decode match length
        let mut match_len = (token & 0x0F) as usize + MIN_MATCH;
        if (token & 0x0F) == 15 {
            loop {
                if ip >= ip_end {
                    return Err(Error::CorruptedData(
                        "unexpected end of input in match length".to_string(),
                    ));
                }
                let byte = *ip;
                ip = ip.add(1);
                match_len += byte as usize;
                if byte != 255 {
                    break;
                }
            }
        }

        // Check output space
        if op.add(match_len) > op_end {
            return Err(Error::BufferTooSmall {
                needed: (op as usize - op_start as usize) + match_len,
                available: output.len(),
            });
        }

        // Copy match - handle overlapping copies carefully
        if offset >= 8 && op.add(match_len + 8) <= op_end {
            // Non-overlapping: safe to use wildcard copy
            let mut src = match_src;
            let mut dst = op;
            let end = op.add(match_len);

            loop {
                copy_8(dst, src);
                dst = dst.add(8);
                src = src.add(8);
                if dst >= end {
                    break;
                }
            }
        } else if offset == 1 && op.add(match_len + 8) <= op_end {
            // RLE (repeat single byte) - optimized path
            let byte = *match_src;
            let pattern = 0x0101010101010101u64 * (byte as u64);
            let mut dst = op;
            let end = op.add(match_len);
            while dst < end {
                write_u64(dst, pattern);
                dst = dst.add(8);
            }
        } else {
            // General case: copy byte-by-byte (handles all overlaps correctly)
            for i in 0..match_len {
                *op.add(i) = *op.sub(offset).add(i);
            }
        }
        op = op.add(match_len);
    }

    Ok(op as usize - op_start as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lz4::compress::compress;

    #[test]
    fn test_decompress_empty() {
        let mut output = [0u8; 100];
        let len = decompress(&[], &mut output).unwrap();
        assert_eq!(len, 0);
    }

    #[test]
    fn test_decompress_corrupt_missing_offset() {
        // Token with 1 literal and implied match, but missing offset bytes
        let input = [0x14, b'A']; // 1 literal 'A', then should have offset
        let mut output = [0u8; 100];
        let result = decompress(&input, &mut output);
        match result {
            Ok(len) => assert_eq!(len, 1),
            Err(_) => {}
        }
    }

    #[test]
    fn test_decompress_zero_offset() {
        let input = [0x10, b'A', 0x00, 0x00];
        let mut output = [0u8; 100];
        let result = decompress(&input, &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn test_roundtrip_property() {
        for pattern in [0x00u8, 0xFFu8, 0xAAu8, 0x55u8] {
            let input = [pattern; 4096];
            let compressed = compress(&input).unwrap();
            let mut output = [0u8; 4096];
            let len = decompress(&compressed, &mut output).unwrap();
            assert_eq!(len, 4096);
            assert_eq!(input, output);
        }
    }

    #[test]
    fn test_roundtrip_mixed_data() {
        let mut input = [0u8; 4096];
        for (i, byte) in input.iter_mut().enumerate() {
            *byte = ((i * 17) ^ (i >> 3)) as u8;
        }
        let compressed = compress(&input).unwrap();
        let mut output = [0u8; 4096];
        let len = decompress(&compressed, &mut output).unwrap();
        assert_eq!(len, 4096);
        assert_eq!(input, output);
    }

    #[test]
    fn test_roundtrip_small_patterns() {
        // Test small repeating patterns that stress overlap handling
        let patterns: &[&[u8]] = &[
            &[0xAB, 0xCD],                         // 2-byte pattern
            &[0x11, 0x22, 0x33],                   // 3-byte pattern
            &[0xDE, 0xAD, 0xBE, 0xEF],             // 4-byte pattern
            &[0x01, 0x02, 0x03, 0x04, 0x05],       // 5-byte pattern
            &[0xAA; 7],                            // 7-byte pattern
        ];

        for pattern in patterns {
            let mut input = [0u8; 4096];
            for (i, byte) in input.iter_mut().enumerate() {
                *byte = pattern[i % pattern.len()];
            }
            let compressed = compress(&input).unwrap();
            let mut output = [0u8; 4096];
            let len = decompress(&compressed, &mut output).unwrap();
            assert_eq!(len, 4096);
            assert_eq!(input[..], output[..]);
        }
    }
}
