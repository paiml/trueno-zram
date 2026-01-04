//! Zstandard decompression.

use super::{BlockHeader, BlockType, FrameHeader, ZSTD_MAGIC};
use crate::{Error, Result};

/// Decompress Zstandard data.
///
/// # Arguments
///
/// * `input` - Compressed data in Zstandard format
/// * `output` - Buffer for decompressed data
///
/// # Returns
///
/// Number of bytes written to output.
///
/// # Errors
///
/// Returns an error if decompression fails.
pub fn decompress(input: &[u8], output: &mut [u8]) -> Result<usize> {
    if input.is_empty() {
        return Ok(0);
    }

    let mut pos = 0usize;
    let mut out_pos = 0usize;

    // Read frame header
    let header = read_frame_header(input, &mut pos)?;

    // Decompress blocks
    loop {
        let block = read_block_header(input, &mut pos)?;

        match block.block_type {
            BlockType::Raw => {
                // Copy raw data
                if pos + block.block_size as usize > input.len() {
                    return Err(Error::CorruptedData("block extends past input".to_string()));
                }
                if out_pos + block.block_size as usize > output.len() {
                    return Err(Error::BufferTooSmall {
                        needed: out_pos + block.block_size as usize,
                        available: output.len(),
                    });
                }

                output[out_pos..out_pos + block.block_size as usize]
                    .copy_from_slice(&input[pos..pos + block.block_size as usize]);
                pos += block.block_size as usize;
                out_pos += block.block_size as usize;
            }
            BlockType::Rle => {
                // Repeat single byte
                if pos >= input.len() {
                    return Err(Error::CorruptedData("missing RLE byte".to_string()));
                }
                let byte = input[pos];
                pos += 1;

                if out_pos + block.block_size as usize > output.len() {
                    return Err(Error::BufferTooSmall {
                        needed: out_pos + block.block_size as usize,
                        available: output.len(),
                    });
                }

                for i in 0..block.block_size as usize {
                    output[out_pos + i] = byte;
                }
                out_pos += block.block_size as usize;
            }
            BlockType::Compressed => {
                let block_end = pos + block.block_size as usize;
                if block_end > input.len() {
                    return Err(Error::CorruptedData("block extends past input".to_string()));
                }

                out_pos += decompress_block(&input[pos..block_end], &mut output[out_pos..])?;
                pos = block_end;
            }
            BlockType::Reserved => {
                return Err(Error::CorruptedData("reserved block type".to_string()));
            }
        }

        if block.last_block {
            break;
        }
    }

    // Verify size if known
    if let Some(expected) = header.frame_content_size {
        if out_pos as u64 != expected {
            return Err(Error::CorruptedData(format!(
                "size mismatch: expected {expected}, got {out_pos}"
            )));
        }
    }

    Ok(out_pos)
}

fn read_frame_header(input: &[u8], pos: &mut usize) -> Result<FrameHeader> {
    if input.len() < 5 {
        return Err(Error::CorruptedData(
            "input too short for frame header".to_string(),
        ));
    }

    // Check magic
    let magic = u32::from_le_bytes([input[0], input[1], input[2], input[3]]);
    if magic != ZSTD_MAGIC {
        return Err(Error::CorruptedData(format!(
            "invalid magic: expected {ZSTD_MAGIC:#X}, got {magic:#X}"
        )));
    }
    *pos = 4;

    // Read descriptor
    let descriptor = input[*pos];
    *pos += 1;

    let fcs_flag = (descriptor >> 6) & 0x03;
    let single_segment = (descriptor >> 5) & 0x01 != 0;
    let checksum = (descriptor >> 2) & 0x01 != 0;
    let dict_id_flag = descriptor & 0x03;

    // Window descriptor (if not single segment)
    let window_size = if single_segment {
        0 // Will be determined by frame content size
    } else {
        if *pos >= input.len() {
            return Err(Error::CorruptedData(
                "missing window descriptor".to_string(),
            ));
        }
        let wd = input[*pos];
        *pos += 1;
        let exponent = u64::from(wd >> 3);
        let mantissa = u64::from(wd & 0x07);
        (1u64 << (10 + exponent)) + (mantissa << (7 + exponent))
    };

    // Dictionary ID
    let dictionary_id = match dict_id_flag {
        0 => None,
        1 => {
            if *pos >= input.len() {
                return Err(Error::CorruptedData("missing dict id".to_string()));
            }
            let id = u32::from(input[*pos]);
            *pos += 1;
            Some(id)
        }
        2 => {
            if *pos + 2 > input.len() {
                return Err(Error::CorruptedData("missing dict id".to_string()));
            }
            let id = u32::from(u16::from_le_bytes([input[*pos], input[*pos + 1]]));
            *pos += 2;
            Some(id)
        }
        3 => {
            if *pos + 4 > input.len() {
                return Err(Error::CorruptedData("missing dict id".to_string()));
            }
            let id = u32::from_le_bytes([
                input[*pos],
                input[*pos + 1],
                input[*pos + 2],
                input[*pos + 3],
            ]);
            *pos += 4;
            Some(id)
        }
        _ => unreachable!(),
    };

    // Frame content size
    let frame_content_size = match fcs_flag {
        0 if single_segment => {
            if *pos >= input.len() {
                return Err(Error::CorruptedData(
                    "missing frame content size".to_string(),
                ));
            }
            let size = u64::from(input[*pos]);
            *pos += 1;
            Some(size)
        }
        0 => None,
        1 => {
            if *pos + 2 > input.len() {
                return Err(Error::CorruptedData(
                    "missing frame content size".to_string(),
                ));
            }
            let size = u64::from(u16::from_le_bytes([input[*pos], input[*pos + 1]])) + 256;
            *pos += 2;
            Some(size)
        }
        2 => {
            if *pos + 4 > input.len() {
                return Err(Error::CorruptedData(
                    "missing frame content size".to_string(),
                ));
            }
            let size = u64::from(u32::from_le_bytes([
                input[*pos],
                input[*pos + 1],
                input[*pos + 2],
                input[*pos + 3],
            ]));
            *pos += 4;
            Some(size)
        }
        3 => {
            if *pos + 8 > input.len() {
                return Err(Error::CorruptedData(
                    "missing frame content size".to_string(),
                ));
            }
            let size = u64::from_le_bytes([
                input[*pos],
                input[*pos + 1],
                input[*pos + 2],
                input[*pos + 3],
                input[*pos + 4],
                input[*pos + 5],
                input[*pos + 6],
                input[*pos + 7],
            ]);
            *pos += 8;
            Some(size)
        }
        _ => unreachable!(),
    };

    Ok(FrameHeader {
        window_size,
        frame_content_size,
        dictionary_id,
        checksum,
        single_segment,
    })
}

fn read_block_header(input: &[u8], pos: &mut usize) -> Result<BlockHeader> {
    if *pos + 3 > input.len() {
        return Err(Error::CorruptedData("missing block header".to_string()));
    }

    let header =
        u32::from(input[*pos]) | (u32::from(input[*pos + 1]) << 8) | (u32::from(input[*pos + 2]) << 16);
    *pos += 3;

    let last_block = (header & 0x01) != 0;
    let block_type = BlockType::from(((header >> 1) & 0x03) as u8);
    let block_size = header >> 3;

    Ok(BlockHeader {
        block_type,
        last_block,
        block_size,
    })
}

fn decompress_block(input: &[u8], output: &mut [u8]) -> Result<usize> {
    if input.is_empty() {
        return Ok(0);
    }

    // Read literals section header
    let (lit_type, lit_size, lit_header_size) = read_literals_header(input)?;

    if lit_header_size >= input.len() {
        return Err(Error::CorruptedData(
            "literals extend past block".to_string(),
        ));
    }

    let lit_start = lit_header_size;
    let lit_end = lit_start
        + match lit_type {
            0 => lit_size, // Raw literals
            1 => 1,        // RLE
            _ => lit_size, // Compressed (simplified)
        };

    if lit_end > input.len() {
        return Err(Error::CorruptedData(
            "literals extend past input".to_string(),
        ));
    }

    // Handle literals
    match lit_type {
        0 => {
            // Raw literals
            if lit_size > output.len() {
                return Err(Error::BufferTooSmall {
                    needed: lit_size,
                    available: output.len(),
                });
            }
            output[..lit_size].copy_from_slice(&input[lit_start..lit_end]);
        }
        1 => {
            // RLE literals
            let byte = input[lit_start];
            if lit_size > output.len() {
                return Err(Error::BufferTooSmall {
                    needed: lit_size,
                    available: output.len(),
                });
            }
            for i in 0..lit_size {
                output[i] = byte;
            }
        }
        _ => {
            // Compressed literals - simplified handling
            return Err(Error::Unsupported("compressed literals".to_string()));
        }
    }

    // Read sequences section
    let seq_start = lit_end;
    if seq_start >= input.len() {
        // No sequences, just literals
        return Ok(lit_size);
    }

    let num_sequences = input[seq_start] as usize;
    if num_sequences == 0 {
        return Ok(lit_size);
    }

    // Full sequence decoding would go here
    // For now, return error for non-zero sequences
    Err(Error::Unsupported("sequence decoding".to_string()))
}

fn read_literals_header(input: &[u8]) -> Result<(u8, usize, usize)> {
    if input.is_empty() {
        return Err(Error::CorruptedData("missing literals header".to_string()));
    }

    let header = input[0];
    let lit_type = header & 0x03;
    let size_format = (header >> 2) & 0x03;

    match size_format {
        0 | 1 => {
            // 1 byte header
            let size = (header >> 3) as usize;
            Ok((lit_type, size, 1))
        }
        2 => {
            // 2 byte header
            if input.len() < 2 {
                return Err(Error::CorruptedData(
                    "truncated literals header".to_string(),
                ));
            }
            let size = ((header >> 4) as usize) | ((input[1] as usize) << 4);
            Ok((lit_type, size, 2))
        }
        3 => {
            // 3 byte header
            if input.len() < 3 {
                return Err(Error::CorruptedData(
                    "truncated literals header".to_string(),
                ));
            }
            let size =
                ((header >> 4) as usize) | ((input[1] as usize) << 4) | ((input[2] as usize) << 12);
            Ok((lit_type, size, 3))
        }
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_magic() {
        let input = [0x00, 0x00, 0x00, 0x00, 0x00];
        let mut output = [0u8; 100];
        let result = decompress(&input, &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_block_header() {
        // Last block, raw type, size 100
        let input = [0x21, 0x03, 0x00]; // 1 | (0 << 1) | (100 << 3)
        let mut pos = 0;
        let header = read_block_header(&input, &mut pos).unwrap();
        assert!(header.last_block);
        assert_eq!(header.block_type, BlockType::Raw);
        assert_eq!(header.block_size, 100);
    }
}
