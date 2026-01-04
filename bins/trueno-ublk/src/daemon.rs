//! Daemon module - ublk I/O processing
//!
//! Handles block device I/O using libublk and io_uring.

use anyhow::Result;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use trueno_zram_core::{CompressedPage as CoreCompressedPage, PageCompressor, PAGE_SIZE};

/// Compressed page storage
pub struct PageStore {
    pages: HashMap<u64, StoredPage>,
    compressor: Arc<dyn PageCompressor>,
    entropy_threshold: f64,

    // Statistics
    bytes_stored: AtomicU64,
    bytes_compressed: AtomicU64,
    zero_pages: AtomicU64,
    gpu_pages: AtomicU64,
    simd_pages: AtomicU64,
    scalar_pages: AtomicU64,
}

struct StoredPage {
    compressed: CoreCompressedPage,
    is_zero: bool,
}

impl PageStore {
    pub fn new(compressor: Arc<dyn PageCompressor>, entropy_threshold: f64) -> Self {
        Self {
            pages: HashMap::new(),
            compressor,
            entropy_threshold,
            bytes_stored: AtomicU64::new(0),
            bytes_compressed: AtomicU64::new(0),
            zero_pages: AtomicU64::new(0),
            gpu_pages: AtomicU64::new(0),
            simd_pages: AtomicU64::new(0),
            scalar_pages: AtomicU64::new(0),
        }
    }

    /// Store a page at the given sector offset
    pub fn store(&mut self, sector: u64, data: &[u8]) -> Result<()> {
        debug_assert_eq!(data.len(), PAGE_SIZE);

        // Convert slice to fixed-size array
        let page: &[u8; PAGE_SIZE] = data.try_into().map_err(|_| {
            anyhow::anyhow!("Invalid page size: expected {}, got {}", PAGE_SIZE, data.len())
        })?;

        // Check for zero page
        if is_zero_page(data) {
            // Create a zero-page compressed representation
            let compressed = self.compressor.compress(page)?;
            self.pages.insert(
                sector,
                StoredPage {
                    compressed,
                    is_zero: true,
                },
            );
            self.zero_pages.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }

        // Calculate entropy
        let entropy = calculate_entropy(data);

        // Compress using trueno-zram-core
        let compressed = self.compressor.compress(page)?;

        // Track which backend was used based on entropy
        if entropy > self.entropy_threshold {
            self.scalar_pages.fetch_add(1, Ordering::Relaxed);
        } else if entropy < 4.0 {
            self.gpu_pages.fetch_add(1, Ordering::Relaxed);
        } else {
            self.simd_pages.fetch_add(1, Ordering::Relaxed);
        }

        self.bytes_stored
            .fetch_add(data.len() as u64, Ordering::Relaxed);
        self.bytes_compressed
            .fetch_add(compressed.data.len() as u64, Ordering::Relaxed);

        self.pages.insert(
            sector,
            StoredPage {
                compressed,
                is_zero: false,
            },
        );

        Ok(())
    }

    /// Load a page from the given sector offset
    pub fn load(&self, sector: u64, buffer: &mut [u8]) -> Result<bool> {
        debug_assert_eq!(buffer.len(), PAGE_SIZE);

        match self.pages.get(&sector) {
            Some(page) if page.is_zero => {
                buffer.fill(0);
                Ok(true)
            }
            Some(page) => {
                // Decompress using trueno-zram-core
                let decompressed = self.compressor.decompress(&page.compressed)?;
                buffer.copy_from_slice(&decompressed);
                Ok(true)
            }
            None => {
                // Page not found - return zeros
                buffer.fill(0);
                Ok(false)
            }
        }
    }

    /// Remove a page
    pub fn remove(&mut self, sector: u64) -> bool {
        self.pages.remove(&sector).is_some()
    }

    /// Get statistics
    pub fn stats(&self) -> PageStoreStats {
        PageStoreStats {
            pages_stored: self.pages.len() as u64,
            bytes_stored: self.bytes_stored.load(Ordering::Relaxed),
            bytes_compressed: self.bytes_compressed.load(Ordering::Relaxed),
            zero_pages: self.zero_pages.load(Ordering::Relaxed),
            gpu_pages: self.gpu_pages.load(Ordering::Relaxed),
            simd_pages: self.simd_pages.load(Ordering::Relaxed),
            scalar_pages: self.scalar_pages.load(Ordering::Relaxed),
        }
    }
}

pub struct PageStoreStats {
    pub pages_stored: u64,
    pub bytes_stored: u64,
    pub bytes_compressed: u64,
    pub zero_pages: u64,
    pub gpu_pages: u64,
    pub simd_pages: u64,
    pub scalar_pages: u64,
}

/// Check if a page is all zeros
fn is_zero_page(data: &[u8]) -> bool {
    // Use SIMD-friendly comparison
    data.iter().all(|&b| b == 0)
}

/// Calculate Shannon entropy of data
fn calculate_entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut counts = [0u32; 256];
    for &byte in data {
        counts[byte as usize] += 1;
    }

    let len = data.len() as f64;
    let mut entropy = 0.0;

    for &count in &counts {
        if count > 0 {
            let p = count as f64 / len;
            entropy -= p * p.log2();
        }
    }

    entropy
}

/// I/O request type
#[derive(Debug, Clone, Copy)]
pub enum IoType {
    Read,
    Write,
    Discard,
}

/// Block I/O request
pub struct IoRequest {
    pub io_type: IoType,
    pub sector: u64,
    pub len: u32,
    pub buffer: *mut u8,
}

unsafe impl Send for IoRequest {}
unsafe impl Sync for IoRequest {}

/// Process a block I/O request
pub fn process_io(store: &mut PageStore, req: &IoRequest) -> Result<()> {
    let pages = (req.len as usize).div_ceil(PAGE_SIZE);

    match req.io_type {
        IoType::Read => {
            for i in 0..pages {
                let sector = req.sector + (i * (PAGE_SIZE / 512)) as u64;
                let offset = i * PAGE_SIZE;
                let buffer =
                    unsafe { std::slice::from_raw_parts_mut(req.buffer.add(offset), PAGE_SIZE) };
                store.load(sector, buffer)?;
            }
        }
        IoType::Write => {
            for i in 0..pages {
                let sector = req.sector + (i * (PAGE_SIZE / 512)) as u64;
                let offset = i * PAGE_SIZE;
                let buffer =
                    unsafe { std::slice::from_raw_parts(req.buffer.add(offset), PAGE_SIZE) };
                store.store(sector, buffer)?;
            }
        }
        IoType::Discard => {
            for i in 0..pages {
                let sector = req.sector + (i * (PAGE_SIZE / 512)) as u64;
                store.remove(sector);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_zero_page() {
        let zeros = vec![0u8; PAGE_SIZE];
        assert!(is_zero_page(&zeros));

        let mut data = vec![0u8; PAGE_SIZE];
        data[100] = 1;
        assert!(!is_zero_page(&data));
    }

    #[test]
    fn test_entropy_calculation() {
        // All zeros - zero entropy
        let zeros = vec![0u8; PAGE_SIZE];
        assert_eq!(calculate_entropy(&zeros), 0.0);

        // Random-like data - high entropy
        let random: Vec<u8> = (0..PAGE_SIZE).map(|i| (i * 17 + 31) as u8).collect();
        let entropy = calculate_entropy(&random);
        assert!(entropy > 7.0);

        // Repetitive data - low entropy
        let repetitive: Vec<u8> = (0..PAGE_SIZE).map(|i| (i % 4) as u8).collect();
        let entropy = calculate_entropy(&repetitive);
        assert!(entropy < 3.0);
    }
}
