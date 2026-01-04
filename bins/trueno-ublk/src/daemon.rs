//! Daemon module - ublk I/O processing
//!
//! Handles block device I/O using direct ublk kernel interface.

use anyhow::Result;
use std::collections::HashMap;
use std::io::{Error as IoError, ErrorKind, Result as IoResult};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use trueno_zram_core::{Algorithm, CompressedPage as CoreCompressedPage, CompressorBuilder, PageCompressor, PAGE_SIZE};

/// Sector size in bytes
const SECTOR_SIZE: u64 = 512;

/// Pages per sector (4096 / 512 = 8)
const SECTORS_PER_PAGE: u64 = PAGE_SIZE as u64 / SECTOR_SIZE;

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
    /// Create a new PageStore with the given device size and algorithm
    pub fn new(dev_size: u64, algorithm: Algorithm) -> Self {
        let compressor: Arc<dyn PageCompressor> = Arc::from(
            CompressorBuilder::new()
                .algorithm(algorithm)
                .build()
                .expect("Failed to create compressor")
        );
        Self {
            pages: HashMap::with_capacity((dev_size / PAGE_SIZE as u64) as usize),
            compressor,
            entropy_threshold: 6.0,
            bytes_stored: AtomicU64::new(0),
            bytes_compressed: AtomicU64::new(0),
            zero_pages: AtomicU64::new(0),
            gpu_pages: AtomicU64::new(0),
            simd_pages: AtomicU64::new(0),
            scalar_pages: AtomicU64::new(0),
        }
    }

    /// Create with custom compressor (for testing)
    pub fn with_compressor(compressor: Arc<dyn PageCompressor>, entropy_threshold: f64) -> Self {
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

    // =========================================================================
    // ublk daemon interface methods
    // =========================================================================

    /// Read data from store at sector offset (ublk daemon interface)
    pub fn read(&self, start_sector: u64, buffer: &mut [u8]) -> IoResult<usize> {
        let mut offset = 0;
        let mut sector = start_sector;

        while offset < buffer.len() {
            let page_sector = (sector / SECTORS_PER_PAGE) * SECTORS_PER_PAGE;
            let sector_offset_in_page = (sector % SECTORS_PER_PAGE) as usize * SECTOR_SIZE as usize;
            let remaining_in_page = PAGE_SIZE - sector_offset_in_page;
            let to_read = (buffer.len() - offset).min(remaining_in_page);

            match self.pages.get(&page_sector) {
                Some(page) if page.is_zero => {
                    buffer[offset..offset + to_read].fill(0);
                }
                Some(page) => {
                    let decompressed = self.compressor.decompress(&page.compressed)
                        .map_err(|e| IoError::new(ErrorKind::InvalidData, e.to_string()))?;
                    buffer[offset..offset + to_read]
                        .copy_from_slice(&decompressed[sector_offset_in_page..sector_offset_in_page + to_read]);
                }
                None => {
                    buffer[offset..offset + to_read].fill(0);
                }
            }

            offset += to_read;
            sector += (to_read / SECTOR_SIZE as usize) as u64;
        }
        Ok(buffer.len())
    }

    /// Write data to store at sector offset (ublk daemon interface)
    pub fn write(&mut self, start_sector: u64, data: &[u8]) -> IoResult<usize> {
        let mut offset = 0;
        let mut sector = start_sector;

        while offset < data.len() {
            let page_sector = (sector / SECTORS_PER_PAGE) * SECTORS_PER_PAGE;
            let sector_offset_in_page = (sector % SECTORS_PER_PAGE) as usize * SECTOR_SIZE as usize;
            let remaining_in_page = PAGE_SIZE - sector_offset_in_page;
            let to_write = (data.len() - offset).min(remaining_in_page);

            if to_write < PAGE_SIZE {
                let mut page_buf = [0u8; PAGE_SIZE];
                if let Some(page) = self.pages.get(&page_sector) {
                    if !page.is_zero {
                        let decompressed = self.compressor.decompress(&page.compressed)
                            .map_err(|e| IoError::new(ErrorKind::InvalidData, e.to_string()))?;
                        page_buf.copy_from_slice(&decompressed);
                    }
                }
                page_buf[sector_offset_in_page..sector_offset_in_page + to_write]
                    .copy_from_slice(&data[offset..offset + to_write]);
                self.store_page(page_sector, &page_buf)?;
            } else {
                let page_data: &[u8; PAGE_SIZE] = (&data[offset..offset + PAGE_SIZE]).try_into().unwrap();
                self.store_page(page_sector, page_data)?;
            }

            offset += to_write;
            sector += (to_write / SECTOR_SIZE as usize) as u64;
        }
        Ok(data.len())
    }

    /// Discard sectors (ublk daemon interface)
    pub fn discard(&mut self, start_sector: u64, nr_sectors: u32) -> IoResult<usize> {
        let end_sector = start_sector + nr_sectors as u64;
        let mut sector = start_sector;
        while sector < end_sector {
            let page_sector = (sector / SECTORS_PER_PAGE) * SECTORS_PER_PAGE;
            self.pages.remove(&page_sector);
            sector = page_sector + SECTORS_PER_PAGE;
        }
        Ok(0)
    }

    /// Write zeros to sectors (ublk daemon interface)
    pub fn write_zeroes(&mut self, start_sector: u64, nr_sectors: u32) -> IoResult<usize> {
        let end_sector = start_sector + nr_sectors as u64;
        let mut sector = start_sector;
        while sector < end_sector {
            let page_sector = (sector / SECTORS_PER_PAGE) * SECTORS_PER_PAGE;
            let compressed = self.compressor.compress(&[0u8; PAGE_SIZE])
                .map_err(|e| IoError::new(ErrorKind::InvalidData, e.to_string()))?;
            self.pages.insert(page_sector, StoredPage { compressed, is_zero: true });
            self.zero_pages.fetch_add(1, Ordering::Relaxed);
            sector = page_sector + SECTORS_PER_PAGE;
        }
        Ok(0)
    }

    fn store_page(&mut self, sector: u64, data: &[u8; PAGE_SIZE]) -> IoResult<()> {
        if is_zero_page(data) {
            let compressed = self.compressor.compress(data)
                .map_err(|e| IoError::new(ErrorKind::InvalidData, e.to_string()))?;
            self.pages.insert(sector, StoredPage { compressed, is_zero: true });
            self.zero_pages.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }

        let entropy = calculate_entropy(data);
        let compressed = self.compressor.compress(data)
            .map_err(|e| IoError::new(ErrorKind::InvalidData, e.to_string()))?;

        if entropy > self.entropy_threshold {
            self.scalar_pages.fetch_add(1, Ordering::Relaxed);
        } else if entropy < 4.0 {
            self.gpu_pages.fetch_add(1, Ordering::Relaxed);
        } else {
            self.simd_pages.fetch_add(1, Ordering::Relaxed);
        }

        self.bytes_stored.fetch_add(PAGE_SIZE as u64, Ordering::Relaxed);
        self.bytes_compressed.fetch_add(compressed.data.len() as u64, Ordering::Relaxed);
        self.pages.insert(sector, StoredPage { compressed, is_zero: false });
        Ok(())
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

    // ========================================================================
    // Helper functions for tests
    // ========================================================================

    fn create_test_store() -> PageStore {
        PageStore::new(1024 * 1024 * 1024, Algorithm::Lz4) // 1GB device
    }

    // ========================================================================
    // Basic functionality tests
    // ========================================================================

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

    // ========================================================================
    // Section B: Data Integrity Tests (from Renacer Verification Matrix)
    // ========================================================================

    /// B.11: Roundtrip: Write 0xDEADBEEF pattern, Read back and verify
    #[test]
    fn test_roundtrip_deadbeef() {
        let mut store = create_test_store();

        // Create page filled with 0xDEADBEEF pattern
        let mut data = [0u8; PAGE_SIZE];
        for i in (0..PAGE_SIZE).step_by(4) {
            data[i] = 0xDE;
            data[i + 1] = 0xAD;
            data[i + 2] = 0xBE;
            data[i + 3] = 0xEF;
        }

        // Write to sector 0
        store.write(0, &data).expect("write failed");

        // Read back
        let mut buffer = [0u8; PAGE_SIZE];
        store.read(0, &mut buffer).expect("read failed");

        // Verify exact match
        assert_eq!(data, buffer, "Roundtrip data mismatch");
    }

    /// B.12: Zero-page logic: Write zeros, verify efficient storage
    #[test]
    fn test_zero_page_optimization() {
        let mut store = create_test_store();

        // Write zero page
        let zeros = [0u8; PAGE_SIZE];
        store.write(0, &zeros).expect("write failed");

        // Verify stats show zero page was detected
        let stats = store.stats();
        assert!(stats.zero_pages > 0, "Zero page not detected");

        // Read back should return zeros
        let mut buffer = [0xFFu8; PAGE_SIZE];
        store.read(0, &mut buffer).expect("read failed");
        assert!(is_zero_page(&buffer), "Zero page not returned correctly");
    }

    /// B.13: Partial pages: Write 512 bytes (1 sector) -> Read 4KB (1 page)
    #[test]
    fn test_partial_page_write() {
        let mut store = create_test_store();

        // Write only 512 bytes (1 sector)
        let sector_data = [0xABu8; 512];
        store.write(0, &sector_data).expect("write failed");

        // Read full page (4KB)
        let mut buffer = [0u8; PAGE_SIZE];
        store.read(0, &mut buffer).expect("read failed");

        // First 512 bytes should match, rest should be zeros
        assert_eq!(&buffer[..512], &sector_data[..], "First sector mismatch");
        assert!(buffer[512..].iter().all(|&b| b == 0), "Rest should be zeros");
    }

    /// B.14: Boundary check: Write to last sector
    #[test]
    fn test_boundary_last_sector() {
        let mut store = create_test_store();

        // Calculate last page sector
        let dev_size = 1024 * 1024 * 1024u64; // 1GB
        let last_page_sector = (dev_size / PAGE_SIZE as u64 - 1) * SECTORS_PER_PAGE;

        // Write to last page
        let data = [0xEFu8; PAGE_SIZE];
        store.write(last_page_sector, &data).expect("write to last sector failed");

        // Read back
        let mut buffer = [0u8; PAGE_SIZE];
        store.read(last_page_sector, &mut buffer).expect("read from last sector failed");
        assert_eq!(data, buffer, "Last sector data mismatch");
    }

    /// B.15: Offset check: Write to sector 8, verify sector 0 is unchanged
    #[test]
    fn test_offset_independence() {
        let mut store = create_test_store();

        // Write to sector 0 (first page)
        let data0 = [0x11u8; PAGE_SIZE];
        store.write(0, &data0).expect("write to sector 0 failed");

        // Write to sector 8 (second page)
        let data8 = [0x22u8; PAGE_SIZE];
        store.write(8, &data8).expect("write to sector 8 failed");

        // Verify sector 0 is unchanged
        let mut buffer = [0u8; PAGE_SIZE];
        store.read(0, &mut buffer).expect("read sector 0 failed");
        assert_eq!(data0, buffer, "Sector 0 was corrupted");

        // Verify sector 8 has correct data
        store.read(8, &mut buffer).expect("read sector 8 failed");
        assert_eq!(data8, buffer, "Sector 8 data mismatch");
    }

    /// B.16: Data pollution: Fill device with random data, verify checksums
    #[test]
    fn test_data_pollution_checksums() {
        let mut store = create_test_store();

        // Write 100 pages with pseudo-random data
        let num_pages = 100;
        let mut expected_checksums = Vec::new();

        for i in 0..num_pages {
            let mut data = [0u8; PAGE_SIZE];
            // Fill with pseudo-random pattern based on page index
            for j in 0..PAGE_SIZE {
                data[j] = ((i * 17 + j * 31 + 7) % 256) as u8;
            }

            let checksum: u64 = data.iter().map(|&b| b as u64).sum();
            expected_checksums.push(checksum);

            let sector = (i as u64) * SECTORS_PER_PAGE;
            store.write(sector, &data).expect("write failed");
        }

        // Verify all pages
        for i in 0..num_pages {
            let mut buffer = [0u8; PAGE_SIZE];
            let sector = (i as u64) * SECTORS_PER_PAGE;
            store.read(sector, &mut buffer).expect("read failed");

            let checksum: u64 = buffer.iter().map(|&b| b as u64).sum();
            assert_eq!(
                checksum, expected_checksums[i],
                "Checksum mismatch at page {}",
                i
            );
        }
    }

    /// B.19: Discard: Verify discard clears data in PageStore
    #[test]
    fn test_discard_clears_data() {
        let mut store = create_test_store();

        // Write data
        let data = [0xCDu8; PAGE_SIZE];
        store.write(0, &data).expect("write failed");

        // Verify data exists
        let mut buffer = [0u8; PAGE_SIZE];
        store.read(0, &mut buffer).expect("read failed");
        assert_eq!(data, buffer, "Data not written");

        // Discard the sector
        store.discard(0, SECTORS_PER_PAGE as u32).expect("discard failed");

        // Read should return zeros (unallocated)
        store.read(0, &mut buffer).expect("read after discard failed");
        assert!(is_zero_page(&buffer), "Data not cleared after discard");
    }

    /// B.20: Write zeros operation
    #[test]
    fn test_write_zeroes_operation() {
        let mut store = create_test_store();

        // Write non-zero data first
        let data = [0xFFu8; PAGE_SIZE];
        store.write(0, &data).expect("write failed");

        // Issue write_zeroes
        store.write_zeroes(0, SECTORS_PER_PAGE as u32).expect("write_zeroes failed");

        // Read should return zeros
        let mut buffer = [0xFFu8; PAGE_SIZE];
        store.read(0, &mut buffer).expect("read failed");
        assert!(is_zero_page(&buffer), "write_zeroes did not zero the data");
    }

    /// Test multiple sector write/read spanning pages
    #[test]
    fn test_multi_sector_io() {
        let mut store = create_test_store();

        // Write 2 pages worth of data (16 sectors)
        let data = vec![0xABu8; PAGE_SIZE * 2];
        store.write(0, &data).expect("write failed");

        // Read back
        let mut buffer = vec![0u8; PAGE_SIZE * 2];
        store.read(0, &mut buffer).expect("read failed");
        assert_eq!(data, buffer, "Multi-sector data mismatch");
    }

    /// Test read from unallocated sector returns zeros
    #[test]
    fn test_read_unallocated_returns_zeros() {
        let store = create_test_store();

        let mut buffer = [0xFFu8; PAGE_SIZE];
        store.read(0, &mut buffer).expect("read failed");
        assert!(is_zero_page(&buffer), "Unallocated sector should return zeros");
    }

    /// Test overwrite existing data
    #[test]
    fn test_overwrite_data() {
        let mut store = create_test_store();

        // Write initial data
        let data1 = [0x11u8; PAGE_SIZE];
        store.write(0, &data1).expect("first write failed");

        // Overwrite with different data
        let data2 = [0x22u8; PAGE_SIZE];
        store.write(0, &data2).expect("second write failed");

        // Read back should return second data
        let mut buffer = [0u8; PAGE_SIZE];
        store.read(0, &mut buffer).expect("read failed");
        assert_eq!(data2, buffer, "Overwrite failed");
    }

    /// Test compression ratio (data should be smaller when compressed)
    #[test]
    fn test_compression_ratio() {
        let mut store = create_test_store();

        // Write highly compressible data (repeated pattern)
        let mut data = [0u8; PAGE_SIZE];
        for i in 0..PAGE_SIZE {
            data[i] = (i % 16) as u8;
        }
        store.write(0, &data).expect("write failed");

        let stats = store.stats();
        // Compressed size should be less than original
        if stats.bytes_stored > 0 {
            assert!(
                stats.bytes_compressed < stats.bytes_stored,
                "Compressible data should compress: stored={} compressed={}",
                stats.bytes_stored,
                stats.bytes_compressed
            );
        }
    }

    /// Test stats tracking
    #[test]
    fn test_stats_tracking() {
        let mut store = create_test_store();

        // Write zero page
        let zeros = [0u8; PAGE_SIZE];
        store.write(0, &zeros).expect("write failed");

        let stats = store.stats();
        assert_eq!(stats.pages_stored, 1);
        assert!(stats.zero_pages >= 1);

        // Write non-zero page
        let data = [0x42u8; PAGE_SIZE];
        store.write(8, &data).expect("write failed");

        let stats = store.stats();
        assert_eq!(stats.pages_stored, 2);
    }
}
