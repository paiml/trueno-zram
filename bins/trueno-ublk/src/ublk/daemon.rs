//! ublk daemon I/O loop
//!
//! Uses io_uring to handle block device I/O.
//!
//! ## Batched Mode
//!
//! For high-throughput scenarios, use `run_daemon_batched` which buffers pages
//! and compresses in batches using GPU when available.

#[cfg(not(test))]
use crate::daemon::PageStore;
#[cfg(not(test))]
use crate::daemon::{BatchConfig, BatchedPageStore, spawn_flush_thread};
use crate::ublk::ctrl::{CtrlError, DeviceConfig};
#[cfg(not(test))]
use crate::ublk::UblkCtrl;
use crate::ublk::sys::*;
#[cfg(not(test))]
use io_uring::{opcode, squeue, types, IoUring};
use nix::libc;
#[cfg(not(test))]
use nix::libc::{mmap, munmap, MAP_ANONYMOUS, MAP_PRIVATE, PROT_READ, PROT_WRITE};
#[cfg(not(test))]
use nix::sys::uio::{pread, pwrite};
#[cfg(not(test))]
use std::os::fd::{AsRawFd, BorrowedFd};
#[cfg(not(test))]
use std::ptr::null_mut;
#[cfg(not(test))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(test))]
use std::sync::Arc;
#[cfg(not(test))]
use std::time::Duration;
use thiserror::Error;
#[cfg(not(test))]
use tracing::{debug, info, warn};

#[derive(Error, Debug)]
pub enum DaemonError {
    #[error("Control error: {0}")]
    Ctrl(#[from] CtrlError),

    #[error("io_uring creation failed: {0}")]
    IoUringCreate(std::io::Error),

    #[error("Failed to mmap shared buffer: {0}")]
    Mmap(std::io::Error),

    #[error("io_uring submission failed: {0}")]
    Submit(std::io::Error),

    #[error("Queue full, cannot submit")]
    QueueFull,

    #[error("Would block")]
    WouldBlock,

    #[error("Device stopped")]
    Stopped,
}

impl DaemonError {
    /// Returns true if this error is transient and the operation should be retried.
    /// Pattern from pepita error handling.
    #[inline]
    pub const fn is_retriable(&self) -> bool {
        matches!(self, Self::QueueFull | Self::WouldBlock)
    }

    /// Returns true if this is a resource exhaustion error.
    #[inline]
    pub const fn is_resource_error(&self) -> bool {
        matches!(self, Self::QueueFull | Self::Mmap(_) | Self::IoUringCreate(_))
    }

    /// Convert to negative errno for POSIX compatibility.
    /// Pattern from pepita error handling.
    pub fn to_errno(&self) -> i32 {
        match self {
            Self::Ctrl(_) => -libc::ENODEV,
            Self::IoUringCreate(e) => -e.raw_os_error().unwrap_or(libc::ENOMEM),
            Self::Mmap(e) => -e.raw_os_error().unwrap_or(libc::ENOMEM),
            Self::Submit(e) => -e.raw_os_error().unwrap_or(libc::EIO),
            Self::QueueFull => -libc::ENOSPC,
            Self::WouldBlock => -libc::EAGAIN,
            Self::Stopped => -libc::ENODEV,
        }
    }
}

/// Running daemon handle
#[cfg(not(test))]
pub struct UblkDaemon {
    ctrl: UblkCtrl,
    char_fd: i32,
    ring: IoUring,
    iod_buf: *mut u8,       // IOD buffer (mmap'd from char device)
    iod_buf_size: usize,
    data_buf: *mut u8,      // Data buffer (anonymous mmap for our use)
    data_buf_size: usize,
    queue_depth: u16,
    max_io_size: u32,
    stop: Arc<AtomicBool>,
    use_ioctl_encode: bool, // Whether kernel uses ioctl-encoded cmd_op
    ready_fd: Option<i32>,
}

// IOD buffer size must be page-aligned
pub(crate) fn iod_buf_size(queue_depth: u16) -> usize {
    let size = (queue_depth as usize) * std::mem::size_of::<UblkIoDesc>();
    (size + 4095) & !4095 // Round up to page size
}

#[cfg(not(test))]
impl UblkDaemon {
    pub fn new(config: DeviceConfig, stop: Arc<AtomicBool>, ready_fd: Option<i32>) -> Result<Self, DaemonError> {
        info!(dev_size = config.dev_size, "Creating ublk daemon");

        let ctrl = UblkCtrl::new(config)?;
        let dev_id = ctrl.dev_id();
        let queue_depth = ctrl.queue_depth();
        let max_io_size = ctrl.max_io_buf_bytes();

        info!(dev_id, queue_depth, max_io_size, "Device created");

        let char_fd_owned = ctrl.open_char_dev()?;
        let char_fd = char_fd_owned.as_raw_fd();
        std::mem::forget(char_fd_owned);

        // FIX B: Disable SQPOLL mode - use standard io_uring submission
        // SQPOLL creates a kernel polling thread but has race conditions with URING_CMD
        // operations used by ublk. Standard mode with explicit submit_and_wait is more reliable.
        //
        // The Five-Whys analysis (Section 8 of ublk-batched-gpu-compression.md) identified
        // that SQPOLL may not process FETCH commands before START_DEV is called, causing
        // the daemon to hang indefinitely.
        let ring: IoUring = IoUring::builder()
            .build(queue_depth as u32 * 2)
            .map_err(DaemonError::IoUringCreate)?;

        // mmap IOD buffer from char device (read-only, kernel writes I/O descriptors here)
        let iod_buf_size = iod_buf_size(queue_depth);
        let iod_buf = unsafe {
            mmap(
                null_mut(),
                iod_buf_size,
                PROT_READ, // Read-only per kernel requirement
                libc::MAP_SHARED,
                char_fd,
                0, // UBLKSRV_CMD_BUF_OFFSET = 0
            )
        };
        if iod_buf == libc::MAP_FAILED {
            return Err(DaemonError::Mmap(std::io::Error::last_os_error()));
        }
        let iod_buf = iod_buf as *mut u8;
        debug!(iod_buf_size, "IOD buffer mmap'd from char device");

        // Allocate anonymous data buffer for our use (compress/decompress workspace)
        let data_buf_size = (queue_depth as usize) * (max_io_size as usize);
        let data_buf = unsafe {
            mmap(null_mut(), data_buf_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
        };
        if data_buf == libc::MAP_FAILED {
            unsafe { munmap(iod_buf as *mut _, iod_buf_size) };
            return Err(DaemonError::Mmap(std::io::Error::last_os_error()));
        }
        let data_buf = data_buf as *mut u8;
        debug!(data_buf_size, "Data buffer allocated");

        // Check if kernel supports ioctl-encoded command opcodes
        let use_ioctl_encode = (ctrl.dev_info().flags & UBLK_F_CMD_IOCTL_ENCODE) != 0;
        eprintln!("[TRACE] Device flags=0x{:X}, use_ioctl_encode={}", ctrl.dev_info().flags, use_ioctl_encode);

        Ok(Self {
            ctrl,
            char_fd,
            ring,
            iod_buf,
            iod_buf_size,
            data_buf,
            data_buf_size,
            queue_depth,
            max_io_size,
            stop,
            use_ioctl_encode,
            ready_fd,
        })
    }

    pub fn dev_id(&self) -> i32 { self.ctrl.dev_id() }

    pub fn block_dev_path(&self) -> String { self.ctrl.block_dev_path() }

    /// FIX F: Restructured run() to ensure io_uring is blocked in submit_and_wait
    /// BEFORE START_DEV is called.
    ///
    /// The ublk kernel driver requires FETCH commands to be "in flight" in io_uring
    /// before START_DEV can complete. The previous approach spawned START_DEV in a
    /// background thread but didn't guarantee io_uring was actively waiting.
    ///
    /// New approach:
    /// 1. Submit all FETCH commands
    /// 2. Spawn a thread that calls START_DEV after a delay
    /// 3. Main thread immediately enters submit_and_wait() loop (blocks in kernel)
    /// 4. START_DEV thread waits to ensure main thread is blocked, then calls START_DEV
    /// 5. Kernel sees io_uring waiting → START_DEV completes → block device appears
    pub fn run(&mut self, store: &mut PageStore) -> Result<(), DaemonError> {
        info!(queue_depth = self.queue_depth, "Starting ublk daemon (FIX B+F applied)");
        debug!(char_fd = self.char_fd, "Daemon initialized");

        // Step 1: Submit all FETCH commands
        info!(queue_depth = self.queue_depth, "Submitting initial FETCH commands");
        for tag in 0..self.queue_depth {
            self.submit_fetch(tag)?;
        }

        // Sync and submit to kernel
        self.ring.submission().sync();
        let submitted = self.ring.submit().map_err(DaemonError::Submit)?;
        info!(submitted, "FETCH commands submitted to io_uring");

        // Step 2: Spawn START_DEV thread with delay
        // This thread waits to ensure the main thread is blocked in submit_and_wait
        // before calling START_DEV
        let mut ctrl_clone = self.ctrl.clone_handle()?;
        let ready_fd = self.ready_fd;
        let dev_id = self.ctrl.dev_id();
        let block_dev_path = self.ctrl.block_dev_path();

        std::thread::spawn(move || {
            // Wait for main thread to enter submit_and_wait and block
            // 200ms should be more than enough for the main thread to enter the kernel
            debug!("START_DEV thread: waiting for io_uring to block...");
            std::thread::sleep(Duration::from_millis(200));

            debug!("START_DEV thread: calling START_DEV ioctl");
            match ctrl_clone.start() {
                Ok(()) => {
                    info!(block_dev = %block_dev_path, "START_DEV succeeded - block device ready!");
                    if let Some(fd) = ready_fd {
                        let val = (dev_id as u64) + 1;
                        unsafe {
                            libc::write(fd, &val as *const u64 as *const libc::c_void, 8);
                        }
                    }
                }
                Err(e) => {
                    warn!(error = %e, "START_DEV failed");
                }
            }
        });

        // Step 3: Main thread enters submit_and_wait loop IMMEDIATELY
        // This is critical - we must be blocked in the kernel when START_DEV is called
        info!(block_dev = %self.ctrl.block_dev_path(), "Entering I/O loop (waiting for START_DEV)");

        loop {
            if self.stop.load(Ordering::Relaxed) {
                info!("Stop signal received");
                return Err(DaemonError::Stopped);
            }

            // Block waiting for completions - this is where we must be when START_DEV is called
            match self.ring.submit_and_wait(1) {
                Ok(_) => {}
                Err(e) if e.raw_os_error() == Some(libc::EINTR) => continue,
                Err(e) => return Err(DaemonError::Submit(e)),
            }

            // Process completions
            let tags: Vec<(u16, i32)> = {
                let cq = self.ring.completion();
                cq.map(|cqe| (cqe.user_data() as u16, cqe.result())).collect()
            };

            for (tag, result) in tags {
                if result < 0 {
                    if result == -libc::ENODEV {
                        info!(tag, "Device stopped (ENODEV)");
                        return Ok(());
                    }
                    warn!(tag, result, "I/O completion error, resubmitting FETCH");
                    self.submit_fetch(tag)?;
                    continue;
                }

                let iod = self.get_iod(tag);
                let op = (iod.op_flags & 0xff) as u8;
                let start_sector = iod.start_sector;
                let nr_sectors = iod.nr_sectors;

                debug!(tag, op, start_sector, nr_sectors, "Processing I/O");

                let len = (nr_sectors as usize) * SECTOR_SIZE as usize;
                let char_fd = self.char_fd;
                let data_buf_ptr = unsafe {
                    self.data_buf.add((tag as usize) * (self.max_io_size as usize))
                };

                let io_result = match op {
                    UBLK_IO_OP_READ => {
                        let buf = unsafe { std::slice::from_raw_parts_mut(data_buf_ptr, len) };
                        match store.read(start_sector, buf) {
                            Ok(n) => {
                                let offset = ublk_user_copy_offset(0, tag, 0);
                                let fd = unsafe { BorrowedFd::borrow_raw(char_fd) };
                                pwrite(fd, buf, offset)
                                    .map(|_| n)
                                    .map_err(|e| std::io::Error::from_raw_os_error(e as i32))
                            }
                            Err(e) => Err(e),
                        }
                    }
                    UBLK_IO_OP_WRITE => {
                        let buf = unsafe { std::slice::from_raw_parts_mut(data_buf_ptr, len) };
                        let offset = ublk_user_copy_offset(0, tag, 0);
                        let fd = unsafe { BorrowedFd::borrow_raw(char_fd) };
                        match pread(fd, buf, offset) {
                            Ok(_) => store.write(start_sector, buf),
                            Err(e) => Err(std::io::Error::from_raw_os_error(e as i32)),
                        }
                    }
                    UBLK_IO_OP_FLUSH => Ok(0),
                    UBLK_IO_OP_DISCARD => store.discard(start_sector, nr_sectors),
                    UBLK_IO_OP_WRITE_ZEROES => store.write_zeroes(start_sector, nr_sectors),
                    _ => {
                        warn!(op, "Unknown I/O operation");
                        Err(std::io::Error::from_raw_os_error(libc::ENOTSUP))
                    }
                };

                let result = match io_result {
                    Ok(n) => n as i32,
                    Err(e) => {
                        warn!(tag, error = %e, "I/O operation failed");
                        -e.raw_os_error().unwrap_or(libc::EIO)
                    }
                };

                self.submit_commit_and_fetch(tag, result)?;
            }

            self.ring.submit().map_err(DaemonError::Submit)?;
        }
    }

    /// Run daemon with BatchedPageStore for high-throughput batched compression.
    ///
    /// This method is similar to `run` but uses `BatchedPageStore` which:
    /// - Buffers writes until batch threshold (1000+ pages)
    /// - Uses GPU for large batches when available
    /// - Falls back to SIMD parallel compression for smaller batches
    ///
    /// Uses FIX B+F: Disabled SQPOLL, main thread enters io_uring before START_DEV.
    #[cfg(not(test))]
    pub fn run_batched(&mut self, store: &Arc<crate::daemon::BatchedPageStore>) -> Result<(), DaemonError> {
        info!(queue_depth = self.queue_depth, "Starting batched I/O loop (FIX B+F applied)");

        // Step 1: Submit initial FETCH commands
        for tag in 0..self.queue_depth {
            self.submit_fetch(tag)?;
        }
        self.ring.submission().sync();
        let submitted = self.ring.submit().map_err(DaemonError::Submit)?;
        info!(submitted, "Batched: FETCH commands submitted");

        // Step 2: Spawn START_DEV thread with delay (FIX F)
        let mut ctrl_clone = self.ctrl.clone_handle()?;
        let ready_fd = self.ready_fd;
        let dev_id = self.ctrl.dev_id();
        let block_dev_path = self.ctrl.block_dev_path();

        std::thread::spawn(move || {
            // Wait for main thread to enter submit_and_wait
            debug!("Batched START_DEV thread: waiting for io_uring to block...");
            std::thread::sleep(Duration::from_millis(200));

            debug!("Batched START_DEV thread: calling START_DEV ioctl");
            match ctrl_clone.start() {
                Ok(()) => {
                    info!(block_dev = %block_dev_path, "Batched START_DEV succeeded!");
                    if let Some(fd) = ready_fd {
                        let val = (dev_id as u64) + 1;
                        unsafe {
                            libc::write(fd, &val as *const u64 as *const libc::c_void, 8);
                        }
                    }
                }
                Err(e) => {
                    warn!(error = %e, "Batched START_DEV failed");
                }
            }
        });

        // Step 3: Enter I/O loop immediately (main thread blocks in kernel)
        info!(block_dev = %self.ctrl.block_dev_path(), "Batched: entering I/O loop");

        loop {
            if self.stop.load(Ordering::Relaxed) {
                info!("Stop signal received");
                return Err(DaemonError::Stopped);
            }

            match self.ring.submit_and_wait(1) {
                Ok(_) => {}
                Err(e) if e.raw_os_error() == Some(libc::EINTR) => continue,
                Err(e) => return Err(DaemonError::Submit(e)),
            }

            // Collect completions
            let tags: Vec<(u16, i32)> = {
                let cq = self.ring.completion();
                cq.map(|cqe| (cqe.user_data() as u16, cqe.result())).collect()
            };

            for (tag, result) in tags {
                if result < 0 {
                    if result == -libc::ENODEV {
                        info!(tag, "Device stopped");
                        return Ok(());
                    }
                    warn!(tag, result, "I/O completion error");
                    self.submit_fetch(tag)?;
                    continue;
                }

                let iod = self.get_iod(tag);
                let op = (iod.op_flags & 0xff) as u8;
                let start_sector = iod.start_sector;
                let nr_sectors = iod.nr_sectors;

                debug!(tag, op, start_sector, nr_sectors, "Processing batched I/O");

                let len = (nr_sectors as usize) * SECTOR_SIZE as usize;
                let char_fd = self.char_fd;
                let data_buf_ptr = unsafe {
                    self.data_buf.add((tag as usize) * (self.max_io_size as usize))
                };

                // Process I/O using BatchedPageStore
                let io_result = match op {
                    UBLK_IO_OP_READ => {
                        let buf = unsafe { std::slice::from_raw_parts_mut(data_buf_ptr, len) };
                        match store.read(start_sector, buf) {
                            Ok(n) => {
                                let offset = ublk_user_copy_offset(0, tag, 0);
                                let fd = unsafe { BorrowedFd::borrow_raw(char_fd) };
                                pwrite(fd, buf, offset)
                                    .map(|_| n)
                                    .map_err(|e| std::io::Error::from_raw_os_error(e as i32))
                            }
                            Err(e) => Err(e),
                        }
                    }
                    UBLK_IO_OP_WRITE => {
                        let buf = unsafe { std::slice::from_raw_parts_mut(data_buf_ptr, len) };
                        let offset = ublk_user_copy_offset(0, tag, 0);
                        let fd = unsafe { BorrowedFd::borrow_raw(char_fd) };
                        match pread(fd, buf, offset) {
                            Ok(_) => store.write(start_sector, buf),
                            Err(e) => Err(std::io::Error::from_raw_os_error(e as i32)),
                        }
                    }
                    UBLK_IO_OP_FLUSH => {
                        // Flush pending batch on FLUSH request
                        if let Err(e) = store.flush_batch() {
                            warn!("Flush batch failed: {}", e);
                        }
                        Ok(0)
                    }
                    UBLK_IO_OP_DISCARD => store.discard(start_sector, nr_sectors),
                    UBLK_IO_OP_WRITE_ZEROES => store.write_zeroes(start_sector, nr_sectors),
                    _ => {
                        warn!(op, "Unknown I/O operation");
                        Err(std::io::Error::from_raw_os_error(libc::ENOTSUP))
                    }
                };

                let result = match io_result {
                    Ok(n) => n as i32,
                    Err(e) => {
                        warn!(tag, error = %e, "Batched I/O operation failed");
                        -e.raw_os_error().unwrap_or(libc::EIO)
                    }
                };

                self.submit_commit_and_fetch(tag, result)?;
            }

            self.ring.submit().map_err(DaemonError::Submit)?;
        }
    }

    fn submit_fetch(&mut self, tag: u16) -> Result<(), DaemonError> {
        // With UBLK_F_USER_COPY, addr must be 0 (data via pread/pwrite)
        let io_cmd = UblkIoCmd {
            q_id: 0, // queue 0
            tag,
            result: -1, // libublk uses -1 for FETCH
            addr: 0,    // USER_COPY mode: kernel uses pread/pwrite for data
        };

        // Use ioctl-encoded opcode if kernel supports it, otherwise raw opcode
        let cmd_op = if self.use_ioctl_encode {
            UBLK_U_IO_FETCH_REQ
        } else {
            UBLK_IO_FETCH_REQ
        };

        eprintln!("[TRACE] submit_fetch: tag={}, fd={}, cmd_op=0x{:08X}, ioctl_encode={}",
                  tag, self.char_fd, cmd_op, self.use_ioctl_encode);

        // Use UringCmd16 like libublk - the 16-byte cmd field is sufficient for UblkIoCmd
        let io_cmd_bytes: [u8; 16] = unsafe { std::mem::transmute(io_cmd) };
        // Use Fd(char_fd) directly - simpler than fd registration
        let sqe = opcode::UringCmd16::new(types::Fd(self.char_fd), cmd_op)
            .cmd(io_cmd_bytes)
            .build()
            .user_data(tag as u64);

        unsafe {
            self.ring.submission()
                .push(&sqe)
                .map_err(|_| DaemonError::Submit(std::io::Error::from_raw_os_error(libc::ENOSPC)))?;
        }
        eprintln!("[TRACE]   pushed to submission queue OK");
        Ok(())
    }

    fn submit_commit_and_fetch(&mut self, tag: u16, result: i32) -> Result<(), DaemonError> {
        // With UBLK_F_USER_COPY, addr must be 0
        let io_cmd = UblkIoCmd {
            q_id: 0,
            tag,
            result,
            addr: 0,
        };

        // Use ioctl-encoded opcode if kernel supports it, otherwise raw opcode
        let cmd_op = if self.use_ioctl_encode {
            UBLK_U_IO_COMMIT_AND_FETCH_REQ
        } else {
            UBLK_IO_COMMIT_AND_FETCH_REQ
        };

        // Use UringCmd16 like libublk
        let io_cmd_bytes: [u8; 16] = unsafe { std::mem::transmute(io_cmd) };
        // Use Fd(char_fd) directly - simpler than fd registration
        let sqe = opcode::UringCmd16::new(types::Fd(self.char_fd), cmd_op)
            .cmd(io_cmd_bytes)
            .build()
            .user_data(tag as u64);

        unsafe {
            self.ring.submission()
                .push(&sqe)
                .map_err(|_| DaemonError::Submit(std::io::Error::from_raw_os_error(libc::ENOSPC)))?;
        }
        Ok(())
    }

    #[inline]
    fn get_iod(&self, tag: u16) -> &UblkIoDesc {
        unsafe {
            let ptr = self.iod_buf.add((tag as usize) * std::mem::size_of::<UblkIoDesc>()) as *const UblkIoDesc;
            &*ptr
        }
    }

    #[inline]
    fn get_data_buf(&mut self, tag: u16) -> &mut [u8] {
        unsafe {
            let ptr = self.data_buf.add((tag as usize) * (self.max_io_size as usize));
            std::slice::from_raw_parts_mut(ptr, self.max_io_size as usize)
        }
    }
}

#[cfg(not(test))]
impl Drop for UblkDaemon {
    fn drop(&mut self) {
        unsafe {
            munmap(self.iod_buf as *mut _, self.iod_buf_size);
            munmap(self.data_buf as *mut _, self.data_buf_size);
            libc::close(self.char_fd);
        }
    }
}

/// Start daemon with a new PageStore (per-page compression)
#[cfg(not(test))]
pub fn run_daemon(
    dev_id: i32,
    dev_size: u64,
    algorithm: trueno_zram_core::Algorithm,
    stop: Arc<AtomicBool>,
    ready_fd: Option<i32>,
) -> Result<(), DaemonError> {
    let config = DeviceConfig {
        dev_id,
        dev_size,
        ..Default::default()
    };

    let mut daemon = UblkDaemon::new(config, stop, ready_fd)?;
    let mut store = PageStore::new(dev_size, algorithm);

    daemon.run(&mut store)
}

/// Batched daemon configuration
#[derive(Debug, Clone)]
pub struct BatchedDaemonConfig {
    /// Device ID
    pub dev_id: i32,
    /// Device size in bytes
    pub dev_size: u64,
    /// Compression algorithm
    pub algorithm: trueno_zram_core::Algorithm,
    /// Batch threshold (pages before triggering compression)
    pub batch_threshold: usize,
    /// Flush timeout for partial batches
    pub flush_timeout_ms: u64,
    /// GPU batch size for optimal throughput
    pub gpu_batch_size: usize,
}

impl Default for BatchedDaemonConfig {
    fn default() -> Self {
        Self {
            dev_id: -1,
            dev_size: 1 << 30, // 1GB
            algorithm: trueno_zram_core::Algorithm::Lz4,
            batch_threshold: 1000,
            flush_timeout_ms: 10,
            gpu_batch_size: 4000,
        }
    }
}

/// Start daemon with BatchedPageStore for high-throughput compression.
///
/// This mode buffers pages and compresses them in batches:
/// - GPU for batches >= 1000 pages
/// - SIMD parallel for 100-999 pages
/// - SIMD sequential for < 100 pages
///
/// Target: >10 GB/s sequential write throughput.
#[cfg(not(test))]
pub fn run_daemon_batched(
    batch_config: BatchedDaemonConfig,
    stop: Arc<AtomicBool>,
    ready_fd: Option<i32>,
) -> Result<(), DaemonError> {
    let device_config = DeviceConfig {
        dev_id: batch_config.dev_id,
        dev_size: batch_config.dev_size,
        ..Default::default()
    };

    let mut daemon = UblkDaemon::new(device_config, stop.clone(), ready_fd)?;

    // Create batched page store
    let store_config = BatchConfig {
        batch_threshold: batch_config.batch_threshold,
        flush_timeout: Duration::from_millis(batch_config.flush_timeout_ms),
        gpu_batch_size: batch_config.gpu_batch_size,
    };
    let store = Arc::new(BatchedPageStore::with_config(batch_config.algorithm, store_config));

    // Spawn background flush thread
    let flush_handle = spawn_flush_thread(Arc::clone(&store));

    info!(
        "Starting batched daemon: batch_threshold={}, flush_timeout={}ms, gpu_batch_size={}",
        batch_config.batch_threshold,
        batch_config.flush_timeout_ms,
        batch_config.gpu_batch_size
    );

    // Run daemon with batched store
    let result = daemon.run_batched(&store);

    // Signal shutdown to flush thread
    store.shutdown();

    // Wait for flush thread to complete final flush
    if let Err(e) = flush_handle.join() {
        warn!("Flush thread panicked: {:?}", e);
    }

    // Log final stats
    let stats = store.stats();
    info!(
        "Batched daemon stats: pages={}, gpu_pages={}, simd_pages={}, batch_flushes={}, ratio={:.2}x",
        stats.pages_stored,
        stats.gpu_pages,
        stats.simd_pages,
        stats.batch_flushes,
        if stats.bytes_compressed > 0 {
            stats.bytes_stored as f64 / stats.bytes_compressed as f64
        } else {
            1.0
        }
    );

    result
}

// ============================================================================
// Mock infrastructure for testing without kernel access
// ============================================================================

/// Mock I/O descriptor for testing
#[cfg(test)]
#[derive(Debug, Clone)]
pub struct MockIoDesc {
    pub op: u8,
    pub nr_sectors: u32,
    pub start_sector: u64,
}

/// Mock daemon for testing I/O processing logic
#[cfg(test)]
pub struct MockUblkDaemon {
    pub dev_id: i32,
    pub queue_depth: u16,
    pub max_io_size: u32,
    pub pending_ios: Vec<MockIoDesc>,
    pub completed_ios: Vec<(u16, i32)>, // (tag, result)
    pub fetch_count: usize,
    pub commit_count: usize,
}

#[cfg(test)]
impl MockUblkDaemon {
    pub fn new(dev_id: i32, queue_depth: u16) -> Self {
        Self {
            dev_id,
            queue_depth,
            max_io_size: UBLK_MAX_IO_BUF_BYTES,
            pending_ios: Vec::new(),
            completed_ios: Vec::new(),
            fetch_count: 0,
            commit_count: 0,
        }
    }

    pub fn submit_fetch(&mut self, tag: u16) -> Result<(), DaemonError> {
        if tag >= self.queue_depth {
            return Err(DaemonError::Submit(std::io::Error::from_raw_os_error(libc::EINVAL)));
        }
        self.fetch_count += 1;
        Ok(())
    }

    pub fn submit_commit_and_fetch(&mut self, tag: u16, result: i32) -> Result<(), DaemonError> {
        if tag >= self.queue_depth {
            return Err(DaemonError::Submit(std::io::Error::from_raw_os_error(libc::EINVAL)));
        }
        self.completed_ios.push((tag, result));
        self.commit_count += 1;
        Ok(())
    }

    pub fn process_io(&mut self, io: MockIoDesc, store: &mut crate::daemon::PageStore) -> i32 {
        let start_sector = io.start_sector;
        let nr_sectors = io.nr_sectors;
        let len = (nr_sectors as usize) * SECTOR_SIZE as usize;

        let result = match io.op {
            UBLK_IO_OP_READ => {
                let mut buf = vec![0u8; len];
                match store.read(start_sector, &mut buf) {
                    Ok(n) => n as i32,
                    Err(e) => -e.raw_os_error().unwrap_or(libc::EIO),
                }
            }
            UBLK_IO_OP_WRITE => {
                let buf = vec![0xABu8; len]; // Mock data
                match store.write(start_sector, &buf) {
                    Ok(n) => n as i32,
                    Err(e) => -e.raw_os_error().unwrap_or(libc::EIO),
                }
            }
            UBLK_IO_OP_FLUSH => 0,
            UBLK_IO_OP_DISCARD => {
                match store.discard(start_sector, nr_sectors) {
                    Ok(n) => n as i32,
                    Err(e) => -e.raw_os_error().unwrap_or(libc::EIO),
                }
            }
            UBLK_IO_OP_WRITE_ZEROES => {
                match store.write_zeroes(start_sector, nr_sectors) {
                    Ok(n) => n as i32,
                    Err(e) => -e.raw_os_error().unwrap_or(libc::EIO),
                }
            }
            _ => -libc::ENOTSUP,
        };

        result
    }

    pub fn dev_id(&self) -> i32 { self.dev_id }

    pub fn block_dev_path(&self) -> String {
        format!("{}{}", UBLK_BLOCK_DEV_FMT, self.dev_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trueno_zram_core::Algorithm;

    // ========================================================================
    // MockUblkDaemon Tests
    // ========================================================================

    #[test]
    fn test_mock_daemon_new() {
        let daemon = MockUblkDaemon::new(0, 128);
        assert_eq!(daemon.dev_id(), 0);
        assert_eq!(daemon.queue_depth, 128);
        assert_eq!(daemon.max_io_size, UBLK_MAX_IO_BUF_BYTES);
    }

    #[test]
    fn test_mock_daemon_submit_fetch() {
        let mut daemon = MockUblkDaemon::new(0, 128);

        // Valid tags
        for tag in 0..128 {
            daemon.submit_fetch(tag).unwrap();
        }
        assert_eq!(daemon.fetch_count, 128);

        // Invalid tag
        assert!(daemon.submit_fetch(128).is_err());
    }

    #[test]
    fn test_mock_daemon_submit_commit_and_fetch() {
        let mut daemon = MockUblkDaemon::new(0, 64);

        daemon.submit_commit_and_fetch(0, 4096).unwrap();
        daemon.submit_commit_and_fetch(1, -5).unwrap();

        assert_eq!(daemon.commit_count, 2);
        assert_eq!(daemon.completed_ios.len(), 2);
        assert_eq!(daemon.completed_ios[0], (0, 4096));
        assert_eq!(daemon.completed_ios[1], (1, -5));
    }

    #[test]
    fn test_mock_daemon_process_read() {
        let mut daemon = MockUblkDaemon::new(0, 128);
        let mut store = crate::daemon::PageStore::new(1 << 30, Algorithm::Lz4);

        // Write some data first
        let data = vec![0xDEu8; 4096];
        store.write(0, &data).unwrap();

        // Process a read
        let io = MockIoDesc {
            op: UBLK_IO_OP_READ,
            nr_sectors: 8,
            start_sector: 0,
        };
        let result = daemon.process_io(io, &mut store);
        assert!(result > 0);
    }

    #[test]
    fn test_mock_daemon_process_write() {
        let mut daemon = MockUblkDaemon::new(0, 128);
        let mut store = crate::daemon::PageStore::new(1 << 30, Algorithm::Lz4);

        let io = MockIoDesc {
            op: UBLK_IO_OP_WRITE,
            nr_sectors: 8,
            start_sector: 0,
        };
        let result = daemon.process_io(io, &mut store);
        assert!(result > 0);
    }

    #[test]
    fn test_mock_daemon_process_flush() {
        let mut daemon = MockUblkDaemon::new(0, 128);
        let mut store = crate::daemon::PageStore::new(1 << 30, Algorithm::Lz4);

        let io = MockIoDesc {
            op: UBLK_IO_OP_FLUSH,
            nr_sectors: 0,
            start_sector: 0,
        };
        let result = daemon.process_io(io, &mut store);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_mock_daemon_process_discard() {
        let mut daemon = MockUblkDaemon::new(0, 128);
        let mut store = crate::daemon::PageStore::new(1 << 30, Algorithm::Lz4);

        // Write data first
        let data = vec![0xABu8; 4096];
        store.write(0, &data).unwrap();

        let io = MockIoDesc {
            op: UBLK_IO_OP_DISCARD,
            nr_sectors: 8,
            start_sector: 0,
        };
        let result = daemon.process_io(io, &mut store);
        assert!(result >= 0);
    }

    #[test]
    fn test_mock_daemon_process_write_zeroes() {
        let mut daemon = MockUblkDaemon::new(0, 128);
        let mut store = crate::daemon::PageStore::new(1 << 30, Algorithm::Lz4);

        let io = MockIoDesc {
            op: UBLK_IO_OP_WRITE_ZEROES,
            nr_sectors: 8,
            start_sector: 0,
        };
        let result = daemon.process_io(io, &mut store);
        assert!(result >= 0);
    }

    #[test]
    fn test_mock_daemon_process_unknown_op() {
        let mut daemon = MockUblkDaemon::new(0, 128);
        let mut store = crate::daemon::PageStore::new(1 << 30, Algorithm::Lz4);

        let io = MockIoDesc {
            op: 255, // Unknown
            nr_sectors: 8,
            start_sector: 0,
        };
        let result = daemon.process_io(io, &mut store);
        assert_eq!(result, -libc::ENOTSUP);
    }

    #[test]
    fn test_mock_daemon_block_dev_path() {
        let daemon = MockUblkDaemon::new(5, 128);
        assert_eq!(daemon.block_dev_path(), "/dev/ublkb5");
    }

    #[test]
    fn test_mock_daemon_io_loop_simulation() {
        let mut daemon = MockUblkDaemon::new(0, 128);
        let mut store = crate::daemon::PageStore::new(1 << 30, Algorithm::Lz4);

        // Submit initial fetches
        for tag in 0..daemon.queue_depth {
            daemon.submit_fetch(tag).unwrap();
        }
        assert_eq!(daemon.fetch_count, 128);

        // Simulate I/O completions
        let ios = vec![
            MockIoDesc { op: UBLK_IO_OP_WRITE, nr_sectors: 8, start_sector: 0 },
            MockIoDesc { op: UBLK_IO_OP_READ, nr_sectors: 8, start_sector: 0 },
            MockIoDesc { op: UBLK_IO_OP_DISCARD, nr_sectors: 8, start_sector: 8 },
            MockIoDesc { op: UBLK_IO_OP_FLUSH, nr_sectors: 0, start_sector: 0 },
        ];

        for (tag, io) in ios.into_iter().enumerate() {
            let result = daemon.process_io(io, &mut store);
            daemon.submit_commit_and_fetch(tag as u16, result).unwrap();
        }

        assert_eq!(daemon.commit_count, 4);
    }

    // ========================================================================
    // iod_buf_size Tests
    // ========================================================================

    #[test]
    fn test_iod_buf_size() {
        let depth = 128u16;
        let size = iod_buf_size(depth);
        // Should be page-aligned
        assert_eq!(size % 4096, 0);
        // Should be at least queue_depth * sizeof(UblkIoDesc)
        assert!(size >= (depth as usize) * std::mem::size_of::<UblkIoDesc>());
    }

    #[test]
    fn test_iod_buf_size_small_depth() {
        let size = iod_buf_size(1);
        assert_eq!(size % 4096, 0);
        assert!(size >= std::mem::size_of::<UblkIoDesc>());
    }

    #[test]
    fn test_iod_buf_size_various_depths() {
        for depth in [1, 8, 16, 32, 64, 128, 256, 512] {
            let size = iod_buf_size(depth);
            assert_eq!(size % 4096, 0, "Buffer must be page-aligned for depth {}", depth);
            assert!(
                size >= (depth as usize) * std::mem::size_of::<UblkIoDesc>(),
                "Buffer too small for depth {}", depth
            );
        }
    }

    #[test]
    fn test_iod_buf_size_boundary() {
        // Find a queue_depth that puts us just under page boundary
        let iod_size = std::mem::size_of::<UblkIoDesc>();
        let depth_for_one_page = (4096 / iod_size) as u16;

        // One less than full page
        let size = iod_buf_size(depth_for_one_page - 1);
        assert_eq!(size, 4096);

        // Exactly one page worth
        let size = iod_buf_size(depth_for_one_page);
        assert_eq!(size, 4096);

        // One more than page boundary
        let size = iod_buf_size(depth_for_one_page + 1);
        assert_eq!(size, 8192);
    }

    // ========================================================================
    // ublk_user_copy_offset Tests
    // ========================================================================

    #[test]
    fn test_user_copy_offset() {
        // Tag 0, queue 0, offset 0
        let offset = ublk_user_copy_offset(0, 0, 0);
        assert_eq!(offset as u64, UBLKSRV_IO_BUF_OFFSET);

        // Tag 1, queue 0, offset 0
        let offset = ublk_user_copy_offset(0, 1, 0);
        assert_eq!(offset as u64, UBLKSRV_IO_BUF_OFFSET + (1u64 << UBLK_TAG_OFF));
    }

    #[test]
    fn test_user_copy_offset_queue_id() {
        // Queue 1, tag 0, offset 0
        let offset = ublk_user_copy_offset(1, 0, 0);
        assert_eq!(offset as u64, UBLKSRV_IO_BUF_OFFSET + (1u64 << UBLK_QID_OFF));
    }

    #[test]
    fn test_user_copy_offset_with_offset() {
        // With a byte offset
        let offset = ublk_user_copy_offset(0, 0, 512);
        assert_eq!(offset as u64, UBLKSRV_IO_BUF_OFFSET + 512);
    }

    #[test]
    fn test_user_copy_offset_combined() {
        // Queue 2, tag 5, offset 1024
        let offset = ublk_user_copy_offset(2, 5, 1024);
        let expected = UBLKSRV_IO_BUF_OFFSET
            + (2u64 << UBLK_QID_OFF)
            + (5u64 << UBLK_TAG_OFF)
            + 1024;
        assert_eq!(offset as u64, expected);
    }

    // ========================================================================
    // DaemonError Tests
    // ========================================================================

    #[test]
    fn test_daemon_error_display_ctrl() {
        let ctrl_err = CtrlError::OpenCtrl(std::io::Error::from_raw_os_error(2));
        let err = DaemonError::Ctrl(ctrl_err);
        let msg = format!("{}", err);
        assert!(msg.contains("Control error"));
    }

    #[test]
    fn test_daemon_error_display_io_uring_create() {
        let err = DaemonError::IoUringCreate(std::io::Error::from_raw_os_error(12));
        let msg = format!("{}", err);
        assert!(msg.contains("io_uring creation failed"));
    }

    #[test]
    fn test_daemon_error_display_mmap() {
        let err = DaemonError::Mmap(std::io::Error::from_raw_os_error(12));
        let msg = format!("{}", err);
        assert!(msg.contains("Failed to mmap shared buffer"));
    }

    #[test]
    fn test_daemon_error_display_submit() {
        let err = DaemonError::Submit(std::io::Error::from_raw_os_error(28));
        let msg = format!("{}", err);
        assert!(msg.contains("io_uring submission failed"));
    }

    #[test]
    fn test_daemon_error_display_stopped() {
        let err = DaemonError::Stopped;
        let msg = format!("{}", err);
        assert!(msg.contains("Device stopped"));
    }

    #[test]
    fn test_daemon_error_debug() {
        let err = DaemonError::Stopped;
        let debug = format!("{:?}", err);
        assert!(debug.contains("Stopped"));
    }

    #[test]
    fn test_daemon_error_from_ctrl() {
        let ctrl_err = CtrlError::OpenCtrl(std::io::Error::from_raw_os_error(1));
        let daemon_err: DaemonError = ctrl_err.into();
        assert!(matches!(daemon_err, DaemonError::Ctrl(_)));
    }

    // ========================================================================
    // DaemonError Method Tests (pepita patterns)
    // ========================================================================

    #[test]
    fn test_daemon_error_is_retriable_queue_full() {
        let err = DaemonError::QueueFull;
        assert!(err.is_retriable());
    }

    #[test]
    fn test_daemon_error_is_retriable_would_block() {
        let err = DaemonError::WouldBlock;
        assert!(err.is_retriable());
    }

    #[test]
    fn test_daemon_error_not_retriable() {
        let err = DaemonError::Stopped;
        assert!(!err.is_retriable());
    }

    #[test]
    fn test_daemon_error_is_resource_error() {
        let err = DaemonError::QueueFull;
        assert!(err.is_resource_error());

        let err = DaemonError::Mmap(std::io::Error::from_raw_os_error(libc::ENOMEM));
        assert!(err.is_resource_error());
    }

    #[test]
    fn test_daemon_error_not_resource_error() {
        let err = DaemonError::Stopped;
        assert!(!err.is_resource_error());
    }

    #[test]
    fn test_daemon_error_to_errno_queue_full() {
        let err = DaemonError::QueueFull;
        assert_eq!(err.to_errno(), -libc::ENOSPC);
    }

    #[test]
    fn test_daemon_error_to_errno_would_block() {
        let err = DaemonError::WouldBlock;
        assert_eq!(err.to_errno(), -libc::EAGAIN);
    }

    #[test]
    fn test_daemon_error_to_errno_stopped() {
        let err = DaemonError::Stopped;
        assert_eq!(err.to_errno(), -libc::ENODEV);
    }

    #[test]
    fn test_daemon_error_to_errno_mmap() {
        let err = DaemonError::Mmap(std::io::Error::from_raw_os_error(libc::ENOMEM));
        assert_eq!(err.to_errno(), -libc::ENOMEM);
    }

    #[test]
    fn test_daemon_error_to_errno_submit() {
        let err = DaemonError::Submit(std::io::Error::from_raw_os_error(libc::EIO));
        assert_eq!(err.to_errno(), -libc::EIO);
    }

    #[test]
    fn test_daemon_error_all_variants_have_errno() {
        let ctrl_err = CtrlError::OpenCtrl(std::io::Error::from_raw_os_error(1));
        let errors: Vec<DaemonError> = vec![
            DaemonError::Ctrl(ctrl_err),
            DaemonError::IoUringCreate(std::io::Error::from_raw_os_error(libc::ENOMEM)),
            DaemonError::Mmap(std::io::Error::from_raw_os_error(libc::ENOMEM)),
            DaemonError::Submit(std::io::Error::from_raw_os_error(libc::EIO)),
            DaemonError::QueueFull,
            DaemonError::WouldBlock,
            DaemonError::Stopped,
        ];

        for err in errors {
            let errno = err.to_errno();
            assert!(errno < 0, "errno should be negative: {} for {:?}", errno, err);
            assert!(errno >= -4095, "errno out of range: {} for {:?}", errno, err);
        }
    }

    // ========================================================================
    // UblkIoCmd Tests
    // ========================================================================

    #[test]
    fn test_io_cmd_layout() {
        let io_cmd = UblkIoCmd {
            q_id: 0,
            tag: 5,
            result: 4096,
            addr: 0,
        };
        let bytes: [u8; 16] = unsafe { std::mem::transmute(io_cmd) };
        // Verify size is exactly 16 bytes for io_uring cmd field
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn test_io_cmd_transmute_roundtrip() {
        let io_cmd = UblkIoCmd {
            q_id: 3,
            tag: 127,
            result: -5,
            addr: 0x12345678ABCD,
        };
        let bytes: [u8; 16] = unsafe { std::mem::transmute(io_cmd) };
        let recovered: UblkIoCmd = unsafe { std::mem::transmute(bytes) };
        assert_eq!(recovered.q_id, 3);
        assert_eq!(recovered.tag, 127);
        assert_eq!(recovered.result, -5);
        assert_eq!(recovered.addr, 0x12345678ABCD);
    }

    // ========================================================================
    // UblkIoDesc Tests
    // ========================================================================

    #[test]
    fn test_io_desc_operation_types() {
        // Verify we can parse operation types from op_flags
        let desc = UblkIoDesc {
            op_flags: UBLK_IO_OP_READ as u32,
            nr_sectors: 8,
            start_sector: 0,
            addr: 0,
        };
        let op = (desc.op_flags & 0xff) as u8;
        assert_eq!(op, UBLK_IO_OP_READ);

        let desc = UblkIoDesc {
            op_flags: UBLK_IO_OP_WRITE as u32 | (1 << 16), // with some flags
            nr_sectors: 8,
            start_sector: 100,
            addr: 0,
        };
        let op = (desc.op_flags & 0xff) as u8;
        assert_eq!(op, UBLK_IO_OP_WRITE);
    }

    #[test]
    fn test_io_desc_sector_calculation() {
        let desc = UblkIoDesc {
            op_flags: UBLK_IO_OP_WRITE as u32,
            nr_sectors: 16,
            start_sector: 1024,
            addr: 0,
        };
        let byte_len = (desc.nr_sectors as usize) * SECTOR_SIZE as usize;
        assert_eq!(byte_len, 16 * 512);
        assert_eq!(desc.start_sector, 1024);
    }

    // ========================================================================
    // DeviceConfig with run_daemon Tests
    // ========================================================================

    #[test]
    fn test_device_config_for_daemon() {
        let config = DeviceConfig {
            dev_id: -1,
            dev_size: 1 << 30,
            ..Default::default()
        };
        assert_eq!(config.dev_id, -1); // -1 means auto-assign
        assert_eq!(config.dev_size, 1 << 30);
        assert!(config.flags & UBLK_F_USER_COPY != 0);
    }

    // ========================================================================
    // Additional DaemonError Tests for Coverage
    // ========================================================================

    #[test]
    fn test_daemon_error_to_errno_ctrl() {
        let ctrl_err = CtrlError::OpenCtrl(std::io::Error::from_raw_os_error(libc::ENOENT));
        let err = DaemonError::Ctrl(ctrl_err);
        assert_eq!(err.to_errno(), -libc::ENODEV);
    }

    #[test]
    fn test_daemon_error_to_errno_io_uring_create_specific() {
        let err = DaemonError::IoUringCreate(std::io::Error::from_raw_os_error(libc::ENOSYS));
        assert_eq!(err.to_errno(), -libc::ENOSYS);
    }

    #[test]
    fn test_daemon_error_to_errno_io_uring_create_default() {
        let err = DaemonError::IoUringCreate(std::io::Error::new(
            std::io::ErrorKind::Other,
            "custom",
        ));
        assert_eq!(err.to_errno(), -libc::ENOMEM);
    }

    #[test]
    fn test_daemon_error_to_errno_mmap_default() {
        let err = DaemonError::Mmap(std::io::Error::new(
            std::io::ErrorKind::Other,
            "custom",
        ));
        assert_eq!(err.to_errno(), -libc::ENOMEM);
    }

    #[test]
    fn test_daemon_error_to_errno_submit_default() {
        let err = DaemonError::Submit(std::io::Error::new(
            std::io::ErrorKind::Other,
            "custom",
        ));
        assert_eq!(err.to_errno(), -libc::EIO);
    }

    #[test]
    fn test_daemon_error_is_resource_io_uring_create() {
        let err = DaemonError::IoUringCreate(std::io::Error::from_raw_os_error(libc::ENOMEM));
        assert!(err.is_resource_error());
    }

    #[test]
    fn test_daemon_error_not_retriable_ctrl() {
        let ctrl_err = CtrlError::OpenCtrl(std::io::Error::from_raw_os_error(1));
        let err = DaemonError::Ctrl(ctrl_err);
        assert!(!err.is_retriable());
    }

    #[test]
    fn test_daemon_error_not_retriable_mmap() {
        let err = DaemonError::Mmap(std::io::Error::from_raw_os_error(libc::ENOMEM));
        assert!(!err.is_retriable());
    }

    #[test]
    fn test_daemon_error_not_retriable_submit() {
        let err = DaemonError::Submit(std::io::Error::from_raw_os_error(libc::EIO));
        assert!(!err.is_retriable());
    }

    #[test]
    fn test_daemon_error_not_retriable_io_uring_create() {
        let err = DaemonError::IoUringCreate(std::io::Error::from_raw_os_error(libc::ENOMEM));
        assert!(!err.is_retriable());
    }

    #[test]
    fn test_daemon_error_not_resource_ctrl() {
        let ctrl_err = CtrlError::OpenCtrl(std::io::Error::from_raw_os_error(1));
        let err = DaemonError::Ctrl(ctrl_err);
        assert!(!err.is_resource_error());
    }

    #[test]
    fn test_daemon_error_not_resource_submit() {
        let err = DaemonError::Submit(std::io::Error::from_raw_os_error(libc::EIO));
        assert!(!err.is_resource_error());
    }

    #[test]
    fn test_daemon_error_not_resource_would_block() {
        let err = DaemonError::WouldBlock;
        assert!(!err.is_resource_error());
    }

    #[test]
    fn test_daemon_error_display_queue_full() {
        let err = DaemonError::QueueFull;
        let msg = format!("{}", err);
        assert!(msg.contains("Queue full"));
    }

    #[test]
    fn test_daemon_error_display_would_block() {
        let err = DaemonError::WouldBlock;
        let msg = format!("{}", err);
        assert!(msg.contains("Would block"));
    }
}
