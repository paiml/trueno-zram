//! ublk daemon I/O loop
//!
//! Uses io_uring to handle block device I/O.

#[cfg(not(test))]
use crate::daemon::PageStore;
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

        // Use regular 64-byte SQE like libublk does - UringCmd16 works for ublk
        // Enable SQPOLL for kernel-side submission queue polling - this allows
        // the kernel to process our FETCH commands while we block on START_DEV
        let ring: IoUring = IoUring::builder()
            .setup_sqpoll(100) // Poll for 100ms before sleeping
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

    pub fn run(&mut self, store: &mut PageStore) -> Result<(), DaemonError> {
        eprintln!("[TRACE] run() starting, queue_depth={}", self.queue_depth);
        eprintln!("[TRACE] char_fd={}, ring entries={}", self.char_fd, self.queue_depth * 2);

        info!(queue_depth = self.queue_depth, "Submitting initial fetches");
        for tag in 0..self.queue_depth {
            eprintln!("[TRACE] Queuing FETCH for tag {}", tag);
            self.submit_fetch(tag)?;
        }
        eprintln!("[TRACE] All {} FETCH commands queued", self.queue_depth);
        debug!("Initial fetches queued, submitting to ring");

        // Submit and trigger kernel to process all SQEs
        // Note: FETCH commands don't complete (they wait for I/O), but they're processed
        eprintln!("[TRACE] Syncing submission queue...");
        self.ring.submission().sync();
        eprintln!("[TRACE] Calling ring.submit()...");
        let submitted = self.ring.submit().map_err(DaemonError::Submit)?;
        eprintln!("[TRACE] ring.submit() returned: {} submitted", submitted);
        debug!(submitted, "Ring submitted");

        // Force kernel to process by entering and exiting - this ensures all SQEs are seen
        // We use submit() again to trigger io_uring processing without waiting for completions
        for i in 0..3 {
            std::thread::sleep(std::time::Duration::from_millis(50));
            let n = self.ring.submit().unwrap_or(0);
            eprintln!("[TRACE] Extra submit {}: {} submitted", i, n);

            // Check for any completions (errors from rejected commands)
            self.ring.completion().sync();
            let cq_len = self.ring.completion().len();
            eprintln!("[TRACE]   CQ len after sync: {}", cq_len);
            for cqe in self.ring.completion() {
                eprintln!("[TRACE]   CQE: user_data={}, result={}", cqe.user_data(), cqe.result());
                if cqe.result() < 0 {
                    return Err(DaemonError::Submit(std::io::Error::from_raw_os_error(-cqe.result())));
                }
            }
        }

        // Force io_uring to actually process our FETCH submissions by entering the kernel
        // The kernel's ublk driver requires the io_uring to have processed the FETCH commands
        // before START_DEV can complete
        eprintln!("[TRACE] Entering io_uring to process FETCH commands...");
        match self.ring.submit_and_wait(0) {
            Ok(n) => eprintln!("[TRACE]   submit_and_wait(0) returned {} completions", n),
            Err(e) => eprintln!("[TRACE]   submit_and_wait(0) error: {}", e),
        }

        eprintln!("[TRACE] Spawning thread to call START_DEV...");
        let mut ctrl_clone = self.ctrl.clone_handle()?; // We need a way to clone the ctrl handle or pass it
        let ready_fd = self.ready_fd;
        let dev_id = self.ctrl.dev_id();
        std::thread::spawn(move || {
            debug!("Background thread calling START_DEV");
            if let Err(e) = ctrl_clone.start() {
                warn!("Background START_DEV failed: {}", e);
            } else {
                info!("Background START_DEV returned OK!");
                if let Some(fd) = ready_fd {
                    // Signal parent that device is ready (write dev_id + 1)
                    let val = (dev_id as u64) + 1;
                    unsafe {
                        libc::write(fd, &val as *const u64 as *const libc::c_void, 8);
                    }
                }
            }
        });

        info!(block_dev = %self.ctrl.block_dev_path(), "Device start initiated in background");

        eprintln!("[TRACE] About to enter main I/O loop...");
        loop {
            eprintln!("[TRACE] Loop iteration: checking stop flag...");
            if self.stop.load(Ordering::Relaxed) {
                info!("Stop signal received");
                return Err(DaemonError::Stopped);
            }

            eprintln!("[TRACE] Calling submit_and_wait(1)...");
            match self.ring.submit_and_wait(1) {
                Ok(n) => {
                    eprintln!("[TRACE] submit_and_wait(1) returned Ok({})", n);
                }
                Err(e) if e.raw_os_error() == Some(libc::EINTR) => {
                    eprintln!("[TRACE] submit_and_wait(1) EINTR, continuing");
                    continue;
                }
                Err(e) => {
                    eprintln!("[TRACE] submit_and_wait(1) error: {} (raw_os_error={:?})", e, e.raw_os_error());
                    return Err(DaemonError::Submit(e));
                }
            }

            // Process completions - collect tags first to avoid borrow issues
            let tags: Vec<(u16, i32)> = {
                let cq = self.ring.completion();
                cq.map(|cqe| (cqe.user_data() as u16, cqe.result())).collect()
            };
            eprintln!("[TRACE] Processing {} completions", tags.len());

            for (tag, result) in tags {
                eprintln!("[TRACE]   CQE tag={}, result={}", tag, result);
                if result < 0 {
                    if result == -libc::ENODEV {
                        eprintln!("[TRACE]   ENODEV received for tag {} - device stopped!", tag);
                        info!(tag, "Device stopped");
                        return Ok(());
                    }
                    eprintln!("[TRACE]   Error result={}, resubmitting FETCH", result);
                    warn!(tag, result, "I/O completion error");
                    self.submit_fetch(tag)?;
                    continue;
                }

                let iod = self.get_iod(tag);
                let op = (iod.op_flags & 0xff) as u8;
                let start_sector = iod.start_sector;
                let nr_sectors = iod.nr_sectors;

                eprintln!("[TRACE]   IOD: op={}, start_sector={}, nr_sectors={}, op_flags=0x{:X}",
                          op, start_sector, nr_sectors, iod.op_flags);
                debug!(tag, op, start_sector, nr_sectors, "Processing I/O");

                // Get buffer and fd before match to avoid borrow issues
                let len = (nr_sectors as usize) * SECTOR_SIZE as usize;
                let char_fd = self.char_fd;
                let data_buf_ptr = unsafe {
                    self.data_buf.add((tag as usize) * (self.max_io_size as usize))
                };

                let io_result = match op {
                    UBLK_IO_OP_READ => {
                        // Read from store, then pwrite to kernel
                        let buf = unsafe { std::slice::from_raw_parts_mut(data_buf_ptr, len) };
                        match store.read(start_sector, buf) {
                            Ok(n) => {
                                // Write data to kernel via pwrite
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
                        // pread from kernel, then write to store
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
                    Ok(n) => {
                        eprintln!("[TRACE]   I/O succeeded: {} bytes", n);
                        n as i32
                    }
                    Err(e) => {
                        eprintln!("[TRACE]   I/O failed: {}", e);
                        warn!(tag, error = %e, "I/O operation failed");
                        -e.raw_os_error().unwrap_or(libc::EIO)
                    }
                };

                eprintln!("[TRACE]   Submitting COMMIT_AND_FETCH for tag={}, result={}", tag, result);
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

/// Start daemon with a new PageStore
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
