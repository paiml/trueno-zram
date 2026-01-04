//! UblkTarget - Pure Rust ublk block device with compression
//!
//! Implements a compressed RAM block device using the Linux ublk interface.
//! Uses trueno-zram-core for SIMD-accelerated LZ4/Zstd compression.

use crate::daemon::PageStore;
use anyhow::{Context, Result};
use io_uring::IoUring;
use libublk::ctrl::UblkCtrlBuilder;
use libublk::ctrl_async::UblkCtrlAsync;
use libublk::helpers::IoBuf;
use libublk::io::{UblkDev, UblkQueue};
use libublk::{BufDesc, UblkError, UblkFlags};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use trueno_zram_core::{CompressorBuilder, PageCompressor, PAGE_SIZE};

/// Target configuration for ublk device.
#[derive(Clone, Debug)]
pub struct TargetConfig {
    /// Device size in bytes.
    pub size: u64,
    /// Compression algorithm.
    pub algorithm: trueno_zram_core::Algorithm,
    /// Number of I/O queues.
    pub nr_queues: u16,
    /// Queue depth per queue.
    pub queue_depth: u16,
    /// Entropy threshold for skipping compression.
    pub entropy_threshold: f64,
    /// Device name prefix.
    pub name: String,
}

impl Default for TargetConfig {
    fn default() -> Self {
        Self {
            size: 1 << 30, // 1GB default
            algorithm: trueno_zram_core::Algorithm::Lz4,
            nr_queues: 1,
            queue_depth: 128,
            entropy_threshold: 7.5,
            name: "trueno".to_string(),
        }
    }
}

/// Runtime statistics for the ublk target.
#[derive(Debug, Default)]
pub struct TargetStats {
    pub reads: AtomicU64,
    pub writes: AtomicU64,
    pub discards: AtomicU64,
    pub read_bytes: AtomicU64,
    pub write_bytes: AtomicU64,
    pub errors: AtomicU64,
}

impl TargetStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_read(&self, bytes: u64) {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.read_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn record_write(&self, bytes: u64) {
        self.writes.fetch_add(1, Ordering::Relaxed);
        self.write_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn record_discard(&self) {
        self.discards.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }
}

/// Handle a single I/O operation.
///
/// This is the core I/O handler that compresses/decompresses pages.
fn handle_io(
    q: &UblkQueue,
    tag: u16,
    io_buf: &mut [u8],
    store: &mut PageStore,
    stats: &TargetStats,
) -> i32 {
    let iod = q.get_iod(tag);
    let sector = iod.start_sector;
    let bytes = (iod.nr_sectors << 9) as usize;
    let op = iod.op_flags & 0xff;

    // Validate I/O size
    if bytes > io_buf.len() {
        stats.record_error();
        return -libc::EINVAL;
    }

    match op {
        libublk::sys::UBLK_IO_OP_READ => {
            // Read and decompress pages
            let pages = bytes.div_ceil(PAGE_SIZE);
            for i in 0..pages {
                let page_sector = sector + (i * (PAGE_SIZE / 512)) as u64;
                let offset = i * PAGE_SIZE;
                let page_buf = &mut io_buf[offset..offset + PAGE_SIZE];

                if let Err(e) = store.load(page_sector, page_buf) {
                    tracing::error!("Read error at sector {}: {}", page_sector, e);
                    stats.record_error();
                    return -libc::EIO;
                }
            }
            stats.record_read(bytes as u64);
            bytes as i32
        }
        libublk::sys::UBLK_IO_OP_WRITE => {
            // Compress and write pages
            let pages = bytes.div_ceil(PAGE_SIZE);
            for i in 0..pages {
                let page_sector = sector + (i * (PAGE_SIZE / 512)) as u64;
                let offset = i * PAGE_SIZE;
                let page_buf = &io_buf[offset..offset + PAGE_SIZE];

                if let Err(e) = store.store(page_sector, page_buf) {
                    tracing::error!("Write error at sector {}: {}", page_sector, e);
                    stats.record_error();
                    return -libc::EIO;
                }
            }
            stats.record_write(bytes as u64);
            bytes as i32
        }
        libublk::sys::UBLK_IO_OP_DISCARD => {
            // Discard pages (free memory)
            let pages = bytes.div_ceil(PAGE_SIZE);
            for i in 0..pages {
                let page_sector = sector + (i * (PAGE_SIZE / 512)) as u64;
                store.remove(page_sector);
            }
            stats.record_discard();
            bytes as i32
        }
        libublk::sys::UBLK_IO_OP_FLUSH => {
            // Flush is a no-op for RAM storage
            0
        }
        _ => {
            stats.record_error();
            -libc::ENOTSUPP
        }
    }
}

/// Async I/O task for a single tag.
async fn io_task(
    q: &UblkQueue<'_>,
    tag: u16,
    store_ptr: *mut PageStore,
    stats: &TargetStats,
) -> std::result::Result<(), UblkError> {
    let buf_size = q.dev.dev_info.max_io_buf_bytes as usize;
    let mut buffer = IoBuf::<u8>::new(buf_size);

    // Submit initial prep command
    q.submit_io_prep_cmd(tag, BufDesc::Slice(buffer.as_slice()), 0, Some(&buffer))
        .await?;

    loop {
        // SAFETY: store_ptr is valid for the lifetime of the device
        let store = unsafe { &mut *store_ptr };
        let io_slice = buffer.as_mut_slice();
        let res = handle_io(q, tag, io_slice, store, stats);

        // Commit and fetch next command
        q.submit_io_commit_cmd(tag, BufDesc::Slice(buffer.as_slice()), res)
            .await?;
    }
}

/// Create async UblkCtrl.
fn create_ublk_ctrl_async(
    dev_id: i32,
    config: &TargetConfig,
    for_add: bool,
) -> std::result::Result<UblkCtrlAsync, UblkError> {
    let dev_flags = if for_add {
        UblkFlags::UBLK_DEV_F_ADD_DEV
    } else {
        UblkFlags::UBLK_DEV_F_RECOVER_DEV
    };

    ublk_uring_run_async_task(|| async move {
        UblkCtrlBuilder::default()
            .name(&config.name)
            .id(dev_id)
            .nr_queues(config.nr_queues)
            .depth(config.queue_depth)
            .dev_flags(dev_flags)
            .ctrl_flags(libublk::sys::UBLK_F_USER_RECOVERY as u64)
            .build_async()
            .await
    })
}

/// Run async task with io_uring.
fn ublk_uring_run_async_task<F, Fut, T>(f: F) -> std::result::Result<T, UblkError>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = std::result::Result<T, UblkError>>,
{
    // Initialize task ring if needed
    libublk::io::ublk_init_task_ring(|cell| {
        if cell.get().is_none() {
            let ring = IoUring::<io_uring::squeue::Entry, io_uring::cqueue::Entry>::builder()
                .setup_cqsize(128)
                .setup_coop_taskrun()
                .build(128)
                .map_err(UblkError::IOError)?;

            cell.set(RefCell::new(ring))
                .map_err(|_| UblkError::OtherError(-libc::EEXIST))?;
        }
        Ok(())
    })?;

    smol::block_on(f())
}

/// Poll and handle both queue ring and control ring.
async fn poll_and_handle_rings<R, I>(
    run_ops: R,
    is_done: I,
    _check_done: bool,
) -> std::result::Result<(), UblkError>
where
    R: Fn(),
    I: Fn() -> bool,
{
    use std::fs::File;
    use std::os::fd::{AsRawFd, FromRawFd};

    let create_async_wrapper =
        |fd: i32| -> std::result::Result<smol::Async<File>, UblkError> {
            let file = unsafe { File::from_raw_fd(fd) };
            smol::Async::new(file).map_err(|_| UblkError::OtherError(-libc::EINVAL))
        };

    let queue_fd = libublk::io::with_task_io_ring(|ring| ring.as_raw_fd());
    let ctrl_fd = libublk::ctrl::with_ctrl_ring(|ring| ring.as_raw_fd());
    let _async_queue = create_async_wrapper(queue_fd)?;
    let _async_ctrl = create_async_wrapper(ctrl_fd)?;

    loop {
        // Submit and wait on both rings
        libublk::io::with_task_io_ring_mut(|ring| ring.submit_and_wait(0))?;
        libublk::ctrl::with_ctrl_ring_mut(|ring| ring.submit_and_wait(0))?;

        // Run operations
        run_ops();

        // Reap events
        libublk::io::with_task_io_ring_mut(|ring| {
            libublk::uring_async::ublk_reap_events_with_handler(ring, |_cqe| {
                libublk::uring_async::ublk_wake_task(_cqe.user_data());
            })
        })?;

        libublk::ctrl::with_ctrl_ring_mut(|ring| {
            libublk::uring_async::ublk_reap_events_with_handler(ring, |_cqe| {
                libublk::uring_async::ublk_wake_task(_cqe.user_data());
            })
        })?;

        if is_done() {
            break;
        }
    }

    Ok(())
}

/// Start the ublk daemon for a device.
pub fn start_daemon(
    dev_id: i32,
    config: TargetConfig,
    for_recovery: bool,
) -> Result<()> {
    tracing::info!(
        dev_id = dev_id,
        size = config.size,
        algorithm = ?config.algorithm,
        "Starting trueno-ublk daemon"
    );

    // Create compressor
    let compressor = CompressorBuilder::new()
        .algorithm(config.algorithm)
        .build()
        .context("Failed to create compressor")?;

    // Create page store
    let mut store = PageStore::new(Arc::from(compressor), config.entropy_threshold);
    let store_ptr = &mut store as *mut PageStore;

    // Create stats
    let stats = Arc::new(TargetStats::new());

    // Create ublk control
    let ctrl = Rc::new(
        create_ublk_ctrl_async(dev_id, &config, !for_recovery)
            .context("Failed to create ublk control")?,
    );

    tracing::info!(
        dev_id = ctrl.dev_info().dev_id,
        "ublk control created"
    );

    // Initialize device
    let tgt_init = |dev: &mut UblkDev| {
        dev.set_default_params(config.size);
        Ok(())
    };

    let dev_rc = Arc::new(
        UblkDev::new_async(&config.name, tgt_init, &ctrl)
            .map_err(|e| anyhow::anyhow!("Failed to create ublk device: {}", e))?,
    );

    let q_rc = Rc::new(
        UblkQueue::new(0, &dev_rc)
            .map_err(|e| anyhow::anyhow!("Failed to create ublk queue: {}", e))?,
    );

    let exec_rc = Rc::new(smol::LocalExecutor::new());
    let exec = exec_rc.clone();

    // Spawn I/O tasks for each tag
    let mut tasks = Vec::new();
    let queue_depth = ctrl.dev_info().queue_depth;

    for tag in 0..queue_depth {
        let q_clone = q_rc.clone();
        let stats_clone = stats.clone();

        tasks.push(exec.spawn(async move {
            match io_task(&q_clone, tag, store_ptr, &stats_clone).await {
                Err(UblkError::QueueIsDown) | Ok(_) => {}
                Err(e) => tracing::error!("I/O task failed for tag {}: {}", tag, e),
            }
        }));
    }

    // Spawn control task
    let ctrl_clone = ctrl.clone();
    let dev_clone = dev_rc.clone();
    tasks.push(exec.spawn(async move {
        let tid = unsafe { libc::gettid() };
        if let Err(e) = ctrl_clone.configure_queue_async(&dev_clone, 0, tid).await {
            tracing::error!("Failed to configure queue: {}", e);
            return;
        }

        if let Err(e) = ctrl_clone.start_dev_async(&dev_clone).await {
            tracing::error!("Failed to start device: {}", e);
            return;
        }

        tracing::info!(
            path = format!("/dev/ublkb{}", ctrl_clone.dev_info().dev_id),
            "Device started"
        );
    }));

    // Run event loop
    smol::block_on(exec_rc.run(async move {
        let run_ops = || {
            while exec.try_tick() {}
        };
        let done = || tasks.iter().all(|task| task.is_finished());

        if let Err(e) = poll_and_handle_rings(run_ops, done, false).await {
            tracing::error!("Event loop failed: {}", e);
        }
    }));

    Ok(())
}

/// Stop a ublk device.
pub fn stop_device(dev_id: i32) -> Result<()> {
    let ctrl = libublk::ctrl::UblkCtrl::new_simple(dev_id)
        .map_err(|e| anyhow::anyhow!("Failed to open device {}: {}", dev_id, e))?;

    ctrl.del_dev()
        .map_err(|e| anyhow::anyhow!("Failed to delete device {}: {}", dev_id, e))?;

    tracing::info!(dev_id = dev_id, "Device stopped");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_config_default() {
        let config = TargetConfig::default();
        assert_eq!(config.size, 1 << 30);
        assert_eq!(config.nr_queues, 1);
        assert_eq!(config.queue_depth, 128);
    }

    #[test]
    fn test_target_stats() {
        let stats = TargetStats::new();

        stats.record_read(4096);
        stats.record_write(8192);
        stats.record_discard();
        stats.record_error();

        assert_eq!(stats.reads.load(Ordering::Relaxed), 1);
        assert_eq!(stats.writes.load(Ordering::Relaxed), 1);
        assert_eq!(stats.read_bytes.load(Ordering::Relaxed), 4096);
        assert_eq!(stats.write_bytes.load(Ordering::Relaxed), 8192);
        assert_eq!(stats.discards.load(Ordering::Relaxed), 1);
        assert_eq!(stats.errors.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_target_stats_concurrent() {
        use std::thread;

        let stats = Arc::new(TargetStats::new());
        let mut handles = vec![];

        for _ in 0..4 {
            let s = stats.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..1000 {
                    s.record_read(4096);
                    s.record_write(4096);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(stats.reads.load(Ordering::Relaxed), 4000);
        assert_eq!(stats.writes.load(Ordering::Relaxed), 4000);
    }
}
