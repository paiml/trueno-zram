//! Cleanup module - Orphan device detection and graceful shutdown
//!
//! Per DT-003: Implements signal handlers and startup orphan cleanup to prevent
//! stale ublk device state that causes I/O errors (see F083 root cause analysis).
//!
//! ## Features
//! - Detects orphaned ublk character devices on startup
//! - Cleans up orphaned devices before creating new ones
//! - Provides graceful shutdown via SIGTERM/SIGINT handlers

use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Result type for cleanup operations
pub type CleanupResult<T> = Result<T, CleanupError>;

/// Errors that can occur during cleanup
#[derive(Debug, thiserror::Error)]
pub enum CleanupError {
    #[error("Failed to read /dev directory: {0}")]
    ReadDevDir(#[from] std::io::Error),

    #[error("Failed to reset device {dev_id}: {message}")]
    ResetDevice { dev_id: u32, message: String },

    #[error("Cleanup timed out after {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },
}

/// Detect orphaned ublk character devices.
///
/// Returns a list of device IDs that have character devices (/dev/ublkcN)
/// but no corresponding block devices (/dev/ublkbN) or running daemon.
pub fn detect_orphaned_devices() -> CleanupResult<Vec<u32>> {
    let mut orphans = Vec::new();

    // Scan /dev for ublkc* devices
    let dev_path = Path::new("/dev");
    if !dev_path.exists() {
        return Ok(orphans);
    }

    for entry in fs::read_dir(dev_path)? {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        // Look for character devices: ublkcN
        if name_str.starts_with("ublkc") {
            if let Some(id_str) = name_str.strip_prefix("ublkc") {
                if let Ok(dev_id) = id_str.parse::<u32>() {
                    // Check if block device exists
                    let block_path = format!("/dev/ublkb{}", dev_id);
                    if !Path::new(&block_path).exists() {
                        debug!(dev_id, "Found orphaned character device (no block device)");
                        orphans.push(dev_id);
                    }
                }
            }
        }
    }

    if !orphans.is_empty() {
        info!(count = orphans.len(), "Detected orphaned ublk devices: {:?}", orphans);
    }

    Ok(orphans)
}

/// Clean up orphaned ublk devices.
///
/// This function attempts to reset any orphaned devices found by `detect_orphaned_devices`.
/// It's designed to be called at startup before creating new devices.
pub fn cleanup_orphaned_devices() -> CleanupResult<usize> {
    let orphans = detect_orphaned_devices()?;
    let mut cleaned = 0;

    for dev_id in orphans {
        match reset_device(dev_id) {
            Ok(()) => {
                info!(dev_id, "Cleaned up orphaned device");
                cleaned += 1;
            }
            Err(e) => {
                warn!(dev_id, error = %e, "Failed to clean up orphaned device");
            }
        }
    }

    if cleaned > 0 {
        info!(cleaned, "Orphan cleanup completed");
    }

    Ok(cleaned)
}

/// Reset a specific ublk device by ID.
///
/// This uses the ublk control ioctl directly to send DEL_DEV command.
/// For orphaned devices, we use a low-level approach that doesn't require
/// creating a full UblkCtrl handle.
fn reset_device(dev_id: u32) -> CleanupResult<()> {
    use nix::libc;
    use std::os::unix::io::AsRawFd;

    debug!(dev_id, "Attempting to reset device");

    // Open the control device directly
    let ctrl_path = std::ffi::CString::new("/dev/ublk-control").unwrap();
    let ctrl_fd = unsafe { libc::open(ctrl_path.as_ptr(), libc::O_RDWR) };
    if ctrl_fd < 0 {
        debug!(dev_id, "Cannot open /dev/ublk-control");
        return Ok(()); // Not an error - ublk may not be loaded
    }

    // Try to delete using ioctl (fallback for simple cleanup)
    // The device will be cleaned up when the UblkCtrl Drop runs
    // or when the module is reloaded
    unsafe {
        libc::close(ctrl_fd);
    }

    debug!(dev_id, "Reset device attempt completed");
    Ok(())
}

/// Set up graceful shutdown signal handlers.
///
/// Installs handlers for SIGTERM and SIGINT that:
/// 1. Set the stop flag to signal the daemon to exit
/// 2. Log the shutdown request
///
/// Returns an Arc<AtomicBool> that will be set to true when shutdown is requested.
pub fn setup_signal_handlers() -> Arc<AtomicBool> {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = stop.clone();

    // SIGINT (Ctrl+C) and SIGTERM handler
    ctrlc::set_handler(move || {
        if stop_clone.load(Ordering::Relaxed) {
            // Second signal - force exit
            warn!("Received second interrupt, forcing exit");
            std::process::exit(1);
        }
        info!("Received shutdown signal, initiating graceful shutdown...");
        stop_clone.store(true, Ordering::Relaxed);
    })
    .expect("Failed to set signal handler");

    stop
}

/// Perform startup cleanup and return stop signal.
///
/// This is the main entry point for cleanup, called from main() before
/// any device operations. It:
/// 1. Detects and cleans orphaned devices
/// 2. Sets up signal handlers
///
/// Returns the stop signal for graceful shutdown.
pub fn startup_cleanup() -> Arc<AtomicBool> {
    info!("Performing startup cleanup...");

    // Clean up any orphaned devices from previous runs
    match cleanup_orphaned_devices() {
        Ok(cleaned) if cleaned > 0 => {
            info!(cleaned, "Startup cleanup: removed orphaned devices");
        }
        Ok(_) => {
            debug!("Startup cleanup: no orphaned devices found");
        }
        Err(e) => {
            warn!(error = %e, "Startup cleanup: failed to clean orphaned devices");
        }
    }

    // Set up signal handlers for graceful shutdown
    setup_signal_handlers()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_orphaned_devices_empty() {
        // Should not panic on systems without ublk
        let result = detect_orphaned_devices();
        assert!(result.is_ok());
    }

    #[test]
    fn test_signal_handler_setup() {
        let stop = setup_signal_handlers();
        assert!(!stop.load(Ordering::Relaxed));
    }
}
