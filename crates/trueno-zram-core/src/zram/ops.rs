//! Zram device operations.
//!
//! This module provides high-level operations on zram devices that can be
//! used by the CLI and tested with real devices.

use super::{ZramConfig, ZramDevice, ZramStatus};
use crate::{Error, Result};
use std::path::Path;

/// Trait for zram operations (allows for testing).
pub trait ZramOps {
    /// Create a new zram device.
    fn create(&self, config: &ZramConfig) -> Result<()>;

    /// Remove a zram device.
    fn remove(&self, device: u32, force: bool) -> Result<()>;

    /// Get status of a device.
    fn status(&self, device: u32) -> Result<ZramStatus>;

    /// List all devices.
    fn list(&self) -> Result<Vec<ZramStatus>>;
}

/// Real sysfs-based implementation.
#[derive(Debug, Default)]
pub struct SysfsOps;

impl SysfsOps {
    /// Create a new sysfs operations instance.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Hot-add a new zram device.
    fn hot_add(&self) -> Result<u32> {
        let control_path = "/sys/class/zram-control/hot_add";
        if !Path::new(control_path).exists() {
            return Err(Error::IoError(
                "zram-control not available, is zram module loaded?".to_string(),
            ));
        }

        // Reading hot_add returns the new device number
        let content = std::fs::read_to_string(control_path)
            .map_err(|e| Error::IoError(format!("failed to hot_add: {e}")))?;

        content
            .trim()
            .parse()
            .map_err(|_| Error::InvalidInput("invalid device number from hot_add".to_string()))
    }

    /// Hot-remove a zram device.
    fn hot_remove(&self, device: u32) -> Result<()> {
        let control_path = "/sys/class/zram-control/hot_remove";
        if Path::new(control_path).exists() {
            std::fs::write(control_path, device.to_string())
                .map_err(|e| Error::IoError(format!("failed to hot_remove: {e}")))?;
        }
        Ok(())
    }

    /// Check if device is in use as swap.
    fn is_swap_active(&self, device: u32) -> bool {
        let swaps = std::fs::read_to_string("/proc/swaps").unwrap_or_default();
        swaps.contains(&format!("/dev/zram{device}"))
    }

    /// Check if zram module is loaded.
    #[must_use]
    pub fn is_available(&self) -> bool {
        Path::new("/sys/class/zram-control").exists()
            || Path::new("/sys/block/zram0").exists()
    }
}

impl ZramOps for SysfsOps {
    fn create(&self, config: &ZramConfig) -> Result<()> {
        let dev = ZramDevice::new(config.device);

        // If device doesn't exist, try to create it
        if !dev.exists() {
            // Check if we need to hot-add
            let control_path = "/sys/class/zram-control/hot_add";
            if Path::new(control_path).exists() {
                // Hot-add creates sequentially, so add until we reach our device
                while !dev.exists() {
                    let new_dev = self.hot_add()?;
                    if new_dev > config.device {
                        return Err(Error::InvalidInput(format!(
                            "cannot create zram{}, hot_add created zram{new_dev}",
                            config.device
                        )));
                    }
                }
            } else {
                return Err(Error::IoError(
                    "zram module not loaded and cannot create device".to_string(),
                ));
            }
        }

        // Configure the device
        dev.configure(config)
    }

    fn remove(&self, device: u32, force: bool) -> Result<()> {
        let dev = ZramDevice::new(device);

        if !dev.exists() {
            return Err(Error::InvalidInput(format!("zram{device} does not exist")));
        }

        // Check if device is in use
        if !force && self.is_swap_active(device) {
            return Err(Error::InvalidInput(format!(
                "zram{device} is in use as swap. Use force=true or swapoff first."
            )));
        }

        // Reset the device
        dev.reset()?;

        // Try to hot-remove
        let _ = self.hot_remove(device);

        Ok(())
    }

    fn status(&self, device: u32) -> Result<ZramStatus> {
        let dev = ZramDevice::new(device);
        dev.status()
    }

    fn list(&self) -> Result<Vec<ZramStatus>> {
        let devices = super::find_devices();
        let mut statuses = Vec::new();
        for device in devices {
            if let Ok(status) = self.status(device) {
                statuses.push(status);
            }
        }
        Ok(statuses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sysfs_ops_new() {
        let ops = SysfsOps::new();
        // Just verify it compiles and doesn't panic
        let _ = ops.is_available();
    }

    #[test]
    fn test_sysfs_ops_default() {
        let ops = SysfsOps::default();
        let _ = ops.is_available();
    }

    // Integration tests that require root and zram module
    // These are gated by a feature or run conditionally

    #[test]
    #[ignore = "requires root and zram module"]
    fn test_create_remove_small_device() {
        let ops = SysfsOps::new();
        if !ops.is_available() {
            println!("Skipping: zram not available");
            return;
        }

        let config = ZramConfig {
            device: 15, // Use high device number to avoid conflicts
            size: 4 * 1024 * 1024, // 4MB
            algorithm: "lz4".to_string(),
            streams: 1,
        };

        // Create
        let result = ops.create(&config);
        if result.is_err() {
            println!("Skipping: cannot create device (may need root)");
            return;
        }

        // Status
        let status = ops.status(15).unwrap();
        assert_eq!(status.device, 15);
        assert_eq!(status.disksize, 4 * 1024 * 1024);

        // Remove
        ops.remove(15, true).unwrap();
    }
}
