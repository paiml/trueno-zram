//! Remove zram device command.

use clap::Args;
use std::fs;
use std::path::Path;

/// Arguments for removing a zram device.
#[derive(Args)]
pub struct RemoveArgs {
    /// Device number to remove.
    #[arg(short, long, default_value = "0")]
    pub device: u32,

    /// Force removal even if in use.
    #[arg(short, long)]
    pub force: bool,
}

/// Remove a zram device.
pub fn remove(args: RemoveArgs) -> Result<(), Box<dyn std::error::Error>> {
    let sys_path = format!("/sys/block/zram{}", args.device);

    if !Path::new(&sys_path).exists() {
        return Err(format!("zram{} does not exist", args.device).into());
    }

    // Check if device is in use (has swap or mount)
    if !args.force {
        let swaps = fs::read_to_string("/proc/swaps").unwrap_or_default();
        if swaps.contains(&format!("/dev/zram{}", args.device)) {
            return Err(format!(
                "zram{} is in use as swap. Use --force or swapoff first.",
                args.device
            )
            .into());
        }
    }

    // Reset the device (disables it)
    let reset_path = format!("{sys_path}/reset");
    fs::write(&reset_path, "1")?;

    // Optionally hot-remove the device
    let hot_remove_path = "/sys/class/zram-control/hot_remove";
    if Path::new(hot_remove_path).exists() {
        let _ = fs::write(hot_remove_path, args.device.to_string());
    }

    println!("Removed zram{}", args.device);
    Ok(())
}
