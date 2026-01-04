//! Create zram device command.

use clap::Args;
use std::fs;
use std::path::Path;

/// Arguments for creating a zram device.
#[derive(Args)]
pub struct CreateArgs {
    /// Device number (0-15).
    #[arg(short, long, default_value = "0")]
    pub device: u32,

    /// Device size (e.g., "4G", "512M", "ram/2").
    #[arg(short, long)]
    pub size: String,

    /// Compression algorithm (lz4, zstd).
    #[arg(short, long, default_value = "lz4")]
    pub algorithm: String,

    /// Number of compression streams (0 = auto).
    #[arg(long, default_value = "0")]
    pub streams: u32,
}

/// Create and configure a zram device.
pub fn create(args: CreateArgs) -> Result<(), Box<dyn std::error::Error>> {
    let device_path = format!("/dev/zram{}", args.device);
    let sys_path = format!("/sys/block/zram{}", args.device);

    // Check if device exists, create if needed
    if !Path::new(&device_path).exists() {
        // Load zram module and create device
        let control_path = "/sys/class/zram-control/hot_add";
        if Path::new(control_path).exists() {
            fs::write(control_path, "")?;
        } else {
            return Err("zram module not loaded and cannot create device".into());
        }
    }

    // Parse size
    let size_bytes = parse_size(&args.size)?;

    // Configure algorithm
    let algo_path = format!("{sys_path}/comp_algorithm");
    if Path::new(&algo_path).exists() {
        fs::write(&algo_path, &args.algorithm)?;
    }

    // Configure streams
    if args.streams > 0 {
        let streams_path = format!("{sys_path}/max_comp_streams");
        if Path::new(&streams_path).exists() {
            fs::write(&streams_path, args.streams.to_string())?;
        }
    }

    // Set size (this activates the device)
    let disksize_path = format!("{sys_path}/disksize");
    fs::write(&disksize_path, size_bytes.to_string())?;

    println!(
        "Created zram{} with size {} using {}",
        args.device,
        format_size(size_bytes),
        args.algorithm
    );

    Ok(())
}

fn parse_size(size: &str) -> Result<u64, Box<dyn std::error::Error>> {
    let size = size.trim().to_uppercase();

    // Handle "ram/N" format
    if size.starts_with("RAM/") {
        let divisor: u64 = size[4..].parse()?;
        let total_ram = get_total_ram()?;
        return Ok(total_ram / divisor);
    }

    // Handle suffix format (K, M, G, T)
    let (num_str, multiplier) = if size.ends_with('K') {
        (&size[..size.len() - 1], 1024u64)
    } else if size.ends_with('M') {
        (&size[..size.len() - 1], 1024u64 * 1024)
    } else if size.ends_with('G') {
        (&size[..size.len() - 1], 1024u64 * 1024 * 1024)
    } else if size.ends_with('T') {
        (&size[..size.len() - 1], 1024u64 * 1024 * 1024 * 1024)
    } else {
        (size.as_str(), 1u64)
    };

    let num: u64 = num_str.parse()?;
    Ok(num * multiplier)
}

fn get_total_ram() -> Result<u64, Box<dyn std::error::Error>> {
    let meminfo = fs::read_to_string("/proc/meminfo")?;
    for line in meminfo.lines() {
        if line.starts_with("MemTotal:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let kb: u64 = parts[1].parse()?;
                return Ok(kb * 1024);
            }
        }
    }
    Err("Could not determine total RAM".into())
}

fn format_size(bytes: u64) -> String {
    const GB: u64 = 1024 * 1024 * 1024;
    const MB: u64 = 1024 * 1024;
    const KB: u64 = 1024;

    if bytes >= GB {
        format!("{:.1}G", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1}M", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1}K", bytes as f64 / KB as f64)
    } else {
        format!("{bytes}B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_size_bytes() {
        assert_eq!(parse_size("1024").unwrap(), 1024);
    }

    #[test]
    fn test_parse_size_kb() {
        assert_eq!(parse_size("4K").unwrap(), 4 * 1024);
    }

    #[test]
    fn test_parse_size_mb() {
        assert_eq!(parse_size("512M").unwrap(), 512 * 1024 * 1024);
    }

    #[test]
    fn test_parse_size_gb() {
        assert_eq!(parse_size("4G").unwrap(), 4 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(4 * 1024 * 1024 * 1024), "4.0G");
        assert_eq!(format_size(512 * 1024 * 1024), "512.0M");
    }
}
