//! Status command for zram devices.

use crate::output::OutputFormat;
use clap::Args;
use serde::Serialize;
use std::fs;
use std::path::Path;

/// Arguments for status command.
#[derive(Args)]
pub struct StatusArgs {
    /// Specific device to show (omit for all devices).
    #[arg(short, long)]
    pub device: Option<u32>,
}

/// Status information for a zram device.
#[derive(Debug, Serialize)]
pub struct ZramStatus {
    /// Device number.
    pub device: u32,
    /// Configured disk size.
    pub disksize: u64,
    /// Original data size (uncompressed).
    pub orig_data_size: u64,
    /// Compressed data size.
    pub compr_data_size: u64,
    /// Memory used.
    pub mem_used_total: u64,
    /// Compression algorithm.
    pub algorithm: String,
    /// Compression ratio.
    pub ratio: f64,
}

/// Show zram device status.
pub fn status(args: StatusArgs, format: OutputFormat) -> Result<(), Box<dyn std::error::Error>> {
    let devices = if let Some(dev) = args.device {
        vec![dev]
    } else {
        find_zram_devices()?
    };

    let mut statuses = Vec::new();
    for dev in devices {
        if let Ok(status) = get_device_status(dev) {
            statuses.push(status);
        }
    }

    match format {
        OutputFormat::Table => print_table(&statuses),
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&statuses)?);
        }
        OutputFormat::Raw => {
            for s in &statuses {
                println!(
                    "{} {} {} {} {} {}",
                    s.device,
                    s.disksize,
                    s.orig_data_size,
                    s.compr_data_size,
                    s.mem_used_total,
                    s.algorithm
                );
            }
        }
    }

    Ok(())
}

fn find_zram_devices() -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let mut devices = Vec::new();
    for i in 0..16 {
        let path = format!("/sys/block/zram{i}");
        if Path::new(&path).exists() {
            devices.push(i);
        }
    }
    Ok(devices)
}

fn get_device_status(device: u32) -> Result<ZramStatus, Box<dyn std::error::Error>> {
    let sys_path = format!("/sys/block/zram{device}");

    let disksize = read_sysfs_u64(&format!("{sys_path}/disksize"))?;
    let orig_data_size = read_sysfs_u64(&format!("{sys_path}/orig_data_size")).unwrap_or(0);
    let compr_data_size = read_sysfs_u64(&format!("{sys_path}/compr_data_size")).unwrap_or(0);
    let mem_used_total = read_sysfs_u64(&format!("{sys_path}/mem_used_total")).unwrap_or(0);
    let algorithm = fs::read_to_string(format!("{sys_path}/comp_algorithm"))
        .unwrap_or_else(|_| "unknown".to_string())
        .trim()
        .to_string();

    let ratio = if compr_data_size > 0 {
        orig_data_size as f64 / compr_data_size as f64
    } else {
        0.0
    };

    Ok(ZramStatus {
        device,
        disksize,
        orig_data_size,
        compr_data_size,
        mem_used_total,
        algorithm,
        ratio,
    })
}

fn read_sysfs_u64(path: &str) -> Result<u64, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    Ok(content.trim().parse()?)
}

fn print_table(statuses: &[ZramStatus]) {
    println!(
        "{:>6} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "DEVICE", "DISKSIZE", "DATA", "COMPR", "TOTAL", "ALGO", "RATIO"
    );

    for s in statuses {
        println!(
            "zram{:<2} {:>10} {:>10} {:>10} {:>10} {:>8} {:>7.2}x",
            s.device,
            format_size(s.disksize),
            format_size(s.orig_data_size),
            format_size(s.compr_data_size),
            format_size(s.mem_used_total),
            s.algorithm,
            s.ratio
        );
    }
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
    } else if bytes > 0 {
        format!("{bytes}B")
    } else {
        "0".to_string()
    }
}
