//! CLI module for trueno-ublk
//!
//! Provides zramctl-compatible command-line interface with extensions
//! for GPU acceleration and entropy analysis.

pub mod compact;
pub mod create;
pub mod entropy;
pub mod find;
pub mod idle;
pub mod list;
pub mod reset;
pub mod set;
pub mod stat;
pub mod top;
pub mod writeback;

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// trueno-ublk - GPU-accelerated ZRAM replacement
#[derive(Parser)]
#[command(name = "trueno-ublk")]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Create a new ublk device
    Create(CreateArgs),

    /// List all trueno-ublk devices
    List(ListArgs),

    /// Show device statistics
    Stat(StatArgs),

    /// Reset/remove a device
    Reset(ResetArgs),

    /// Find a free device
    Find,

    /// Trigger compaction
    Compact(DeviceArg),

    /// Mark pages as idle (for writeback)
    Idle(DeviceArg),

    /// Trigger writeback to backing device
    Writeback(WritebackArgs),

    /// Live reconfiguration
    Set(SetArgs),

    /// Interactive TUI dashboard
    Top(TopArgs),

    /// Analyze file/directory entropy
    Entropy(EntropyArgs),
}

/// Compression algorithm selection
#[derive(Clone, Copy, Debug, ValueEnum, Default)]
pub enum Algorithm {
    /// LZ4 - fast compression (default)
    #[default]
    Lz4,
    /// LZ4 HC - higher compression ratio
    Lz4hc,
    /// Zstd level 1 - balanced
    Zstd1,
    /// Zstd level 3 - better ratio
    Zstd3,
    /// Zstd level 9 - maximum compression
    Zstd9,
}

impl Algorithm {
    pub fn to_trueno(self) -> trueno_zram_core::Algorithm {
        match self {
            Algorithm::Lz4 => trueno_zram_core::Algorithm::Lz4,
            Algorithm::Lz4hc => trueno_zram_core::Algorithm::Lz4Hc,
            Algorithm::Zstd1 => trueno_zram_core::Algorithm::Zstd { level: 1 },
            Algorithm::Zstd3 => trueno_zram_core::Algorithm::Zstd { level: 3 },
            Algorithm::Zstd9 => trueno_zram_core::Algorithm::Zstd { level: 9 },
        }
    }
}

/// Create command arguments
#[derive(Parser)]
pub struct CreateArgs {
    /// Device size (e.g., 1T, 256G, 1024M)
    #[arg(short, long)]
    pub size: String,

    /// Compression algorithm
    #[arg(short, long, value_enum, default_value = "lz4")]
    pub algorithm: Algorithm,

    /// Number of compression streams/threads
    #[arg(short = 't', long, default_value = "0")]
    pub streams: usize,

    /// Enable GPU acceleration
    #[arg(long)]
    pub gpu: bool,

    /// Memory limit for compressed data
    #[arg(long)]
    pub mem_limit: Option<String>,

    /// Backing device for writeback
    #[arg(long)]
    pub backing_dev: Option<PathBuf>,

    /// Writeback limit
    #[arg(long)]
    pub writeback_limit: Option<String>,

    /// Entropy threshold for skipping compression (0.0-8.0)
    #[arg(long, default_value = "7.5")]
    pub entropy_skip: f64,

    /// GPU batch size threshold
    #[arg(long, default_value = "1000")]
    pub gpu_batch: usize,
}

/// List command arguments
#[derive(Parser)]
pub struct ListArgs {
    /// Output columns (comma-separated)
    #[arg(short, long)]
    pub output: Option<String>,

    /// Output all columns
    #[arg(long)]
    pub output_all: bool,

    /// Print sizes in bytes
    #[arg(short, long)]
    pub bytes: bool,

    /// Don't print headings
    #[arg(short = 'n', long)]
    pub no_headers: bool,

    /// Raw output format
    #[arg(long)]
    pub raw: bool,

    /// JSON output
    #[arg(long)]
    pub json: bool,
}

/// Stat command arguments
#[derive(Parser)]
pub struct StatArgs {
    /// Device path (e.g., /dev/ublkb0)
    pub device: Option<PathBuf>,

    /// Output mm_stat format (zram compatible)
    #[arg(long)]
    pub mm_stat: bool,

    /// Output io_stat format
    #[arg(long)]
    pub io_stat: bool,

    /// Output bd_stat format (backing device)
    #[arg(long)]
    pub bd_stat: bool,

    /// JSON output
    #[arg(long)]
    pub json: bool,

    /// Show entropy distribution
    #[arg(long)]
    pub entropy: bool,

    /// Show debug stats
    #[arg(long)]
    pub debug: bool,
}

/// Reset command arguments
#[derive(Parser)]
pub struct ResetArgs {
    /// Device path(s) to reset
    pub devices: Vec<PathBuf>,

    /// Reset all devices
    #[arg(long)]
    pub all: bool,
}

/// Simple device argument
#[derive(Parser)]
pub struct DeviceArg {
    /// Device path
    pub device: PathBuf,
}

/// Writeback command arguments
#[derive(Parser)]
pub struct WritebackArgs {
    /// Device path
    pub device: PathBuf,

    /// Only writeback idle pages
    #[arg(long)]
    pub idle: bool,

    /// Only writeback huge pages
    #[arg(long)]
    pub huge: bool,

    /// Writeback all pages
    #[arg(long)]
    pub all: bool,
}

/// Set command arguments
#[derive(Parser)]
pub struct SetArgs {
    /// Device path
    pub device: PathBuf,

    /// Set memory limit
    #[arg(long)]
    pub mem_limit: Option<String>,

    /// Enable/disable GPU
    #[arg(long)]
    pub gpu: Option<bool>,

    /// Set entropy skip threshold
    #[arg(long)]
    pub entropy_skip: Option<f64>,

    /// Set writeback limit
    #[arg(long)]
    pub writeback_limit: Option<String>,

    /// Enable/disable writeback limit
    #[arg(long)]
    pub writeback_limit_enable: Option<bool>,

    /// Reset mem_used_max watermark
    #[arg(long)]
    pub reset_mem_used_max: bool,
}

/// Top (TUI) command arguments
#[derive(Parser)]
pub struct TopArgs {
    /// Device to monitor (optional, monitors all if not specified)
    pub device: Option<PathBuf>,

    /// Demo mode with simulated data
    #[arg(long)]
    pub demo: bool,

    /// Non-interactive report mode
    #[arg(long)]
    pub report: bool,
}

/// Entropy analysis arguments
#[derive(Parser)]
pub struct EntropyArgs {
    /// Paths to analyze
    pub paths: Vec<PathBuf>,

    /// JSON output
    #[arg(long)]
    pub json: bool,

    /// Recursive directory scan
    #[arg(short, long)]
    pub recursive: bool,
}

/// Output columns available for list command
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum OutputColumn {
    Name,
    Disksize,
    Data,
    Compr,
    Algorithm,
    Streams,
    ZeroPages,
    Total,
    MemLimit,
    MemUsed,
    Migrated,
    Mountpoint,
    Gpu,
    Throughput,
    Backend,
    Entropy,
}

impl OutputColumn {
    pub fn all() -> &'static [OutputColumn] {
        &[
            OutputColumn::Name,
            OutputColumn::Disksize,
            OutputColumn::Data,
            OutputColumn::Compr,
            OutputColumn::Algorithm,
            OutputColumn::Streams,
            OutputColumn::ZeroPages,
            OutputColumn::Total,
            OutputColumn::MemLimit,
            OutputColumn::MemUsed,
            OutputColumn::Migrated,
            OutputColumn::Mountpoint,
            OutputColumn::Gpu,
            OutputColumn::Throughput,
            OutputColumn::Backend,
            OutputColumn::Entropy,
        ]
    }

    pub fn default_columns() -> &'static [OutputColumn] {
        &[
            OutputColumn::Name,
            OutputColumn::Disksize,
            OutputColumn::Data,
            OutputColumn::Compr,
            OutputColumn::Algorithm,
            OutputColumn::Streams,
        ]
    }

    pub fn header(&self) -> &'static str {
        match self {
            OutputColumn::Name => "NAME",
            OutputColumn::Disksize => "DISKSIZE",
            OutputColumn::Data => "DATA",
            OutputColumn::Compr => "COMPR",
            OutputColumn::Algorithm => "ALGORITHM",
            OutputColumn::Streams => "STREAMS",
            OutputColumn::ZeroPages => "ZERO-PAGES",
            OutputColumn::Total => "TOTAL",
            OutputColumn::MemLimit => "MEM-LIMIT",
            OutputColumn::MemUsed => "MEM-USED",
            OutputColumn::Migrated => "MIGRATED",
            OutputColumn::Mountpoint => "MOUNTPOINT",
            OutputColumn::Gpu => "GPU",
            OutputColumn::Throughput => "THROUGHPUT",
            OutputColumn::Backend => "BACKEND",
            OutputColumn::Entropy => "ENTROPY",
        }
    }
}

/// Parse size string like "1T", "256G", "1024M" into bytes
pub fn parse_size(s: &str) -> anyhow::Result<u64> {
    let s = s.trim().to_uppercase();
    let (num, multiplier) = if s.ends_with("T") || s.ends_with("TIB") {
        (s.trim_end_matches("TIB").trim_end_matches("T"), 1u64 << 40)
    } else if s.ends_with("G") || s.ends_with("GIB") {
        (s.trim_end_matches("GIB").trim_end_matches("G"), 1u64 << 30)
    } else if s.ends_with("M") || s.ends_with("MIB") {
        (s.trim_end_matches("MIB").trim_end_matches("M"), 1u64 << 20)
    } else if s.ends_with("K") || s.ends_with("KIB") {
        (s.trim_end_matches("KIB").trim_end_matches("K"), 1u64 << 10)
    } else {
        (s.as_str(), 1u64)
    };

    let num: u64 = num.trim().parse()?;
    Ok(num * multiplier)
}

/// Format bytes as human-readable string
pub fn format_size(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = KIB * 1024;
    const GIB: u64 = MIB * 1024;
    const TIB: u64 = GIB * 1024;

    if bytes >= TIB {
        format!("{:.1}T", bytes as f64 / TIB as f64)
    } else if bytes >= GIB {
        format!("{:.1}G", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1}M", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1}K", bytes as f64 / KIB as f64)
    } else {
        format!("{}B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_size_terabytes() {
        assert_eq!(parse_size("1T").unwrap(), 1u64 << 40);
        assert_eq!(parse_size("2TiB").unwrap(), 2u64 << 40);
    }

    #[test]
    fn test_parse_size_gigabytes() {
        assert_eq!(parse_size("256G").unwrap(), 256u64 << 30);
        assert_eq!(parse_size("128GiB").unwrap(), 128u64 << 30);
    }

    #[test]
    fn test_parse_size_megabytes() {
        assert_eq!(parse_size("1024M").unwrap(), 1024u64 << 20);
    }

    #[test]
    fn test_parse_size_bytes() {
        assert_eq!(parse_size("4096").unwrap(), 4096);
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(1u64 << 40), "1.0T");
        assert_eq!(format_size(256u64 << 30), "256.0G");
        assert_eq!(format_size(1024u64 << 20), "1.0G");
        assert_eq!(format_size(512u64 << 20), "512.0M");
    }

    #[test]
    fn test_output_columns() {
        assert!(OutputColumn::all().len() >= 12);
        assert!(OutputColumn::default_columns().len() >= 6);
    }

    #[test]
    fn test_algorithm_conversion() {
        let lz4 = Algorithm::Lz4;
        assert!(matches!(lz4.to_trueno(), trueno_zram_core::Algorithm::Lz4));
    }
}
