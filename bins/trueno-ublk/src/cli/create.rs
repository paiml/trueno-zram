//! Create command - creates a new trueno-ublk device

use super::{parse_size, CreateArgs};
use crate::device::UblkDevice;
use anyhow::Result;

pub fn run(args: CreateArgs) -> Result<()> {
    let size = parse_size(&args.size)?;
    let mem_limit = args.mem_limit.as_ref().map(|s| parse_size(s)).transpose()?;
    let writeback_limit = args
        .writeback_limit
        .as_ref()
        .map(|s| parse_size(s))
        .transpose()?;

    let streams = if args.streams == 0 {
        num_cpus()
    } else {
        args.streams
    };

    tracing::info!(
        size = %super::format_size(size),
        algorithm = ?args.algorithm,
        streams = streams,
        gpu = args.gpu,
        "Creating trueno-ublk device"
    );

    let config = crate::device::DeviceConfig {
        dev_id: args.dev_id,
        size,
        algorithm: args.algorithm.to_trueno(),
        streams,
        gpu_enabled: args.gpu,
        mem_limit,
        backing_dev: args.backing_dev,
        writeback_limit,
        entropy_skip_threshold: args.entropy_skip,
        gpu_batch_size: args.gpu_batch,
        foreground: args.foreground,
    };

    let device = UblkDevice::create(config)?;
    println!("{}", device.path().display());

    Ok(())
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
}
