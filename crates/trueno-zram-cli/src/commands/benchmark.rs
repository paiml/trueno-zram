//! Benchmark command for compression performance testing.

use clap::Args;
use std::time::Instant;
use trueno_zram_core::{Algorithm, CompressorBuilder, PAGE_SIZE};

/// Arguments for benchmark command.
#[derive(Args)]
pub struct BenchmarkArgs {
    /// Number of pages to compress.
    #[arg(short, long, default_value = "10000")]
    pub pages: usize,

    /// Algorithm to benchmark (lz4, zstd, all).
    #[arg(short, long, default_value = "all")]
    pub algorithm: String,

    /// Data pattern (zero, random, text, mixed).
    #[arg(short = 'p', long, default_value = "mixed")]
    pub pattern: String,
}

/// Run compression benchmarks.
pub fn benchmark(args: BenchmarkArgs) -> Result<(), Box<dyn std::error::Error>> {
    println!("trueno-zram Compression Benchmark");
    println!("==================================");
    println!("Pages: {}", args.pages);
    println!("Pattern: {}", args.pattern);
    println!();

    // Generate test data
    let pages = generate_test_pages(args.pages, &args.pattern);
    let total_bytes = pages.len() * PAGE_SIZE;

    let algorithms: Vec<Algorithm> = if args.algorithm == "all" {
        vec![
            Algorithm::Lz4,
            Algorithm::Zstd { level: 1 },
            Algorithm::Zstd { level: 3 },
        ]
    } else if args.algorithm == "lz4" {
        vec![Algorithm::Lz4]
    } else if args.algorithm.starts_with("zstd") {
        let level = args
            .algorithm
            .strip_prefix("zstd")
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);
        vec![Algorithm::Zstd { level }]
    } else {
        return Err(format!("Unknown algorithm: {}", args.algorithm).into());
    };

    println!(
        "{:<15} {:>10} {:>12} {:>12} {:>10}",
        "Algorithm", "Backend", "Compress", "Decompress", "Ratio"
    );
    println!("{}", "-".repeat(60));

    for algo in algorithms {
        let compressor = CompressorBuilder::new().algorithm(algo).build()?;

        // Compression benchmark
        let start = Instant::now();
        let mut compressed_pages = Vec::with_capacity(pages.len());
        let mut total_compressed_size = 0usize;

        for page in &pages {
            let compressed = compressor.compress(page)?;
            total_compressed_size += compressed.data.len();
            compressed_pages.push(compressed);
        }
        let compress_time = start.elapsed();
        let compress_throughput = total_bytes as f64 / compress_time.as_secs_f64() / 1e9;

        // Decompression benchmark
        let start = Instant::now();
        for compressed in &compressed_pages {
            let _page = compressor.decompress(compressed)?;
        }
        let decompress_time = start.elapsed();
        let decompress_throughput = total_bytes as f64 / decompress_time.as_secs_f64() / 1e9;

        let ratio = total_bytes as f64 / total_compressed_size as f64;

        println!(
            "{:<15} {:>10} {:>10.2} GB/s {:>10.2} GB/s {:>9.2}x",
            format!("{algo:?}"),
            format!("{:?}", compressor.backend()),
            compress_throughput,
            decompress_throughput,
            ratio
        );
    }

    Ok(())
}

fn generate_test_pages(count: usize, pattern: &str) -> Vec<[u8; PAGE_SIZE]> {
    let mut pages = Vec::with_capacity(count);
    let mut rng_state = 12345u64;

    for i in 0..count {
        let mut page = [0u8; PAGE_SIZE];

        match pattern {
            "zero" => {
                // Already zeros
            }
            "random" => {
                for byte in &mut page {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    *byte = (rng_state >> 33) as u8;
                }
            }
            "text" => {
                // Simulate text-like data
                let text = b"The quick brown fox jumps over the lazy dog. ";
                for (j, byte) in page.iter_mut().enumerate() {
                    *byte = text[j % text.len()];
                }
            }
            "mixed" | _ => {
                // Mix of patterns
                match i % 4 {
                    0 => {} // zeros
                    1 => {
                        // random
                        for byte in &mut page {
                            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                            *byte = (rng_state >> 33) as u8;
                        }
                    }
                    2 => {
                        // repeating pattern
                        for (j, byte) in page.iter_mut().enumerate() {
                            *byte = (j % 16) as u8;
                        }
                    }
                    3 => {
                        // text-like
                        let text = b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. ";
                        for (j, byte) in page.iter_mut().enumerate() {
                            *byte = text[j % text.len()];
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }

        pages.push(page);
    }

    pages
}
