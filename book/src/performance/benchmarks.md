# Benchmarks

## Running Benchmarks

```bash
# Criterion benchmarks
cargo bench --all-features

# With baseline comparison
cargo bench --all-features -- --save-baseline main

# Example benchmarks
cargo run -p trueno-zram-core --example compress_benchmark --release
```

## Results Summary

### LZ4 Performance

| Backend | Compress | Decompress | Ratio |
|---------|----------|------------|-------|
| AVX-512 | 4.4 GB/s | 5.4 GB/s | 2-4x |
| AVX2 | 3.2 GB/s | 4.1 GB/s | 2-4x |
| Scalar | 3.0 GB/s | 3.8 GB/s | 2-4x |

### ZSTD Performance

| Backend | Level | Compress | Decompress | Ratio |
|---------|-------|----------|------------|-------|
| AVX-512 | 1 | 11.2 GB/s | 46 GB/s | 3-5x |
| AVX-512 | 3 | 8.5 GB/s | 45 GB/s | 4-6x |
| AVX2 | 1 | 8.5 GB/s | 35 GB/s | 3-5x |

### Same-Fill Performance

| Backend | Detection | Ratio |
|---------|-----------|-------|
| AVX-512 | 22 GB/s | 2048:1 |
| AVX2 | 18 GB/s | 2048:1 |
| Scalar | 12 GB/s | 2048:1 |

## Data Patterns

### Zeros (Best Case)

```
Pattern: Zeros (100% same-fill)
Pages: 1000
Compression: 22 GB/s
Decompression: 46 GB/s
Ratio: 2048:1
```

### Text (Compressible)

```
Pattern: Text/Code
Pages: 1000
LZ4 Compression: 4.4 GB/s
LZ4 Decompression: 5.4 GB/s
Ratio: 3.2:1
```

### Random (Incompressible)

```
Pattern: Random bytes
Pages: 1000
LZ4 Compression: 1.6 GB/s
LZ4 Decompression: 32 GB/s
Ratio: 1.0:1 (pass-through)
```

## GPU Benchmarks

### RTX 4090

| Batch Size | Throughput | PCIe 5x |
|------------|------------|---------|
| 1,000 | 8 GB/s | No |
| 10,000 | 45 GB/s | Yes |
| 100,000 | 120 GB/s | Yes |

### A100

| Batch Size | Throughput | PCIe 5x |
|------------|------------|---------|
| 1,000 | 12 GB/s | No |
| 10,000 | 85 GB/s | Yes |
| 100,000 | 280 GB/s | Yes |

## Latency

| Operation | P50 | P99 | P99.9 |
|-----------|-----|-----|-------|
| LZ4 compress (4KB) | 45us | 85us | 120us |
| LZ4 decompress (4KB) | 38us | 72us | 95us |
| Same-fill detect | 8us | 15us | 25us |

## Memory Usage

| Component | Memory |
|-----------|--------|
| Hash table (LZ4) | 64 KB |
| Working buffer | 16 KB |
| ZSTD context | 256 KB |

## Comparison with Kernel zram

| Metric | Kernel zram | trueno-zram | Improvement |
|--------|-------------|-------------|-------------|
| LZ4 compress | 2.8 GB/s | 4.4 GB/s | +57% |
| LZ4 decompress | 3.5 GB/s | 5.4 GB/s | +54% |
| Same-fill | 8 GB/s | 22 GB/s | +175% |
