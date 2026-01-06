#!/bin/bash
# shellcheck disable=SC2046,SC2031,SC2032,SC2005,SC2006,SC2002
# SC2046: quotes in assignments
# SC2031: false positive on local vars
# SC2032: false positive about sourcing
# SC2005: echo in command substitution (needed for bc)
# SC2006: backticks in heredocs
# SC2002: cat with head (style preference)
# scientific-swap-benchmark.sh - Reproducible swap technology comparison
#
# BENCH-001: Scientific falsification benchmark for swap technologies
# Compares: Regular Swap vs Kernel ZRAM vs trueno-zram
#
# CROSS-PLATFORM: Supports Linux (full swap tests) and macOS (compression only)
# GPU BACKENDS: CUDA (Linux NVIDIA) and WGPU (macOS Metal)
#
# Usage:
#   sudo ./scripts/scientific-swap-benchmark.sh [--quick|--full]
#   sudo ./scripts/scientific-swap-benchmark.sh --remote mac  # Run on remote macOS
#
# Requirements:
#   Linux:
#     - Root access
#     - fio >= 3.35
#     - 32GB+ RAM
#     - Kernel >= 6.0 with ublk support
#   macOS:
#     - Root access (sudo)
#     - fio (brew install fio)
#     - 16GB+ RAM

set -euo pipefail

#═══════════════════════════════════════════════════════════════════════════════
# Configuration
#═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${PROJECT_ROOT}/benchmark-results/$(date +%Y%m%d-%H%M%S)"
SWAP_SIZE_GB=8
RUNS_PER_TEST=3
TEST_RUNTIME=30
QUICK_MODE=false
REMOTE_HOST=""  # Set via --remote for SSH execution

# Platform detection
OS_TYPE="$(uname -s)"
IS_LINUX=false
IS_MACOS=false
case "$OS_TYPE" in
    Linux*)  IS_LINUX=true ;;
    Darwin*) IS_MACOS=true ;;
esac

# GPU backend detection
GPU_BACKEND="none"
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    GPU_BACKEND="cuda"
elif $IS_MACOS && system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Metal"; then
    GPU_BACKEND="wgpu"  # Metal via WGPU on macOS
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

#═══════════════════════════════════════════════════════════════════════════════
# Argument Parsing
#═══════════════════════════════════════════════════════════════════════════════

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            RUNS_PER_TEST=1
            TEST_RUNTIME=10
            shift
            ;;
        --full)
            RUNS_PER_TEST=5
            TEST_RUNTIME=60
            shift
            ;;
        --remote)
            REMOTE_HOST="$2"
            shift 2
            ;;
        --remote=*)
            REMOTE_HOST="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--quick|--full] [--remote HOST]"
            exit 1
            ;;
    esac
done

#═══════════════════════════════════════════════════════════════════════════════
# Logging
#═══════════════════════════════════════════════════════════════════════════════

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
log_fail() { echo -e "${RED}[FAIL]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

# If remote host specified, execute there
if [[ -n "$REMOTE_HOST" ]]; then
    log_info "Executing benchmark on remote host: $REMOTE_HOST"
    # Copy script to remote and execute
    scp "$0" "${REMOTE_HOST}:/tmp/scientific-swap-benchmark.sh"
    # shellcheck disable=SC2029
    ssh "$REMOTE_HOST" "chmod +x /tmp/scientific-swap-benchmark.sh && sudo /tmp/scientific-swap-benchmark.sh ${QUICK_MODE:+--quick}"
    exit $?
fi

#═══════════════════════════════════════════════════════════════════════════════
# Environment Checks
#═══════════════════════════════════════════════════════════════════════════════

check_requirements() {
    log_info "Checking requirements... (platform: $OS_TYPE, GPU: $GPU_BACKEND)"

    # Root check
    if [[ $EUID -ne 0 ]]; then
        log_fail "This script must be run as root"
        exit 1
    fi

    # fio check
    if ! command -v fio &>/dev/null; then
        if $IS_LINUX; then
            log_fail "fio not found. Install with: apt install fio"
        else
            log_fail "fio not found. Install with: brew install fio"
        fi
        exit 1
    fi

    # Memory check (cross-platform)
    local mem_gb
    if $IS_LINUX; then
        mem_gb=$(free -g | awk '/^Mem:/{print $2}')
    elif $IS_MACOS; then
        mem_gb=$(( $(sysctl -n hw.memsize) / 1073741824 ))
    else
        mem_gb=0
    fi

    if [[ $mem_gb -lt 32 ]]; then
        log_warn "Less than 32GB RAM detected ($mem_gb GB). Results may be affected."
    fi

    # Linux-specific kernel module checks
    if $IS_LINUX; then
        if ! modprobe -n zram 2>/dev/null; then
            log_warn "zram module not available"
        fi

        if ! modprobe -n ublk_drv 2>/dev/null; then
            log_fail "ublk_drv module not available. Kernel >= 6.0 required."
            exit 1
        fi

        # trueno-ublk binary (Linux only)
        if [[ ! -x "${PROJECT_ROOT}/target/release/trueno-ublk" ]]; then
            log_info "Building trueno-ublk..."
            (cd "$PROJECT_ROOT" && cargo build --release -p trueno-ublk)
        fi
    fi

    # macOS: build compression benchmark binary
    if $IS_MACOS; then
        log_info "macOS detected - swap tests will use compression benchmark only"
        if [[ ! -x "${PROJECT_ROOT}/target/release/examples/compress_benchmark" ]]; then
            log_info "Building compression benchmark..."
            local features=""
            if [[ "$GPU_BACKEND" == "wgpu" ]]; then
                features="--features wgpu"
            fi
            (cd "$PROJECT_ROOT" && cargo build --release --example compress_benchmark $features)
        fi
    fi

    log_pass "All requirements satisfied (platform: $OS_TYPE)"
}

#═══════════════════════════════════════════════════════════════════════════════
# Environment Setup
#═══════════════════════════════════════════════════════════════════════════════

setup_environment() {
    log_info "Setting up isolated test environment..."

    mkdir -p "$RESULTS_DIR"

    # Record environment (cross-platform)
    local cpu_model cpu_cores ram_gb has_avx512 has_avx2

    if $IS_LINUX; then
        cpu_model=$(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
        cpu_cores=$(nproc)
        ram_gb=$(free -g | awk '/^Mem:/{print $2}')
        has_avx512=$(if grep -q avx512 /proc/cpuinfo; then echo true; else echo false; fi)
        has_avx2=$(if grep -q avx2 /proc/cpuinfo; then echo true; else echo false; fi)
    elif $IS_MACOS; then
        cpu_model=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
        cpu_cores=$(sysctl -n hw.ncpu)
        ram_gb=$(( $(sysctl -n hw.memsize) / 1073741824 ))
        # macOS ARM doesn't have AVX, x86 Macs might
        has_avx512=$(sysctl -n hw.optional.avx512f 2>/dev/null || echo false)
        has_avx2=$(sysctl -n hw.optional.avx2_0 2>/dev/null || echo false)
    fi

    cat > "${RESULTS_DIR}/environment.json" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "hostname": "$(hostname)",
    "platform": "$OS_TYPE",
    "kernel": "$(uname -r)",
    "cpu": "$cpu_model",
    "cpu_cores": $cpu_cores,
    "ram_gb": $ram_gb,
    "gpu_backend": "$GPU_BACKEND",
    "avx512": $has_avx512,
    "avx2": $has_avx2
}
EOF

    # Disable swap (Linux only - macOS swap is different)
    if $IS_LINUX; then
        swapoff -a 2>/dev/null || true
    fi

    # Drop caches (Linux only)
    sync
    if $IS_LINUX; then
        echo 3 > /proc/sys/vm/drop_caches
    elif $IS_MACOS; then
        purge 2>/dev/null || true
    fi

    # Set CPU governor to performance (Linux only)
    if $IS_LINUX; then
        for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            echo performance > "$gov" 2>/dev/null || true
        done
    fi

    log_pass "Environment configured"
}

cleanup_environment() {
    log_info "Cleaning up..."

    if $IS_LINUX; then
        # Disable all test swap
        swapoff -a 2>/dev/null || true

        # Remove swap file
        rm -f /swapfile 2>/dev/null || true

        # Unload zram
        echo 1 > /sys/block/zram0/reset 2>/dev/null || true
        rmmod zram 2>/dev/null || true

        # Stop trueno-ublk
        pkill -9 trueno-ublk 2>/dev/null || true
        sleep 1

        # Clean up ublk devices
        for dev in /dev/ublkb*; do
            if [[ -b "$dev" ]]; then
                "${PROJECT_ROOT}/target/release/trueno-ublk" delete --dev-id "${dev##/dev/ublkb}" 2>/dev/null || true
            fi
        done
    fi

    log_pass "Cleanup complete"
}

trap cleanup_environment EXIT

#═══════════════════════════════════════════════════════════════════════════════
# Swap Setup Functions
#═══════════════════════════════════════════════════════════════════════════════

setup_regular_swap() {
    log_info "Setting up regular swap file (${SWAP_SIZE_GB}GB)..."

    # Create swap file on fastest available storage
    dd if=/dev/zero of=/swapfile bs=1G count=$SWAP_SIZE_GB status=progress 2>/dev/null
    chmod 600 /swapfile
    mkswap /swapfile >/dev/null
    swapon /swapfile -p 100

    echo "/swapfile"
}

setup_kernel_zram() {
    log_info "Setting up kernel ZRAM (${SWAP_SIZE_GB}GB, LZ4)..."

    modprobe zram num_devices=1

    # Use LZ4 for fair comparison
    echo lz4 > /sys/block/zram0/comp_algorithm
    echo "${SWAP_SIZE_GB}G" > /sys/block/zram0/disksize

    mkswap /dev/zram0 >/dev/null
    swapon /dev/zram0 -p 150

    echo "/dev/zram0"
}

setup_trueno_zram() {
    log_info "Setting up trueno-zram (${SWAP_SIZE_GB}GB)..."

    # Start daemon
    "${PROJECT_ROOT}/target/release/trueno-ublk" --size "${SWAP_SIZE_GB}G" --id 0 &
    local daemon_pid=$!

    # Wait for device
    local retries=10
    while [[ $retries -gt 0 ]] && [[ ! -b /dev/ublkb0 ]]; do
        sleep 1
        ((retries--))
    done

    if [[ ! -b /dev/ublkb0 ]]; then
        log_fail "trueno-ublk device not created"
        return 1
    fi

    # Verify mlock (DT-007)
    local vmlck=$(grep VmLck /proc/$daemon_pid/status 2>/dev/null | awk '{print $2}')
    if [[ -z "$vmlck" ]] || [[ "$vmlck" -lt 100000 ]]; then
        log_warn "DT-007: VmLck = ${vmlck:-0} kB (expected > 100000 kB)"
    else
        log_pass "DT-007: VmLck = $vmlck kB (mlock active)"
    fi

    mkswap /dev/ublkb0 >/dev/null
    swapon /dev/ublkb0 -p 200

    echo "/dev/ublkb0"
}

teardown_swap() {
    local device=$1
    swapoff "$device" 2>/dev/null || true

    case "$device" in
        /swapfile)
            rm -f /swapfile
            ;;
        /dev/zram0)
            echo 1 > /sys/block/zram0/reset 2>/dev/null || true
            rmmod zram 2>/dev/null || true
            ;;
        /dev/ublkb0)
            pkill -9 trueno-ublk 2>/dev/null || true
            sleep 1
            ;;
    esac
}

#═══════════════════════════════════════════════════════════════════════════════
# Benchmark Functions
#═══════════════════════════════════════════════════════════════════════════════

run_fio_benchmark() {
    local name=$1
    local device=$2
    local pattern=$3
    local output_file=$4

    local rw bs extra_opts
    case $pattern in
        SEQ_READ)  rw=read;      bs=1M;  extra_opts="--numjobs=8" ;;
        SEQ_WRITE) rw=write;     bs=1M;  extra_opts="--numjobs=8" ;;
        RAND_READ) rw=randread;  bs=4k;  extra_opts="--iodepth=32 --numjobs=4" ;;
        RAND_WRITE) rw=randwrite; bs=4k; extra_opts="--iodepth=32 --numjobs=4" ;;
        MIXED)     rw=randrw;    bs=4k;  extra_opts="--iodepth=32 --numjobs=4 --rwmixread=70" ;;
    esac

    fio --name="$name" \
        --filename="$device" \
        --rw="$rw" \
        --bs="$bs" \
        --direct=1 \
        $extra_opts \
        --runtime="$TEST_RUNTIME" \
        --time_based \
        --group_reporting \
        --output-format=json \
        --output="$output_file" \
        2>/dev/null
}

extract_metrics() {
    local json_file=$1

    # Extract key metrics from fio JSON output
    local bw_bytes=$(jq -r '.jobs[0].read.bw_bytes // .jobs[0].write.bw_bytes // 0' "$json_file")
    local iops=$(jq -r '.jobs[0].read.iops // .jobs[0].write.iops // 0' "$json_file")
    local lat_p50=$(jq -r '.jobs[0].read.clat_ns.percentile["50.000000"] // .jobs[0].write.clat_ns.percentile["50.000000"] // 0' "$json_file")
    local lat_p99=$(jq -r '.jobs[0].read.clat_ns.percentile["99.000000"] // .jobs[0].write.clat_ns.percentile["99.000000"] // 0' "$json_file")

    # Convert to human-readable
    local bw_gbps=$(echo "scale=2; $bw_bytes / 1000000000" | bc)
    local iops_k=$(echo "scale=1; $iops / 1000" | bc)
    local lat_p50_us=$(echo "scale=1; $lat_p50 / 1000" | bc)
    local lat_p99_us=$(echo "scale=1; $lat_p99 / 1000" | bc)

    echo "$bw_gbps $iops_k $lat_p50_us $lat_p99_us"
}

#═══════════════════════════════════════════════════════════════════════════════
# Main Benchmark Loop
#═══════════════════════════════════════════════════════════════════════════════

run_full_benchmark() {
    log_info "Starting scientific swap benchmark (${RUNS_PER_TEST} runs × ${TEST_RUNTIME}s each)"

    local technologies=("regular_swap" "kernel_zram" "trueno_zram")
    local patterns=("SEQ_READ" "SEQ_WRITE" "RAND_READ" "RAND_WRITE" "MIXED")

    if $QUICK_MODE; then
        patterns=("SEQ_READ" "RAND_READ")
    fi

    # Results array
    declare -A results

    for tech in "${technologies[@]}"; do
        log_info "=== Testing: $tech ==="

        # Setup
        local device
        case $tech in
            regular_swap) device=$(setup_regular_swap) ;;
            kernel_zram)  device=$(setup_kernel_zram) ;;
            trueno_zram)  device=$(setup_trueno_zram) ;;
        esac

        if [[ -z "$device" ]]; then
            log_fail "Failed to setup $tech"
            continue
        fi

        for pattern in "${patterns[@]}"; do
            log_info "  Pattern: $pattern"

            local sum_bw=0 sum_iops=0 sum_lat50=0 sum_lat99=0

            for run in $(seq 1 $RUNS_PER_TEST); do
                local output_file="${RESULTS_DIR}/${tech}_${pattern}_run${run}.json"

                # Drop caches between runs
                sync
                if $IS_LINUX; then
                    echo 3 > /proc/sys/vm/drop_caches
                elif $IS_MACOS; then
                    purge 2>/dev/null || true
                fi
                sleep 1

                run_fio_benchmark "${tech}_${pattern}" "$device" "$pattern" "$output_file"

                read bw iops lat50 lat99 <<< $(extract_metrics "$output_file")

                sum_bw=$(echo "$sum_bw + $bw" | bc)
                sum_iops=$(echo "$sum_iops + $iops" | bc)
                sum_lat50=$(echo "$sum_lat50 + $lat50" | bc)
                sum_lat99=$(echo "$sum_lat99 + $lat99" | bc)

                log_info "    Run $run: ${bw} GB/s, ${iops}K IOPS, P50=${lat50}us, P99=${lat99}us"
            done

            # Calculate averages
            local avg_bw=$(echo "scale=2; $sum_bw / $RUNS_PER_TEST" | bc)
            local avg_iops=$(echo "scale=1; $sum_iops / $RUNS_PER_TEST" | bc)
            local avg_lat50=$(echo "scale=1; $sum_lat50 / $RUNS_PER_TEST" | bc)
            local avg_lat99=$(echo "scale=1; $sum_lat99 / $RUNS_PER_TEST" | bc)

            results["${tech}_${pattern}_bw"]=$avg_bw
            results["${tech}_${pattern}_iops"]=$avg_iops
            results["${tech}_${pattern}_lat50"]=$avg_lat50
            results["${tech}_${pattern}_lat99"]=$avg_lat99

            log_pass "  Average: ${avg_bw} GB/s, ${avg_iops}K IOPS"
        done

        teardown_swap "$device"
        sleep 2
    done

    # Generate report
    generate_report results
}

#═══════════════════════════════════════════════════════════════════════════════
# Compression Benchmark (Cross-Platform)
#═══════════════════════════════════════════════════════════════════════════════

run_compression_benchmark() {
    log_info "Starting compression benchmark (GPU backend: $GPU_BACKEND)"

    local output_file="${RESULTS_DIR}/compression_benchmark.txt"
    local features=""

    if [[ "$GPU_BACKEND" == "cuda" ]]; then
        features="--features cuda"
    elif [[ "$GPU_BACKEND" == "wgpu" ]]; then
        features="--features wgpu"
    fi

    # Run compression benchmark
    log_info "Running trueno-zram compression benchmark..."
    (cd "$PROJECT_ROOT" && cargo run --release --example compress_benchmark $features 2>&1) | tee "$output_file"

    # Parse results
    local compress_gbps decompress_gbps ratio
    compress_gbps=$(grep -E "Text.*Lz4.*GB/s" "$output_file" | head -1 | awk '{print $3}' || echo "N/A")
    decompress_gbps=$(grep -E "Text.*Lz4.*GB/s" "$output_file" | head -1 | awk '{print $4}' || echo "N/A")
    ratio=$(grep -E "Text.*Lz4" "$output_file" | head -1 | awk '{print $5}' || echo "N/A")

    log_pass "Compression: ${compress_gbps} GB/s, Decompression: ${decompress_gbps} GB/s, Ratio: ${ratio}"

    # Generate compression report
    local report_file="${RESULTS_DIR}/COMPRESSION_REPORT.md"
    cat > "$report_file" << EOF
# Compression Benchmark Report

**Generated:** $(date -Iseconds)
**Platform:** $OS_TYPE
**GPU Backend:** $GPU_BACKEND

## Environment

\`\`\`json
$(cat "${RESULTS_DIR}/environment.json")
\`\`\`

## Results

| Pattern | Compress (GB/s) | Decompress (GB/s) | Ratio |
|---------|-----------------|-------------------|-------|
$(grep -E "^\s+\d+\s+(Lz4|Zstd)" "$output_file" | while read line; do echo "| $line |"; done || echo "| See raw output |")

## Raw Output

\`\`\`
$(cat "$output_file")
\`\`\`

---

*Report generated by scientific-swap-benchmark.sh (compression mode)*
EOF

    log_pass "Compression report: $report_file"

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "              COMPRESSION BENCHMARK RESULTS"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "Platform: $OS_TYPE"
    echo "GPU Backend: $GPU_BACKEND"
    echo ""
    cat "$output_file" | head -40
    echo ""
    echo "Full report: $report_file"
    echo "════════════════════════════════════════════════════════════════"
}

#═══════════════════════════════════════════════════════════════════════════════
# Report Generation
#═══════════════════════════════════════════════════════════════════════════════

generate_report() {
    local -n res=$1

    local report_file="${RESULTS_DIR}/REPORT.md"

    cat > "$report_file" << 'EOF'
# Scientific Swap Benchmark Report

**Generated:** TIMESTAMP
**Benchmark ID:** BENCH_ID

## Environment

```
EOF

    cat "${RESULTS_DIR}/environment.json" >> "$report_file"

    cat >> "$report_file" << 'EOF'
```

## Results Summary

### Sequential Read (1MB blocks, 8 threads)

| Technology | Throughput (GB/s) | P50 Latency (μs) | P99 Latency (μs) |
|------------|-------------------|------------------|------------------|
EOF

    for tech in regular_swap kernel_zram trueno_zram; do
        local key_bw="${tech}_SEQ_READ_bw"
        local key_lat50="${tech}_SEQ_READ_lat50"
        local key_lat99="${tech}_SEQ_READ_lat99"
        local bw=${res[$key_bw]:-N/A}
        local lat50=${res[$key_lat50]:-N/A}
        local lat99=${res[$key_lat99]:-N/A}
        echo "| $tech | $bw | $lat50 | $lat99 |" >> "$report_file"
    done

    cat >> "$report_file" << 'EOF'

### Random Read (4KB blocks, 32 iodepth)

| Technology | IOPS (K) | P50 Latency (μs) | P99 Latency (μs) |
|------------|----------|------------------|------------------|
EOF

    for tech in regular_swap kernel_zram trueno_zram; do
        local key_iops="${tech}_RAND_READ_iops"
        local key_lat50="${tech}_RAND_READ_lat50"
        local key_lat99="${tech}_RAND_READ_lat99"
        local iops=${res[$key_iops]:-N/A}
        local lat50=${res[$key_lat50]:-N/A}
        local lat99=${res[$key_lat99]:-N/A}
        echo "| $tech | $iops | $lat50 | $lat99 |" >> "$report_file"
    done

    cat >> "$report_file" << 'EOF'

## Falsification Analysis

### Claim: Sequential I/O 12x faster than kernel ZRAM

EOF

    local trueno_seq=${res["trueno_zram_SEQ_READ_bw"]:-0}
    local kernel_seq=${res["kernel_zram_SEQ_READ_bw"]:-1}
    local speedup=$(echo "scale=2; $trueno_seq / $kernel_seq" | bc 2>/dev/null || echo "N/A")

    if [[ "$speedup" != "N/A" ]] && (( $(echo "$speedup >= 5" | bc -l) )); then
        echo "**PASS**: trueno-zram ${trueno_seq} GB/s vs kernel ZRAM ${kernel_seq} GB/s = **${speedup}x** (threshold: 5x)" >> "$report_file"
    else
        echo "**FAIL**: trueno-zram ${trueno_seq} GB/s vs kernel ZRAM ${kernel_seq} GB/s = **${speedup}x** (threshold: 5x)" >> "$report_file"
    fi

    cat >> "$report_file" << 'EOF'

### Claim: 228K Random IOPS

EOF

    local trueno_iops=${res["trueno_zram_RAND_READ_iops"]:-0}

    if (( $(echo "$trueno_iops >= 180" | bc -l 2>/dev/null || echo 0) )); then
        echo "**PASS**: ${trueno_iops}K IOPS (threshold: 180K)" >> "$report_file"
    else
        echo "**FAIL**: ${trueno_iops}K IOPS (threshold: 180K)" >> "$report_file"
    fi

    cat >> "$report_file" << EOF

---

*Report generated by scientific-swap-benchmark.sh*
*See specification: docs/specifications/scientific-swap-benchmark.md*
EOF

    # Replace placeholders (macOS sed compatibility)
    if $IS_MACOS; then
        sed -i '' "s/TIMESTAMP/$(date -Iseconds)/" "$report_file"
        sed -i '' "s/BENCH_ID/BENCH-001-$(date +%Y%m%d-%H%M%S)/" "$report_file"
    else
        sed -i "s/TIMESTAMP/$(date -Iseconds)/" "$report_file"
        sed -i "s/BENCH_ID/BENCH-001-$(date +%Y%m%d-%H%M%S)/" "$report_file"
    fi

    log_pass "Report generated: $report_file"

    # Print summary to console
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "                    BENCHMARK RESULTS SUMMARY"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "Sequential Read (GB/s):"
    echo "  Regular Swap:  ${res["regular_swap_SEQ_READ_bw"]:-N/A}"
    echo "  Kernel ZRAM:   ${res["kernel_zram_SEQ_READ_bw"]:-N/A}"
    echo "  trueno-zram:   ${res["trueno_zram_SEQ_READ_bw"]:-N/A}"
    echo ""
    echo "Random Read (K IOPS):"
    echo "  Regular Swap:  ${res["regular_swap_RAND_READ_iops"]:-N/A}"
    echo "  Kernel ZRAM:   ${res["kernel_zram_RAND_READ_iops"]:-N/A}"
    echo "  trueno-zram:   ${res["trueno_zram_RAND_READ_iops"]:-N/A}"
    echo ""
    echo "Speedup vs Kernel ZRAM: ${speedup}x"
    echo ""
    echo "Full report: $report_file"
    echo "════════════════════════════════════════════════════════════════"
}

#═══════════════════════════════════════════════════════════════════════════════
# Main
#═══════════════════════════════════════════════════════════════════════════════

main() {
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║     SCIENTIFIC SWAP BENCHMARK - BENCH-001                        ║"
    echo "║     Platform: $OS_TYPE | GPU: $GPU_BACKEND                       ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""

    check_requirements
    setup_environment

    if $IS_LINUX; then
        # Linux: Full swap technology comparison
        log_info "Running full swap benchmark (Regular Swap vs Kernel ZRAM vs trueno-zram)"
        run_full_benchmark

        # Also run compression benchmark for complete picture
        log_info "Running compression benchmark..."
        run_compression_benchmark
    elif $IS_MACOS; then
        # macOS: Compression benchmark only (no zram/ublk)
        log_info "macOS: Running compression benchmark (swap tests not available)"
        run_compression_benchmark
    else
        log_fail "Unsupported platform: $OS_TYPE"
        exit 1
    fi

    log_pass "Benchmark complete!"
}

main "$@"
