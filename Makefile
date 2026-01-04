# Use bash for shell commands
SHELL := /bin/bash

# Test thread limit - allows parallel test execution
TEST_THREADS ?= $(shell nproc)
export RUST_TEST_THREADS=$(TEST_THREADS)

# PERFORMANCE TARGETS (Toyota Way: Zero Defects, Fast Feedback)
# - make test-fast:  < 30 seconds (minimal property tests)
# - make coverage:   < 2 minutes (core tests, excludes CLI)
# - make test:       comprehensive

.PHONY: build test test-fast lint fmt fmt-check coverage coverage-sudo coverage-quick clean

build:
	cargo build --release

# Fast tests - target: <30 seconds
test-fast:
	@echo "⚡ Running fast tests (target: <30s)..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		PROPTEST_CASES=10 cargo nextest run \
			--workspace \
			--lib \
			--status-level skip \
			--failure-output immediate \
			-E 'not test(/benchmark|stress|fuzz/)'; \
	else \
		PROPTEST_CASES=10 cargo test --workspace --lib; \
	fi
	@echo "✅ Fast tests completed!"

# Comprehensive test suite
test: test-fast
	@echo "🧪 Running comprehensive tests..."
	@cargo test --workspace --lib
	@cargo test --doc --workspace
	@echo "✅ All tests completed!"

lint:
	cargo clippy -- -D warnings

fmt:
	cargo fmt

fmt-check:
	cargo fmt --check

# Coverage exclusion for modules requiring external/root access
COVERAGE_EXCLUDE := --ignore-filename-regex='cli|zram/device\.rs|zram/ops\.rs'

# Fast coverage - target: <90s (cold), <30s (warm)
coverage:
	@echo "📊 Running fast coverage analysis..."
	@which cargo-llvm-cov > /dev/null 2>&1 || (echo "📦 Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov --locked)
	@mkdir -p target/coverage
	@echo "🧪 Running tests with instrumentation..."
	@env PROPTEST_CASES=5 QUICKCHECK_TESTS=5 cargo llvm-cov \
		--no-cfg-coverage \
		--workspace \
		--lib \
		--html --output-dir target/coverage/html \
		$(COVERAGE_EXCLUDE)
	@echo ""
	@echo "💡 HTML report: target/coverage/html/index.html"

# Quick coverage - even faster for iteration (<30s warm)
coverage-quick:
	@echo "⚡ Quick coverage..."
	@env PROPTEST_CASES=1 QUICKCHECK_TESTS=1 cargo llvm-cov \
		--no-cfg-coverage \
		--workspace \
		--lib \
		--html --output-dir target/coverage/html \
		$(COVERAGE_EXCLUDE)
	@cargo llvm-cov report --summary-only $(COVERAGE_EXCLUDE)
	@echo "💡 HTML: target/coverage/html/index.html"

# Coverage with sudo (includes zram integration tests requiring root)
coverage-sudo:
	sudo bash -c 'source $$HOME/.cargo/env && \
		cd $(PWD) && \
		cargo llvm-cov --lib --ignore-filename-regex="cli" --html -- --include-ignored'
	sudo chown -R $(USER):$(USER) target
	@echo "Coverage report: target/llvm-cov/html/index.html"

# CI coverage - LCOV output
coverage-ci:
	@echo "📊 Generating LCOV report for CI..."
	@env PROPTEST_CASES=10 QUICKCHECK_TESTS=10 cargo llvm-cov \
		--no-cfg-coverage \
		--workspace \
		--lib \
		--lcov --output-path lcov.info \
		$(COVERAGE_EXCLUDE)
	@echo "✓ Coverage report: lcov.info"

coverage-open:
	@if [ -f target/coverage/html/index.html ]; then \
		xdg-open target/coverage/html/index.html 2>/dev/null || \
		open target/coverage/html/index.html 2>/dev/null || \
		echo "Open: target/coverage/html/index.html"; \
	else \
		echo "❌ Run 'make coverage' first"; \
	fi

coverage-clean:
	@rm -f lcov.info
	@rm -rf target/llvm-cov target/coverage
	@find . -name "*.profraw" -delete
	@echo "✓ Coverage artifacts cleaned"

clean:
	cargo clean
