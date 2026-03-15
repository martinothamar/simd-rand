bindir := ./target/release/examples
outbin := ${bindir}/profile
TARGET := x86_64-unknown-linux-gnu
RUSTFLAGS_AVX512 := -C target-feature=+avx2,+avx512f,+avx512dq,+avx512vl
CARGO_NIGHTLY := cargo +nightly
BENCH_CPU ?= 31
DASM_SYMBOLS ?= do_u64x4_xoshiro_baseline,do_u64x4_xoshiro_portable,do_u64x4_portable_frand,do_u64x4_xoshiro_specific,do_u64x4_specific_frand,do_u64x8_xoshiro_baseline,do_u64x8_frand_baseline,do_u64x8_xoshiro_portable,do_u64x8_portable_frand,do_u64x8_xoshiro_specific,do_u64x8_specific_frand,do_f64x4_xoshiro_specific,do_f64x4_specific_frand,do_f64x4_xoshiro_portable,do_f64x4_portable_frand
PRACTRAND_VERSION ?= 0.96
PRACTRAND_ROOT := external/PractRand
PRACTRAND_ARCHIVE := PractRand_$(PRACTRAND_VERSION).zip
PRACTRAND_DIR := $(PRACTRAND_ROOT)/PractRand
PRACTRAND_RNG_TEST := $(PRACTRAND_DIR)/RNG_test
PRACTRAND_RNG ?= portable-frand-x8
PRACTRAND_SEED ?= 0
PRACTRAND_ARGS ?= stdin64 -multithreaded

all: run

setup:
	cargo install --locked cargo-nextest

fmt:
	cargo fmt

lint:
	RUSTFLAGS="$(RUSTFLAGS_AVX512)" $(CARGO_NIGHTLY) clippy --all-targets --all-features --target $(TARGET)

test:
	# Cover the default local build, which on this machine means specific backends are available.
	cargo nextest run && cargo nextest run --release
	# Cover portable-only explicitly; omitting RUSTFLAGS is not enough because .cargo/config.toml uses target-cpu=native.
	$(CARGO_NIGHTLY) nextest run --no-default-features --features portable --target $(TARGET)
	$(CARGO_NIGHTLY) nextest run --release --no-default-features --features portable --target $(TARGET)
	# Cover the combined portable+specific build with the full SIMD feature set used locally.
	RUSTFLAGS="$(RUSTFLAGS_AVX512)" $(CARGO_NIGHTLY) nextest run --features portable --target $(TARGET)
	RUSTFLAGS="$(RUSTFLAGS_AVX512)" $(CARGO_NIGHTLY) nextest run --release --features portable --target $(TARGET)

test-miri:
	$(CARGO_NIGHTLY) miri test -p simd_rand --no-default-features --features portable --lib

doc:
	RUSTDOCFLAGS="--cfg docsrs $(RUSTFLAGS_AVX512)" RUSTFLAGS="$(RUSTFLAGS_AVX512)" $(CARGO_NIGHTLY) doc --all-features --no-deps

bench:
	$(CARGO_NIGHTLY) bench --features portable -- "$(F)" --verbose

bench-top:
	# Default benchmark setup here is adapted to my machine, currently a 9950X3D CPU
	# It has 32 logical cores, of which I've dedicated 1 SMT pair (15 and 31) to benchmarking currently.
	# To improve stability I use taskset to pin to a specific core.
	# Avoid scheduling userspace processes to these cores by doing something like
	# `nohz_full=15,31 rcu_nocbs=15,31 isolcpus=managed_irq,domain,15,31 irqaffinity=0-14,16-30 systemd.cpu_affinity=0-14,16-30`
	# in limine.conf cmdline in my case (I run cachyOS atm)
	@test "$$(cat /sys/devices/system/cpu/cpu$(BENCH_CPU)/cpufreq/scaling_governor)" = performance || \
		(echo "cpu$(BENCH_CPU) scaling governor must be performance" >&2; exit 1)
	taskset --cpu-list $(BENCH_CPU) env RUSTFLAGS="$(RUSTFLAGS_AVX512)" \
		$(CARGO_NIGHTLY) bench --features portable -- "Top" --warm-up-time 5 --measurement-time 10 --sample-size 200

stat: build
	perf stat -d -d -d $(outbin)

build:
	RUSTFLAGS="$(RUSTFLAGS_AVX512)" $(CARGO_NIGHTLY) build --all-targets --all-features

check:
	cargo check

run: build
	$(outbin)

dasm:
	RUSTFLAGS="$(RUSTFLAGS_AVX512)" $(CARGO_NIGHTLY) objdump --example dasm --features portable --release -- \
		--disassemble --disassemble-symbols=$(DASM_SYMBOLS) --x86-asm-syntax=intel \
		--no-show-raw-insn --no-leading-addr > $(bindir)/dasm.asm 2> $(bindir)/dasm.asm.log

dasmbench:
	$(CARGO_NIGHTLY) objdump --bench main --features portable --release -- \
	-d -M intel > target/release/bench.asm 2> target/release/bench.asm.log

asm:
	$(CARGO_NIGHTLY) rustc --release --example dasm --features portable -- --emit asm -C "llvm-args=-x86-asm-syntax=intel"

dasmexp: dasm
	$(CARGO_NIGHTLY) rustc --release --example dasm --features portable -- --emit asm=/dev/stdout | c++filt > $(bindir)/dasm.S

# Tested on Ubuntu 22 with bash - run this as a one-off
getpractrand:
	mkdir -p $(PRACTRAND_ROOT) && \
	rm -rf $(PRACTRAND_DIR) && \
	curl -fL https://downloads.sourceforge.net/project/pracrand/$(PRACTRAND_ARCHIVE) -o $(PRACTRAND_ROOT)/$(PRACTRAND_ARCHIVE) && \
	unzip -qo $(PRACTRAND_ROOT)/$(PRACTRAND_ARCHIVE) -d $(PRACTRAND_ROOT) && \
	pushd $(PRACTRAND_DIR) && \
	g++ -std=c++11 -c src/*.cpp src/RNGs/*.cpp src/RNGs/other/*.cpp -O3 -Iinclude -pthread && \
	ar rcs libPractRand.a *.o && rm *.o && \
	g++ -std=c++11 -o RNG_test tools/RNG_test.cpp libPractRand.a -O3 -Iinclude -pthread && \
	popd

practrand:
	$(CARGO_NIGHTLY) build --example practrand --features portable --release && ./target/release/examples/practrand $(PRACTRAND_RNG) $(PRACTRAND_SEED) | $(PRACTRAND_RNG_TEST) $(PRACTRAND_ARGS)

clean:
	cargo clean --release && cargo clean
