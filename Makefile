bindir := ./target/release/examples
outbin := ${bindir}/profile
TARGET := x86_64-unknown-linux-gnu
RUSTFLAGS_AVX512 := -C target-feature=+avx2,+avx512f,+avx512dq,+avx512vl
CARGO_NIGHTLY := cargo +nightly

all: run

setup:
	cargo install --locked cargo-nextest

fmt:
	cargo fmt

lint:
	RUSTFLAGS="$(RUSTFLAGS_AVX512)" $(CARGO_NIGHTLY) clippy --all-targets --all-features --target $(TARGET)

test:
	cargo nextest run && cargo nextest run --release

test-miri:
	$(CARGO_NIGHTLY) miri test -p simd_rand --no-default-features --features portable --lib

bench:
	$(CARGO_NIGHTLY) bench --features portable -- "$(F)" --verbose

stat: build
	perf stat -d -d -d $(outbin)

build:
	RUSTFLAGS="$(RUSTFLAGS_AVX512)" $(CARGO_NIGHTLY) build --all-targets --all-features

check:
	cargo check

run: build
	$(outbin)

dasm:
	$(CARGO_NIGHTLY) objdump --example dasm --features portable --release -- \
	-d -M intel > $(bindir)/dasm.asm 2> $(bindir)/dasm.asm.log

dasmbench:
	$(CARGO_NIGHTLY) objdump --bench main --features portable --release -- \
	-d -M intel > target/release/bench.asm 2> target/release/bench.asm.log

asm:
	$(CARGO_NIGHTLY) rustc --release --example dasm --features portable -- --emit asm -C "llvm-args=-x86-asm-syntax=intel"

dasmexp: dasm
	$(CARGO_NIGHTLY) rustc --release --example dasm --features portable -- --emit asm=/dev/stdout | c++filt > $(bindir)/dasm.S

# Tested on Ubuntu 22 with bash - run this as a one-off
# may need to fix the compiler warning for RNG_test.cpp (I segfaulted otherwise)
getpractrand:
	mkdir -p external/PractRand/ && \
	pushd external/PractRand/ && \
	curl -OL https://downloads.sourceforge.net/project/pracrand/PractRand_0.93.zip && \
	unzip -q PractRand_0.93.zip && \
	curl -sL http://www.pcg-random.org/downloads/practrand-0.93-bigbuffer.patch | patch -p0 && \
	g++ -std=c++14 -c src/*.cpp src/RNGs/*.cpp src/RNGs/other/*.cpp -O3 -Iinclude -pthread && \
	ar rcs libPractRand.a *.o && rm *.o && \
	g++ -std=c++14 -o RNG_test tools/RNG_test.cpp libPractRand.a -O3 -march=native -Iinclude -pthread && \
	popd

practrand:
	$(CARGO_NIGHTLY) build --example practrand --features portable --release && ./target/release/examples/practrand | ./external/PractRand/RNG_test stdin64 -multithreaded

clean:
	cargo clean --release && cargo clean
