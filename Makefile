bindir := ./target/release/examples
outbin := ${bindir}/profile
TARGET := x86_64-unknown-linux-gnu
RUSTFLAGS_AVX512 := -C target-feature=+avx2,+avx512f,+avx512dq,+avx512vl

all: run

fmt:
	cargo fmt

lint:
	RUSTFLAGS="$(RUSTFLAGS_AVX512)" cargo clippy --all-targets --target $(TARGET) -- -D warnings

test:
	cargo test && cargo test --release

test-miri:
	cargo +nightly miri test -p simd_rand -- portable

bench:
	cargo bench -- "$(F)" --verbose

stat: build
	perf stat -d -d -d $(outbin)

build:
	cargo build --release --example profile

check:
	cargo check

run: build
	$(outbin)

dasm:
	cargo objdump --example dasm --release -- \
	-d -M intel > $(bindir)/dasm.asm 2> $(bindir)/dasm.asm.log

dasmbench:
	cargo objdump --bench main --release -- \
	-d -M intel > target/release/bench.asm 2> target/release/bench.asm.log

asm:
	cargo rustc --release --example dasm -- --emit asm -C "llvm-args=-x86-asm-syntax=intel"

dasmexp: dasm
	cargo rustc --release --bin dasm -- --emit asm=/dev/stdout | c++filt > src/bin/dasm.S

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
	cargo build --example practrand --release && ./target/release/examples/practrand | ./external/PractRand/RNG_test stdin64 -multithreaded

clean:
	cargo clean --release && cargo clean
