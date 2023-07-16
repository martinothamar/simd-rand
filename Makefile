bindir := ./target/release
outbin := ${bindir}/profile

all: run

test:
	cargo test --lib --release

memtest:
	RUSTFLAGS="--cfg mem_test" cargo test --lib --release -- --test-threads=1

benchcomparison:
	cargo bench --bench comparison -- --verbose --save-baseline comparison

benchshishua:
	cargo bench --bench shishua -- --verbose --save-baseline shishua

benchxoshiro:
	cargo bench --bench xoshiro256plusplus -- --verbose --save-baseline xoshiro256plusplus

stat: build
	perf stat -d -d -d ./target/release/profile

build:
	cargo build --release --bin profile

check:
	cargo check

run: build
	$(outbin)

dasm:
	cargo objdump --bin dasm --release -- \
	-d -S -M intel > $(bindir)/dasm.asm 2> $(bindir)/dasm.asm.log

dasmexp: dasm
	cargo rustc --release --bin dasm -- --emit asm=/dev/stdout | c++filt > src/bin/dasm.S

clean:
	cargo clean --release
