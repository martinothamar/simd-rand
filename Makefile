bindir := ./target/release
outbin := ${bindir}/profile

all: run

test:
	cargo test --lib --release

memtest:
	RUSTFLAGS="--cfg mem_test" cargo test --lib --release -- --test-threads=1

benchshishua:
	cargo bench --bench shishua -- --verbose --save-baseline shishua

benchcomparison:
	cargo bench --bench vectorized -- --verbose --save-baseline vectorized

benchxoshiro:
	cargo bench --bench xoshiro -- --verbose --save-baseline xoshiro

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
