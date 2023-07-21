bindir := ./target/release/examples
outbin := ${bindir}/profile

all: run

test:
	cargo test && cargo test --release

bench:
	cargo bench -- $(F) --verbose

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

asm:
	cargo rustc --release --example dasm -- --emit asm -C "llvm-args=-x86-asm-syntax=intel"

dasmexp: dasm
	cargo rustc --release --bin dasm -- --emit asm=/dev/stdout | c++filt > src/bin/dasm.S

clean:
	cargo clean --release && cargo clean
