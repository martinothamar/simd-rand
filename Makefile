bindir := ./target/release
outbin := ${bindir}/profile

all: run

test:
	cargo test --lib --release

memtest:
	RUSTFLAGS="--cfg mem_test" cargo test --lib --release -- --test-threads=1

bench:
	cargo bench --benches

build:
	cargo build --release --bin profile

check:
	cargo check

run: build
	$(outbin)

dasm:
	cargo objdump --bin dasm --release -- \
	-d -S -M intel --no-show-raw-insn > $(bindir)/dasm.asm 2> $(bindir)/dasm.asm.log

clean:
	cargo clean --release
