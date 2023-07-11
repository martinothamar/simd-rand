bindir := ./target/release
outbin := ${bindir}/profile

ifndef FN
override FN = dsa::ring_buffer::RingBuffer<T,_>::new_inline
endif

all: build run

test:
	cargo test --release -- --test-threads=1

build:
	cargo build --release --bin profile

check:
	cargo check

run:
	$(outbin)

dasm:
	RUSTFLAGS="--cfg dasm" cargo objdump --bin dasm --release -- -d --no-show-raw-insn > $(bindir)/dasm.asm

dasmfn:
	cargo asm "$(FN)"

clean:
	cargo clean --release
