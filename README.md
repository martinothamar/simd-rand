# simd-rand - SIMD implementations of random number generators

Provides SIMD implementations of common PRNGs in Rust. 
Categories:
- [`portable`] - portable implementations using `std::simd` (nightly required) 
- [`specific`] - implementations using architecture-specific hardware intrinsics
  - [`specific::avx2`] - AVX2 for x86_64 architecture (4 lanes for 64bit)
  - [`specific::avx512`] - AVX512 for x86_64 architecture (8 lanes for 64bit)

Vectorized PRNG implementations may perform anywhere from 4-6 times faster in my experience,
of course very dependent on hardware used ("old" CPUs with AVX512 for example may have excessive thermal throttling).
I'm no expert in statistical testing, but I have ran some of these through practrand using random bytes from the `rand` crate
as seeding, and vectorized and non-vectorized implementations seemed to perform identically.

This library is meant to be used in high performance codepaths typically using
hardware intrisics to accelerate compute, for example 
[Monte Carlo simulations](https://github.com/martinothamar/building-x-in-y/tree/main/monte-carlo-sim/rust).

This library is under active development. No version has been published to cargo yet.

Choice of PRNGs, unvectorized sources and general advice has been taken from the great [PRNG shootout by Sebastiano Vigna](https://prng.di.unimi.it/).

## Usage

```toml
[dependencies]
simd_rand = { git = "https://github.com/martinothamar/simd-rand" }
```

```rust
use rand_core::{RngCore, SeedableRng};
use simd_rand::portable::*;

fn main() {
    let mut seed: Xoshiro256PlusPlusX8Seed = Default::default();
    rand::thread_rng().fill_bytes(&mut *seed);
    let mut rng = Xoshiro256PlusPlusX8::from_seed(seed);

    let vector = rng.next_u64x8();
}
```

## Performance

The top performing generator (on my current hardware) is currently Xoshiro256+ using AVX512 intrinsics.
It is about 5.9x faster. The below benchmarks generates `u64x8` numbers in a loop.
Note that the RandVectorized variant uses `simd_support` from the rand crate,
but this doesn't actually vectorize random number generation.

If you want to actually use these generators, you should benchmark them yourself on your own hardware. See the `bench` target in the [Makefile](/Makefile).
Benchmark results below is from a laptop with 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz CPU.
There is a `portable` variant of Xoshiro256+ for u64x8/f64x8 as well, but in those cases no guarantees are made about performance. The compiler
decides what to do with the vectors, whereas with AVX512 specific ones in the `specific` module will either not compile or run very fast.

```
top/Rand/Xoshiro256+/64
                        time:   [390.00 ns 393.60 ns 397.90 ns]
                        thrpt:  [9.5872 GiB/s 9.6918 GiB/s 9.7812 GiB/s]
slope  [390.00 ns 397.90 ns] R^2            [0.8089080 0.8052865]
mean   [388.88 ns 394.12 ns] std. dev.      [9.3148 ns 17.684 ns]
median [385.26 ns 390.14 ns] med. abs. dev. [6.5440 ns 12.296 ns]


top/RandVectorized/Xoshiro256+/64
                        time:   [470.69 ns 474.05 ns 477.66 ns]
                        thrpt:  [7.9861 GiB/s 8.0470 GiB/s 8.1044 GiB/s]
slope  [470.69 ns 477.66 ns] R^2            [0.8833014 0.8824255]
mean   [471.13 ns 477.21 ns] std. dev.      [12.249 ns 18.496 ns]
median [468.56 ns 474.69 ns] med. abs. dev. [9.4037 ns 16.385 ns]


top/AVX512/Xoshiro256+/64
                        time:   [66.718 ns 67.222 ns 67.722 ns]
                        thrpt:  [56.329 GiB/s 56.747 GiB/s 57.176 GiB/s]
slope  [66.718 ns 67.722 ns] R^2            [0.9055332 0.9056426]
mean   [66.456 ns 67.282 ns] std. dev.      [1.8470 ns 2.3434 ns]
median [66.435 ns 67.161 ns] med. abs. dev. [1.6376 ns 3.0408 ns]
```

## Safety

There is a decent amount of `unsafe` used, due to direct use of hardware intrisics (e.g. `__m256{i|d}` for AVX2).
`unsafe` will also generally be used for performance optimizations, since the purpose of this library is to provide
high performance in vectorized codepaths. 
If you don't need that kind of performance, stick to [rand](https://docs.rs/rand) and [rand_core](https://docs.rs/rand_core)

There is also some inline assembly used, where the C-style intrinsics haven't been exposed as Rust APIs in `std::arch`.

## Notes

Prereqs for disassembly:

```sh
cargo install cargo-binutils
rustup component add llvm-tools-preview
```

Then you can run `make dasm`

## TODO

* Implement jumps between lanes for Xoshiro-variants
* More PRNGs
* More docs
* Cleanup code around seeding
