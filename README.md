# simd-rand - SIMD implementations of random number generators

Provides SIMD implementations of common PRNGs in Rust. 
Categories:
- [`portable`] - portable implementations using `std::simd` (feature `portable`, nightly required) 
- [`specific`] - implementations using architecture-specific hardware intrinsics
  - [`specific::avx2`] - AVX2 for x86_64 architecture (4 lanes for 64bit)
    - Requires `avx2` CPU flag, but has additional optimization if you have `avx512dq` and `avx512vl`
  - [`specific::avx512`] - AVX512 for x86_64 architecture (8 lanes for 64bit)
    - Requires `avx512f`, `avx512dq` CPU flags

Vectorized PRNG implementations may perform anywhere from 4-6 times faster in my experience,
of course very dependent on hardware used ("old" CPUs with AVX512 for example may have excessive thermal throttling).
I'm no expert in statistical testing, but I have ran some of these through practrand using random bytes from the `rand` crate
as seeding, and vectorized and non-vectorized implementations seemed to perform identically.

This library is meant to be used in high performance codepaths typically using
hardware intrisics to accelerate compute, for example 
[Monte Carlo simulations](https://github.com/martinothamar/building-x-in-y/tree/main/monte-carlo-sim/rust).

Choice of PRNGs, unvectorized sources and general advice has been taken from the great [PRNG shootout resource by Sebastiano Vigna](https://prng.di.unimi.it/).

## Usage

MSRV: 1.89.

```toml
[dependencies]
simd_rand = "0.1"
```

Portable SIMD on nightly:

```toml
[dependencies]
simd_rand = { version = "0.1", features = ["portable"] }
```

The example below uses `portable`, which requires the `portable` feature and a nightly toolchain until `std::simd` is stabilized.

```rust
use rand_core::{RngCore, SeedableRng};
use simd_rand::portable::*;

fn main() {
    let mut seed: Xoshiro256PlusPlusX8Seed = Default::default();
    rand::rng().fill_bytes(&mut *seed);
    let mut rng = Xoshiro256PlusPlusX8::from_seed(seed);

    let vector = rng.next_u64x8();
}
```

The `portable` module will be available on any architecture, e.g. even on x86_64 with only AVX2 you can still use `Xoshiro256PlusPluxX8` which uses
8-lane/512bit vectors (u64x8 from `std::simd`). The compiler is able to make it reasonably fast even if using only 256bit wide registers (AVX2) in the generated code.

The `specific` submodules (AVX2 and AVX512 currently) are only compiled in depending on target arch/features.

In general, use the `portable` module. The only risk/drawback to using the `portable` module is that in principle
the compiler isn't _forced_ to use the "optimal" instructions and registers for your hardware. In practice, it probably will though.
In the `specific` submodules the respective hardware intrinsics are "hardcoded" so to speak so we always know what the generated code looks like.
In some contexts that may be useful.

## Performance

The fastest result below on my current hardware is currently `simd_rand/Specific/FrandX8`,
with `simd_rand/Portable/FrandX8` effectively tied.
These top benchmarks generate and accumulate `u64x8` batches to keep the work observable to the compiler.

If you want to actually use these generators, you should benchmark them yourself on your own hardware. See the `bench` target in the [Makefile](/Makefile).
Benchmark results below is from desktop with an AMD Ryzen 9 9950X3D 16-Core CPU.
There is a `portable` variant of Xoshiro256+ for u64x8/f64x8 as well, but in those cases no guarantees are made about performance. The compiler
decides what to do with the vectors, whereas with AVX512 specific ones in the `specific` module will either not compile or run very fast.

```text
Top/rand/Xoshiro256+
                        time:   [33.102 ns 33.103 ns 33.105 ns]
                        thrpt:  [1.9333 Gelem/s 1.9333 Gelem/s 1.9334 Gelem/s]
                        thrpt:  [14.404 GiB/s 14.404 GiB/s 14.405 GiB/s]
Top/frand
                        time:   [3.1798 ns 3.1800 ns 3.1802 ns]
                        thrpt:  [20.125 Gelem/s 20.126 Gelem/s 20.127 Gelem/s]
                        thrpt:  [149.94 GiB/s 149.95 GiB/s 149.96 GiB/s]
Top/simd_rand/Portable/Xoshiro256+X8
                        time:   [7.5046 ns 7.5081 ns 7.5116 ns]
                        thrpt:  [8.5202 Gelem/s 8.5242 Gelem/s 8.5281 Gelem/s]
                        thrpt:  [63.480 GiB/s 63.510 GiB/s 63.539 GiB/s]
Top/simd_rand/Portable/FrandX8
                        time:   [2.5294 ns 2.5304 ns 2.5313 ns]
                        thrpt:  [25.284 Gelem/s 25.293 Gelem/s 25.302 Gelem/s]
                        thrpt:  [188.38 GiB/s 188.45 GiB/s 188.51 GiB/s]
Top/simd_rand/Specific/Xoshiro256+X8
                        time:   [7.0799 ns 7.0802 ns 7.0805 ns]
                        thrpt:  [9.0389 Gelem/s 9.0393 Gelem/s 9.0396 Gelem/s]
                        thrpt:  [67.345 GiB/s 67.348 GiB/s 67.351 GiB/s]
Top/simd_rand/Specific/FrandX8
                        time:   [2.5287 ns 2.5298 ns 2.5309 ns]
                        thrpt:  [25.288 Gelem/s 25.298 Gelem/s 25.309 Gelem/s]
                        thrpt:  [188.41 GiB/s 188.49 GiB/s 188.57 GiB/s]
```

## Safety

There is a decent amount of `unsafe` used, due to direct use of hardware intrisics (e.g. `__m256{i|d}` for AVX2).
`unsafe` will also generally be used for performance optimizations, since the purpose of this library is to provide
high performance in vectorized codepaths. 
If you don't need that kind of performance, stick to [rand](https://docs.rs/rand) and [rand_core](https://docs.rs/rand_core)

## Notes

Prereqs for disassembly:

```sh
cargo install cargo-binutils
rustup component add llvm-tools-preview
```

Then you can run `make dasm`

## TODO

* Implement jumps between lanes for Xoshiro-variants?
* More PRNGs
* More docs
* Cleanup code around seeding
