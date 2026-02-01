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

This library is under active development. No version has been published to cargo yet.

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

The top performing generator (on my current hardware) is currently Xoshiro256+ using AVX512 instruction set.
It is about 6x faster. The below benchmarks generates `u64x8` numbers.
Note that the RandVectorized variant uses `simd_support` from the `rand` crate,
but this doesn't actually vectorize random number generation.

If you want to actually use these generators, you should benchmark them yourself on your own hardware. See the `bench` target in the [Makefile](/Makefile).
Benchmark results below is from a laptop with 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz CPU.
There is a `portable` variant of Xoshiro256+ for u64x8/f64x8 as well, but in those cases no guarantees are made about performance. The compiler
decides what to do with the vectors, whereas with AVX512 specific ones in the `specific` module will either not compile or run very fast.

```
Top/Rand/Xoshiro256+/1  time:   [5.8505 ns 5.8653 ns 5.8823 ns]
                        thrpt:  [10.133 GiB/s 10.162 GiB/s 10.188 GiB/s]
Found 13 outliers among 100 measurements (13.00%)
  8 (8.00%) high mild
  5 (5.00%) high severe
slope  [5.8505 ns 5.8823 ns] R^2            [0.9830550 0.9828000]
mean   [5.8567 ns 5.8866 ns] std. dev.      [55.633 ps 96.630 ps]
median [5.8421 ns 5.8533 ns] med. abs. dev. [23.662 ps 45.862 ps]


Top/RandVectorized/Xoshiro256+/1
                        time:   [7.1770 ns 7.1938 ns 7.2142 ns]
                        thrpt:  [8.2621 GiB/s 8.2855 GiB/s 8.3050 GiB/s]
Found 13 outliers among 100 measurements (13.00%)
  6 (6.00%) high mild
  7 (7.00%) high severe
slope  [7.1770 ns 7.2142 ns] R^2            [0.9858523 0.9855185]
mean   [7.1769 ns 7.2059 ns] std. dev.      [50.956 ps 94.011 ps]
median [7.1633 ns 7.1748 ns] med. abs. dev. [20.377 ps 38.094 ps]


Top/Portable/Xoshiro256+X8/1
                        time:   [916.36 ps 920.53 ps 925.57 ps]
                        thrpt:  [64.398 GiB/s 64.750 GiB/s 65.045 GiB/s]
Found 1 outliers among 100 measurements (1.00%)
  1 (1.00%) high severe
slope  [916.36 ps 925.57 ps] R^2            [0.9466915 0.9454417]
mean   [916.46 ps 923.16 ps] std. dev.      [13.352 ps 21.906 ps]
median [915.04 ps 925.21 ps] med. abs. dev. [12.841 ps 20.338 ps]


Top/Specific/Xoshiro256+X8/1
                        time:   [961.32 ps 965.08 ps 968.96 ps]
                        thrpt:  [61.514 GiB/s 61.761 GiB/s 62.003 GiB/s]
slope  [961.32 ps 968.96 ps] R^2            [0.9651056 0.9649879]
mean   [963.76 ps 971.23 ps] std. dev.      [16.813 ps 21.217 ps]
median [964.21 ps 975.76 ps] med. abs. dev. [16.026 ps 26.276 ps]
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
