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

The top performing generator (on my current hardware) is currently Xoshiro256+ using AVX512 instruction set.
It is about 6x faster. The below benchmarks generates `u64x8` numbers.
Note that the RandVectorized variant uses `simd_support` from the `rand` crate,
but this doesn't actually vectorize random number generation.

If you want to actually use these generators, you should benchmark them yourself on your own hardware. See the `bench` target in the [Makefile](/Makefile).
Benchmark results below is from desktop with an AMD Ryzen 9 9950X3D 16-Core CPU.
There is a `portable` variant of Xoshiro256+ for u64x8/f64x8 as well, but in those cases no guarantees are made about performance. The compiler
decides what to do with the vectors, whereas with AVX512 specific ones in the `specific` module will either not compile or run very fast.

```text
Top/rand/Xoshiro256+
                        time:   [24.963 ns 24.964 ns 24.965 ns]
                        thrpt:  [2.5636 Gelem/s 2.5637 Gelem/s 2.5638 Gelem/s]
                        thrpt:  [19.100 GiB/s 19.101 GiB/s 19.102 GiB/s]
Top/simd_rand/Portable/Xoshiro256+X8
                        time:   [6.9966 ns 6.9975 ns 6.9983 ns]
                        thrpt:  [9.1451 Gelem/s 9.1462 Gelem/s 9.1473 Gelem/s]
                        thrpt:  [68.136 GiB/s 68.144 GiB/s 68.152 GiB/s]
Top/simd_rand/Specific/Xoshiro256+X8
                        time:   [6.9453 ns 6.9455 ns 6.9457 ns]
                        thrpt:  [9.2144 Gelem/s 9.2147 Gelem/s 9.2149 Gelem/s]
                        thrpt:  [68.652 GiB/s 68.655 GiB/s 68.657 GiB/s]
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
