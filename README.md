# simd-rand - SIMD implementations of random number generators

Provides SIMD implementations of common PRNGs in Rust. 
Categories:
- [`portable`] - TODO
- [`specific`] - implementations using architecture-specific hardware intrinsics
  - [`specific::avx2`] - AVX2 for x86_64 architecture (4 lanes for 64bit)
  - [`specific::avx512`] - AVX512 for x86_64 architecture (8 lanes for 64bit)

This library is meant to be used in higly performance codepaths typically using
hardware intrisics to accelerate compute, for example 
[Monte Carlo simulations](https://github.com/martinothamar/building-x-in-y/tree/main/monte-carlo-sim/rust).

This library is under active development. No version has been published to cargo yet.

Sources:
* [PRNG shootout by Sebastiano Vigna](https://prng.di.unimi.it/)

## Performance

The top performing generator is currently Xoshiro256+ using AVX2 intrinsics.
It is about 5x faster. The blow benchmarks generates `u64` numbers in a loop

```
top/Original/Xoshiro256+/64
                        time:   [418.06 ns 419.05 ns 420.18 ns]
                        thrpt:  [9.0788 GiB/s 9.1031 GiB/s 9.1248 GiB/s]
top/AVX512/Xoshiro256+/64
                        time:   [75.274 ns 75.668 ns 76.059 ns]
                        thrpt:  [50.155 GiB/s 50.413 GiB/s 50.677 GiB/s]
```

### Benchmarks

Here is an excerpt from benchmarks run on my laptop with the 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz CPU.
For accurate benchmarks for your usecase, you should run them on your own hardware, see the `bench` target in the [Makefile](/Makefile).

Time measured is per iteration, where an iteration generates lanecount (4/8 for AVX2/512) numbers (64bit) 8 times 
 = 8 * 8 * 8 = 512 bytes per time measured in the AVX512 case.

```
AVX2/m256i/Shishua/Time/8
                        time:   [9.9825 ns 10.099 ns 10.235 ns]
                        thrpt:  [23.293 GiB/s 23.608 GiB/s 23.884 GiB/s]
AVX2/m256i/Xoshiro256++/Time/8
                        time:   [10.411 ns 10.451 ns 10.495 ns]
                        thrpt:  [22.717 GiB/s 22.813 GiB/s 22.900 GiB/s]
AVX2/m256i/Xoshiro256+/Time/8
                        time:   [9.0401 ns 9.0851 ns 9.1293 ns]
                        thrpt:  [26.116 GiB/s 26.243 GiB/s 26.373 GiB/s]


AVX2/m256d/Shishua/Time/8
                        time:   [10.676 ns 10.740 ns 10.810 ns]
                        thrpt:  [22.055 GiB/s 22.198 GiB/s 22.332 GiB/s]
AVX2/m256d/Xoshiro256++/Time/8
                        time:   [12.566 ns 12.649 ns 12.731 ns]
                        thrpt:  [18.727 GiB/s 18.848 GiB/s 18.974 GiB/s]
AVX2/m256d/Xoshiro256+/Time/8
                        time:   [10.848 ns 10.916 ns 10.985 ns]
                        thrpt:  [21.704 GiB/s 21.841 GiB/s 21.978 GiB/s]


AVX512/m512i/Xoshiro256++/Time/8
                        time:   [13.510 ns 13.650 ns 13.797 ns]
                        thrpt:  [34.561 GiB/s 34.933 GiB/s 35.295 GiB/s]
AVX512/m512i/Xoshiro256+/Time/8
                        time:   [12.307 ns 12.470 ns 12.616 ns]
                        thrpt:  [37.797 GiB/s 38.238 GiB/s 38.746 GiB/s]


AVX512/m512d/Xoshiro256++/Time/8
                        time:   [16.064 ns 16.138 ns 16.209 ns]
                        thrpt:  [29.419 GiB/s 29.547 GiB/s 29.684 GiB/s]
AVX512/m512d/Xoshiro256+/Time/8
                        time:   [14.425 ns 14.515 ns 14.607 ns]
                        thrpt:  [32.644 GiB/s 32.852 GiB/s 33.057 GiB/s]
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
