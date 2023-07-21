//! # simd-rand - SIMD implementations of random number generators
//! 
//! Provides SIMD implementations of common PRNGs in Rust. 
//! Categories:
//! - [`portable`] - TODO
//! - [`specific`] - implementations using architecture-specific hardware intrinsics
//!   - [`specific::avx2`] - AVX2 for x86_64 architecture (4 lanes for 64bit)
//!   - [`specific::avx512`] - AVX512 for x86_64 architecture (8 lanes for 64bit)
//! 
//! This library is meant to be used in higly performance codepaths typically using
//! hardware intrisics to accelerate compute, for example 
//! [Monte Carlo simulations](https://github.com/martinothamar/building-x-in-y/tree/main/monte-carlo-sim/rust).
//! 
//! This library is under active development. No version has been published to cargo yet.
//! 
//! ## Performance
//! 
//! TODO - benchmarks, guidance
//! 
//! ## Safety
//! 
//! There is a decent amount of `unsafe` used, due to direct use of hardware intrisics (e.g. `__m256{i|d}` for AVX2).
//! `unsafe` will also generally be used for performance optimizations, since the purpose of this library is to provide
//! high performance in vectorized codepaths. 
//! If you don't need that kind of performance, stick to [rand](https://docs.rs/rand) and [rand_core](https://docs.rs/rand_core)
//! 
//! There is also some inline assembly used, where the C-style intrinsics haven't been exposed as Rust APIs in `std::arch`.

#![cfg_attr(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512dq"), feature(stdsimd))]
#![feature(portable_simd)]

pub mod portable;
pub mod specific;

#[cfg(test)]
mod testutil;
