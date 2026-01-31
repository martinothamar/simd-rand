#![cfg_attr(feature = "portable", feature(portable_simd))]

use rand_core::{RngCore, SeedableRng};
use simd_rand::portable;
use simd_rand::portable::{SimdRandX4 as PortableSimdRandX4, SimdRandX8 as PortableSimdRandX8};
use simd_rand::specific;
use simd_rand::specific::avx2::SimdRand as SpecificSimdRandX4;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl"
))]
use simd_rand::specific::avx512::SimdRand as SpecificSimdRandX8;
use std::arch::x86_64::*;
use std::hint::black_box;
use std::simd::{f64x4, u64x4, u64x8};

/// This is a small binary meant to aid in analyzing generated code
/// For example to see differences between portable and specific code,
/// and simd_rand and rand code
#[inline(never)]
fn do_u64x4_baseline<RNG: RngCore>(rng: &mut RNG) -> u64x4 {
    u64x4::from_array([rng.next_u64(), rng.next_u64(), rng.next_u64(), rng.next_u64()])
}

#[inline(never)]
fn do_u64x4_portable<RNG: PortableSimdRandX4>(rng: &mut RNG) -> u64x4 {
    rng.next_u64x4()
}

#[inline(never)]
fn do_u64x4_specific<RNG: SpecificSimdRandX4>(rng: &mut RNG) -> __m256i {
    rng.next_m256i()
}

#[inline(never)]
fn do_u64x8_baseline<RNG: RngCore>(rng: &mut RNG) -> u64x8 {
    u64x8::from_array([
        rng.next_u64(),
        rng.next_u64(),
        rng.next_u64(),
        rng.next_u64(),
        rng.next_u64(),
        rng.next_u64(),
        rng.next_u64(),
        rng.next_u64(),
    ])
}

#[inline(never)]
fn do_u64x8_portable<RNG: PortableSimdRandX8>(rng: &mut RNG) -> u64x8 {
    rng.next_u64x8()
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl"
))]
#[inline(never)]
fn do_u64x8_specific<RNG: SpecificSimdRandX8>(rng: &mut RNG) -> __m512i {
    rng.next_m512i()
}

#[inline(never)]
fn do_f64x4_specific<RNG: SpecificSimdRandX4>(rng: &mut RNG) -> __m256d {
    rng.next_m256d()
}

#[inline(never)]
fn do_f64x4_portable<RNG: PortableSimdRandX4>(rng: &mut RNG) -> f64x4 {
    rng.next_f64x4()
}

fn main() {
    let mut rng_base = rand_xoshiro::Xoshiro256Plus::seed_from_u64(0);
    let mut rng1 = portable::Xoshiro256PlusX4::seed_from_u64(0);
    let mut rng2 = specific::avx2::Xoshiro256PlusX4::seed_from_u64(0);
    let mut rng3 = portable::Xoshiro256PlusX8::seed_from_u64(0);
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512dq",
        target_feature = "avx512vl"
    ))]
    let mut rng4 = specific::avx512::Xoshiro256PlusX8::seed_from_u64(0);

    black_box(do_u64x4_baseline(&mut rng_base));
    black_box(do_u64x4_portable(&mut rng1));
    black_box(do_u64x4_specific(&mut rng2));
    black_box(do_u64x8_baseline(&mut rng_base));
    black_box(do_u64x8_portable(&mut rng3));
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512dq",
        target_feature = "avx512vl"
    ))]
    black_box(do_u64x8_specific(&mut rng4));
    black_box(do_f64x4_specific(&mut rng2));
    black_box(do_f64x4_portable(&mut rng1));
}
