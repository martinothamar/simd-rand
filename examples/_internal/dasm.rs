#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
#![cfg_attr(feature = "portable", feature(portable_simd))]

use biski64::Biski64Rng;
use frand::Rand;
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
/// and `simd_rand` and `rand` code
#[unsafe(no_mangle)]
#[inline(never)]
extern "Rust" fn do_u64x4_xoshiro_baseline(rng: &mut rand_xoshiro::Xoshiro256Plus) -> u64x4 {
    u64x4::from_array([rng.next_u64(), rng.next_u64(), rng.next_u64(), rng.next_u64()])
}

#[unsafe(no_mangle)]
#[inline(never)]
extern "Rust" fn do_u64x4_xoshiro_portable(rng: &mut portable::Xoshiro256PlusX4) -> u64x4 {
    rng.next_u64x4()
}

#[unsafe(no_mangle)]
#[inline(never)]
extern "Rust" fn do_u64x4_portable_frand(rng: &mut portable::FrandX4) -> u64x4 {
    rng.next_u64x4()
}

#[unsafe(no_mangle)]
#[inline(never)]
extern "Rust" fn do_u64x4_portable_biski(rng: &mut portable::Biski64X4) -> u64x4 {
    rng.next_u64x4()
}

#[unsafe(no_mangle)]
#[inline(never)]
extern "Rust" fn do_u64x4_xoshiro_specific(rng: &mut specific::avx2::Xoshiro256PlusX4) -> __m256i {
    rng.next_m256i()
}

#[unsafe(no_mangle)]
#[inline(never)]
extern "Rust" fn do_u64x4_specific_frand(rng: &mut specific::avx2::FrandX4) -> __m256i {
    rng.next_m256i()
}

#[unsafe(no_mangle)]
#[inline(never)]
extern "Rust" fn do_u64x4_specific_biski(rng: &mut specific::avx2::Biski64X4) -> __m256i {
    rng.next_m256i()
}

#[unsafe(no_mangle)]
#[inline(never)]
extern "Rust" fn do_u64x8_xoshiro_baseline(rng: &mut rand_xoshiro::Xoshiro256Plus) -> u64x8 {
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

#[unsafe(no_mangle)]
#[inline(never)]
extern "Rust" fn do_u64x8_frand_baseline(rng: &mut Rand) -> u64x8 {
    u64x8::from_array([
        rng.r#gen::<u64>(),
        rng.r#gen::<u64>(),
        rng.r#gen::<u64>(),
        rng.r#gen::<u64>(),
        rng.r#gen::<u64>(),
        rng.r#gen::<u64>(),
        rng.r#gen::<u64>(),
        rng.r#gen::<u64>(),
    ])
}

#[unsafe(no_mangle)]
#[inline(never)]
extern "Rust" fn do_u64x8_biski_baseline(rng: &mut Biski64Rng) -> u64x8 {
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

#[unsafe(no_mangle)]
#[inline(never)]
extern "Rust" fn do_u64x8_xoshiro_portable(rng: &mut portable::Xoshiro256PlusX8) -> u64x8 {
    rng.next_u64x8()
}

#[unsafe(no_mangle)]
#[inline(never)]
extern "Rust" fn do_u64x8_portable_frand(rng: &mut portable::FrandX8) -> u64x8 {
    rng.next_u64x8()
}

#[unsafe(no_mangle)]
#[inline(never)]
extern "Rust" fn do_u64x8_portable_biski(rng: &mut portable::Biski64X8) -> u64x8 {
    rng.next_u64x8()
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl"
))]
#[unsafe(no_mangle)]
#[inline(never)]
extern "Rust" fn do_u64x8_xoshiro_specific(rng: &mut specific::avx512::Xoshiro256PlusX8) -> __m512i {
    rng.next_m512i()
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl"
))]
#[unsafe(no_mangle)]
#[inline(never)]
extern "Rust" fn do_u64x8_specific_frand(rng: &mut specific::avx512::FrandX8) -> __m512i {
    rng.next_m512i()
}

#[unsafe(no_mangle)]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl"
))]
#[inline(never)]
extern "Rust" fn do_u64x8_specific_biski(rng: &mut specific::avx512::Biski64X8) -> __m512i {
    rng.next_m512i()
}

#[unsafe(no_mangle)]
#[inline(never)]
extern "Rust" fn do_f64x4_xoshiro_specific(rng: &mut specific::avx2::Xoshiro256PlusX4) -> __m256d {
    rng.next_m256d()
}

#[unsafe(no_mangle)]
#[inline(never)]
extern "Rust" fn do_f64x4_specific_frand(rng: &mut specific::avx2::FrandX4) -> __m256d {
    rng.next_m256d()
}

#[unsafe(no_mangle)]
#[inline(never)]
extern "Rust" fn do_f64x4_specific_biski(rng: &mut specific::avx2::Biski64X4) -> __m256d {
    rng.next_m256d()
}

#[unsafe(no_mangle)]
#[inline(never)]
extern "Rust" fn do_f64x4_xoshiro_portable(rng: &mut portable::Xoshiro256PlusX4) -> f64x4 {
    rng.next_f64x4()
}

#[unsafe(no_mangle)]
#[inline(never)]
extern "Rust" fn do_f64x4_portable_frand(rng: &mut portable::FrandX4) -> f64x4 {
    rng.next_f64x4()
}

#[unsafe(no_mangle)]
#[inline(never)]
extern "Rust" fn do_f64x4_portable_biski(rng: &mut portable::Biski64X4) -> f64x4 {
    rng.next_f64x4()
}

fn main() {
    let mut rng_base = rand_xoshiro::Xoshiro256Plus::seed_from_u64(0);
    let mut rng_frand = Rand::with_seed(0);
    let mut rng_biski = Biski64Rng::from_seed_for_stream(0, 0, 1);
    let mut rng_portable_x4 = portable::Xoshiro256PlusX4::seed_from_u64(0);
    let mut rng_portable_frand_x4 = portable::FrandX4::seed_from_u64(0);
    let mut rng_portable_biski_x4 = portable::Biski64X4::seed_from_u64(0);
    let mut rng_specific_x4 = specific::avx2::Xoshiro256PlusX4::seed_from_u64(0);
    let mut rng_specific_frand_x4 = specific::avx2::FrandX4::seed_from_u64(0);
    let mut rng_specific_biski_x4 = specific::avx2::Biski64X4::seed_from_u64(0);
    let mut rng_portable_x8 = portable::Xoshiro256PlusX8::seed_from_u64(0);
    let mut rng_portable_frand_x8 = portable::FrandX8::seed_from_u64(0);
    let mut rng_portable_biski_x8 = portable::Biski64X8::seed_from_u64(0);
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512dq",
        target_feature = "avx512vl"
    ))]
    let mut rng_specific_x8 = specific::avx512::Xoshiro256PlusX8::seed_from_u64(0);
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512dq",
        target_feature = "avx512vl"
    ))]
    let mut rng_specific_frand_x8 = specific::avx512::FrandX8::seed_from_u64(0);
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512dq",
        target_feature = "avx512vl"
    ))]
    let mut rng_specific_biski_x8 = specific::avx512::Biski64X8::seed_from_u64(0);

    black_box(do_u64x4_xoshiro_baseline(&mut rng_base));
    black_box(do_u64x4_xoshiro_portable(&mut rng_portable_x4));
    black_box(do_u64x4_portable_frand(&mut rng_portable_frand_x4));
    black_box(do_u64x4_portable_biski(&mut rng_portable_biski_x4));
    black_box(do_u64x4_xoshiro_specific(&mut rng_specific_x4));
    black_box(do_u64x4_specific_frand(&mut rng_specific_frand_x4));
    black_box(do_u64x4_specific_biski(&mut rng_specific_biski_x4));
    black_box(do_u64x8_xoshiro_baseline(&mut rng_base));
    black_box(do_u64x8_frand_baseline(&mut rng_frand));
    black_box(do_u64x8_biski_baseline(&mut rng_biski));
    black_box(do_u64x8_xoshiro_portable(&mut rng_portable_x8));
    black_box(do_u64x8_portable_frand(&mut rng_portable_frand_x8));
    black_box(do_u64x8_portable_biski(&mut rng_portable_biski_x8));
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512dq",
        target_feature = "avx512vl"
    ))]
    black_box(do_u64x8_xoshiro_specific(&mut rng_specific_x8));
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512dq",
        target_feature = "avx512vl"
    ))]
    black_box(do_u64x8_specific_frand(&mut rng_specific_frand_x8));
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512dq",
        target_feature = "avx512vl"
    ))]
    black_box(do_u64x8_specific_biski(&mut rng_specific_biski_x8));
    black_box(do_f64x4_xoshiro_specific(&mut rng_specific_x4));
    black_box(do_f64x4_specific_frand(&mut rng_specific_frand_x4));
    black_box(do_f64x4_specific_biski(&mut rng_specific_biski_x4));
    black_box(do_f64x4_xoshiro_portable(&mut rng_portable_x4));
    black_box(do_f64x4_portable_frand(&mut rng_portable_frand_x4));
    black_box(do_f64x4_portable_biski(&mut rng_portable_biski_x4));
}
