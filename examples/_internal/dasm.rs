#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
#![cfg_attr(feature = "portable", feature(portable_simd))]

use biski64::Biski64Rng;
use frand::Rand;
use rand_core::{RngCore, SeedableRng};
use simd_rand::portable::{SimdRandX4 as PortableSimdRandX4, SimdRandX8 as PortableSimdRandX8};
use simd_rand::specific::avx2::SimdRand as SpecificSimdRandX4;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl"
))]
use simd_rand::specific::avx512::SimdRand as SpecificSimdRandX8;
use std::hint::black_box;
use std::simd::{u64x4, u64x8};

type Shishua = simd_rand::specific::avx2::Shishua<{ simd_rand::specific::avx2::DEFAULT_BUFFER_SIZE }>;

/// This is a small binary meant to aid in analyzing generated code
/// For example to see differences between portable and specific code,
/// and `simd_rand` and `rand` code
#[unsafe(no_mangle)]
#[inline(never)]
extern "Rust" fn do_u64x4_xoshiro256plus_baseline(rng: &mut rand_xoshiro::Xoshiro256Plus) -> u64x4 {
    u64x4::from_array([rng.next_u64(), rng.next_u64(), rng.next_u64(), rng.next_u64()])
}

#[unsafe(no_mangle)]
#[inline(never)]
extern "Rust" fn do_u64x8_xoshiro256plus_baseline(rng: &mut rand_xoshiro::Xoshiro256Plus) -> u64x8 {
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
extern "Rust" fn do_u64x4_xoshiro256plusplus_baseline(rng: &mut rand_xoshiro::Xoshiro256PlusPlus) -> u64x4 {
    u64x4::from_array([rng.next_u64(), rng.next_u64(), rng.next_u64(), rng.next_u64()])
}

#[unsafe(no_mangle)]
#[inline(never)]
extern "Rust" fn do_u64x8_xoshiro256plusplus_baseline(rng: &mut rand_xoshiro::Xoshiro256PlusPlus) -> u64x8 {
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

macro_rules! define_dasm_wrapper {
    ($binding:ident, $fn_name:ident, $rng_ty:path, $ret_ty:ty, $method:ident) => {
        #[unsafe(no_mangle)]
        #[inline(never)]
        extern "Rust" fn $fn_name(rng: &mut $rng_ty) -> $ret_ty {
            rng.$method()
        }
    };
}

macro_rules! for_each_dasm_case {
    ($m:ident) => {
        $m!(
            rng_portable_x4,
            do_u64x4_xoshiro256plus_portable,
            simd_rand::portable::Xoshiro256PlusX4,
            std::simd::u64x4,
            next_u64x4
        );
        $m!(
            rng_portable_x4,
            do_u64x4_xoshiro256plusplus_portable,
            simd_rand::portable::Xoshiro256PlusPlusX4,
            std::simd::u64x4,
            next_u64x4
        );
        $m!(
            rng_portable_frand_x4,
            do_u64x4_portable_frand,
            simd_rand::portable::FrandX4,
            std::simd::u64x4,
            next_u64x4
        );
        $m!(
            rng_portable_biski_x4,
            do_u64x4_portable_biski,
            simd_rand::portable::Biski64X4,
            std::simd::u64x4,
            next_u64x4
        );
        $m!(
            rng_specific_x4,
            do_u64x4_xoshiro256plus_specific,
            simd_rand::specific::avx2::Xoshiro256PlusX4,
            core::arch::x86_64::__m256i,
            next_m256i
        );
        $m!(
            rng_specific_x4,
            do_u64x4_xoshiro256plusplus_specific,
            simd_rand::specific::avx2::Xoshiro256PlusPlusX4,
            core::arch::x86_64::__m256i,
            next_m256i
        );
        $m!(
            rng_specific_frand_x4,
            do_u64x4_specific_frand,
            simd_rand::specific::avx2::FrandX4,
            core::arch::x86_64::__m256i,
            next_m256i
        );
        $m!(
            rng_specific_biski_x4,
            do_u64x4_specific_biski,
            simd_rand::specific::avx2::Biski64X4,
            core::arch::x86_64::__m256i,
            next_m256i
        );
        $m!(
            rng_specific_x4,
            do_u64x4_shishua_specific,
            Shishua,
            core::arch::x86_64::__m256i,
            next_m256i
        );
        $m!(
            rng_portable_x8,
            do_u64x8_xoshiro256plus_portable,
            simd_rand::portable::Xoshiro256PlusX8,
            std::simd::u64x8,
            next_u64x8
        );
        $m!(
            rng_portable_x8,
            do_u64x8_xoshiro256plusplus_portable,
            simd_rand::portable::Xoshiro256PlusPlusX8,
            std::simd::u64x8,
            next_u64x8
        );
        $m!(
            rng_portable_frand_x8,
            do_u64x8_portable_frand,
            simd_rand::portable::FrandX8,
            std::simd::u64x8,
            next_u64x8
        );
        $m!(
            rng_portable_biski_x8,
            do_u64x8_portable_biski,
            simd_rand::portable::Biski64X8,
            std::simd::u64x8,
            next_u64x8
        );
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512dq",
            target_feature = "avx512vl"
        ))]
        $m!(
            rng_specific_x8,
            do_u64x8_xoshiro256plus_specific,
            simd_rand::specific::avx512::Xoshiro256PlusX8,
            core::arch::x86_64::__m512i,
            next_m512i
        );
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512dq",
            target_feature = "avx512vl"
        ))]
        $m!(
            rng_specific_x8,
            do_u64x8_xoshiro256plusplus_specific,
            simd_rand::specific::avx512::Xoshiro256PlusPlusX8,
            core::arch::x86_64::__m512i,
            next_m512i
        );
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512dq",
            target_feature = "avx512vl"
        ))]
        $m!(
            rng_specific_frand_x8,
            do_u64x8_specific_frand,
            simd_rand::specific::avx512::FrandX8,
            core::arch::x86_64::__m512i,
            next_m512i
        );
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512dq",
            target_feature = "avx512vl"
        ))]
        $m!(
            rng_specific_biski_x8,
            do_u64x8_specific_biski,
            simd_rand::specific::avx512::Biski64X8,
            core::arch::x86_64::__m512i,
            next_m512i
        );
        $m!(
            rng_specific_x4,
            do_f64x4_xoshiro256plus_specific,
            simd_rand::specific::avx2::Xoshiro256PlusX4,
            core::arch::x86_64::__m256d,
            next_m256d
        );
        $m!(
            rng_specific_x4,
            do_f64x4_xoshiro256plusplus_specific,
            simd_rand::specific::avx2::Xoshiro256PlusPlusX4,
            core::arch::x86_64::__m256d,
            next_m256d
        );
        $m!(
            rng_specific_frand_x4,
            do_f64x4_specific_frand,
            simd_rand::specific::avx2::FrandX4,
            core::arch::x86_64::__m256d,
            next_m256d
        );
        $m!(
            rng_specific_biski_x4,
            do_f64x4_specific_biski,
            simd_rand::specific::avx2::Biski64X4,
            core::arch::x86_64::__m256d,
            next_m256d
        );
        $m!(
            rng_specific_x4,
            do_f64x4_shishua_specific,
            Shishua,
            core::arch::x86_64::__m256d,
            next_m256d
        );
        $m!(
            rng_portable_x4,
            do_f64x4_xoshiro256plus_portable,
            simd_rand::portable::Xoshiro256PlusX4,
            std::simd::f64x4,
            next_f64x4
        );
        $m!(
            rng_portable_x4,
            do_f64x4_xoshiro256plusplus_portable,
            simd_rand::portable::Xoshiro256PlusPlusX4,
            std::simd::f64x4,
            next_f64x4
        );
        $m!(
            rng_portable_frand_x4,
            do_f64x4_portable_frand,
            simd_rand::portable::FrandX4,
            std::simd::f64x4,
            next_f64x4
        );
        $m!(
            rng_portable_biski_x4,
            do_f64x4_portable_biski,
            simd_rand::portable::Biski64X4,
            std::simd::f64x4,
            next_f64x4
        );
    };
}

for_each_dasm_case!(define_dasm_wrapper);

fn main() {
    let mut rng_xoshiro256plus_baseline = rand_xoshiro::Xoshiro256Plus::seed_from_u64(0);
    let mut rng_xoshiro256plusplus_baseline = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(0);
    let mut rng_frand = Rand::with_seed(0);
    let mut rng_biski = Biski64Rng::from_seed_for_stream(0, 0, 1);

    black_box(do_u64x4_xoshiro256plus_baseline(&mut rng_xoshiro256plus_baseline));
    black_box(do_u64x8_xoshiro256plus_baseline(&mut rng_xoshiro256plus_baseline));
    black_box(do_u64x4_xoshiro256plusplus_baseline(
        &mut rng_xoshiro256plusplus_baseline,
    ));
    black_box(do_u64x8_xoshiro256plusplus_baseline(
        &mut rng_xoshiro256plusplus_baseline,
    ));
    black_box(do_u64x8_frand_baseline(&mut rng_frand));
    black_box(do_u64x8_biski_baseline(&mut rng_biski));

    macro_rules! call_dasm_rng {
        ($binding:ident, $fn_name:ident, $rng_ty:path, $ret_ty:ty, $method:ident) => {{
            let mut rng = <$rng_ty>::seed_from_u64(0);
            black_box($fn_name(&mut rng));
        }};
    }

    for_each_dasm_case!(call_dasm_rng);
}
