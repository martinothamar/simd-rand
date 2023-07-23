#![feature(stdsimd)]
#![feature(portable_simd)]

use std::arch::x86_64::*;
use std::simd::f64x4;
use std::simd::u64x4;

use criterion::black_box;
use rand::{
    distributions::uniform::{UniformFloat, UniformSampler},
    rngs::SmallRng,
};
use rand_core::SeedableRng;
use simd_rand::portable;
use simd_rand::portable::SimdRand as PortableSimdRand;
use simd_rand::specific;
use simd_rand::specific::avx2::SimdRand as SpecificSimdRand;

use packed_simd_2::f64x4 as ps_f64x4;

/// This is a small binary meant to aid in analyzing generated code
/// For example to see differences between portable and specific code,
/// and simd_rand and rand code

#[inline(never)]
fn do_u64_portable<RNG: PortableSimdRand>(rng: &mut RNG) -> u64x4 {
    rng.next_u64x4()
}

#[inline(never)]
fn do_u64_specific<RNG: SpecificSimdRand>(rng: &mut RNG) -> __m256i {
    rng.next_m256i()
}

#[inline(never)]
fn do_f64_specific<RNG: SpecificSimdRand>(rng: &mut RNG) -> __m256d {
    rng.next_m256d()
}

#[inline(never)]
fn do_f64_portable<RNG: PortableSimdRand>(rng: &mut RNG) -> f64x4 {
    rng.next_f64x4()
}

#[inline(never)]
fn do_f64_rand(dist: &mut UniformFloat<ps_f64x4>, rng: &mut SmallRng) -> ps_f64x4 {
    dist.sample(rng)
}

fn main() {
    let mut rng1 = portable::Xoshiro256PlusX4::seed_from_u64(0);
    let mut rng2 = specific::avx2::Xoshiro256PlusX4::seed_from_u64(0);

    let mut rand_rng = SmallRng::seed_from_u64(0);
    let mut rand_dist = UniformFloat::<ps_f64x4>::new::<ps_f64x4, ps_f64x4>(ps_f64x4::splat(0.), ps_f64x4::splat(1.));

    black_box(do_u64_portable(&mut rng1));
    black_box(do_u64_specific(&mut rng2));
    black_box(do_f64_specific(&mut rng2));
    black_box(do_f64_portable(&mut rng1));

    black_box(do_f64_rand(&mut rand_dist, &mut rand_rng));
}
