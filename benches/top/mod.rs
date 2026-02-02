use std::{mem, simd::u64x8};

use criterion::{Criterion, Throughput, measurement::Measurement};
use frand::Rand;
use rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use simd_rand::portable::*;
#[cfg(feature = "specific")]
use simd_rand::specific;
#[cfg(all(feature = "specific", target_arch = "x86_64"))]
use std::arch::x86_64::*;
use std::hint::black_box;

const SEED: u64 = 0x0DDB1A5E5BAD5EEDu64;

#[inline(always)]
fn execute_rand<RNG: RngCore>(rng: &mut RNG, data: &mut u64x8) {
    for i in 0..8 {
        data[i] = rng.next_u64();
    }
}

#[inline(always)]
fn execute_vectorized_portable<RNG: SimdRandX8>(rng: &mut RNG, data: &mut u64x8) {
    *data = rng.next_u64x8();
}

#[cfg(all(
    feature = "specific",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl"
))]
#[inline(always)]
fn execute_vectorized_specific<RNG: specific::avx512::SimdRand>(rng: &mut RNG, data: &mut __m512i) {
    *data = rng.next_m512i();
}

pub fn add_top_benchmark<M: Measurement, const ITERATIONS: usize>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Top");

    group.throughput(Throughput::ElementsAndBytes {
        elements: 8,
        bytes: mem::size_of::<u64x8>() as u64,
    });
    group.noise_threshold(0.03);

    group.bench_function("rand/Xoshiro256+", |b| {
        let mut rng = Xoshiro256Plus::seed_from_u64(SEED);
        let mut data = u64x8::default();

        b.iter(|| {
            execute_rand(&mut rng, &mut data);
            black_box(&data);
        });
    });

    group.bench_function("frand", |b| {
        let mut rng = Rand::with_seed(SEED);
        let mut data = u64x8::default();

        b.iter(|| {
            for i in 0..8 {
                data[i] = black_box(rng.r#gen::<u64>());
            }
            black_box(&data);
        });
    });

    group.bench_function("simd_rand/Portable/Xoshiro256+X8", |b| {
        let mut rng = Xoshiro256PlusX8::seed_from_u64(SEED);
        let mut data = u64x8::default();

        b.iter(|| {
            execute_vectorized_portable(&mut rng, &mut data);
            black_box(&data);
        });
    });

    group.bench_function("simd_rand/Portable/FrandX8", |b| {
        let mut rng = FrandX8::seed_from_u64(SEED);
        let mut data = u64x8::default();

        b.iter(|| {
            execute_vectorized_portable(&mut rng, &mut data);
            black_box(&data);
        });
    });

    #[cfg(all(
        feature = "specific",
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512dq",
        target_feature = "avx512vl"
    ))]
    group.bench_function("simd_rand/Specific/Xoshiro256+X8", |b| unsafe {
        let mut rng = specific::avx512::Xoshiro256PlusX8::seed_from_u64(SEED);
        let mut data: __m512i = _mm512_setzero_si512();

        b.iter(|| {
            execute_vectorized_specific(&mut rng, &mut data);
            black_box(&data);
        });
    });

    #[cfg(all(
        feature = "specific",
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512dq",
        target_feature = "avx512vl"
    ))]
    group.bench_function("simd_rand/Specific/FrandX8", |b| unsafe {
        let mut rng = specific::avx512::FrandX8::seed_from_u64(SEED);
        let mut data: __m512i = _mm512_setzero_si512();

        b.iter(|| {
            execute_vectorized_specific(&mut rng, &mut data);
            black_box(&data);
        });
    });

    group.finish();
}
