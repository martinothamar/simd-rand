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

pub fn add_top_benchmark<M: Measurement, const ITERATIONS: usize>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Top");

    group.throughput(Throughput::ElementsAndBytes {
        elements: (ITERATIONS * 8) as u64,
        bytes: (ITERATIONS * mem::size_of::<u64x8>()) as u64,
    });
    group.noise_threshold(0.03);

    let mut rng = rand::rng();
    let init = rng.next_u64();
    let init_i = init as i64;

    group.bench_function("rand/Xoshiro256+", |b| {
        let mut rng = Xoshiro256Plus::seed_from_u64(init);

        b.iter(|| {
            let mut data = u64x8::splat(init);

            for _ in 0..ITERATIONS {
                for i in 0..8 {
                    data[i] += rng.next_u64();
                }
            }

            data
        });
    });

    group.bench_function("frand", |b| {
        let mut rng = Rand::with_seed(init);

        b.iter(|| {
            let mut data = u64x8::splat(init);

            for _ in 0..ITERATIONS {
                for i in 0..8 {
                    data[i] += rng.r#gen::<u64>();
                }
            }

            data
        });
    });

    group.bench_function("simd_rand/Portable/Xoshiro256+X8", |b| {
        let mut rng = Xoshiro256PlusX8::seed_from_u64(init);

        b.iter(|| {
            let mut data = u64x8::splat(init);

            for _ in 0..ITERATIONS {
                data += rng.next_u64x8();
            }

            data
        });
    });

    group.bench_function("simd_rand/Portable/FrandX8", |b| {
        let mut rng = FrandX8::seed_from_u64(init);

        b.iter(|| {
            let mut data = u64x8::splat(init);

            for _ in 0..ITERATIONS {
                data += rng.next_u64x8();
            }

            data
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
        let mut rng = specific::avx512::Xoshiro256PlusX8::seed_from_u64(init);

        b.iter(|| {
            let mut data = _mm512_set1_epi64(init_i);

            for _ in 0..ITERATIONS {
                data = _mm512_add_epi64(data, specific::avx512::SimdRand::next_m512i(&mut rng));
            }

            data
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
        let mut rng = specific::avx512::FrandX8::seed_from_u64(init);

        b.iter(|| {
            let mut data = _mm512_set1_epi64(init_i);

            for _ in 0..ITERATIONS {
                data = _mm512_add_epi64(data, specific::avx512::SimdRand::next_m512i(&mut rng));
            }

            data
        });
    });

    group.finish();
}
