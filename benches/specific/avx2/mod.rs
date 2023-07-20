use std::{arch::x86_64::*, mem};

use criterion::{black_box, measurement::Measurement, BenchmarkId, Criterion, Throughput};
use rand_core::SeedableRng;
use simd_rand::specific::avx2::*;

type Shishua = simd_rand::specific::avx2::Shishua<DEFAULT_BUFFER_SIZE>;

pub fn add_benchmarks<M: Measurement, const ITERATIONS: usize>(c: &mut Criterion<M>, suffix: &str) {
    let group_prefix = "AVX2";
    add_m256i_benchmarks::<_, ITERATIONS>(c, group_prefix, suffix);
    add_m256d_benchmarks::<_, ITERATIONS>(c, group_prefix, suffix);
}

fn add_m256i_benchmarks<M: Measurement, const ITERATIONS: usize>(
    c: &mut Criterion<M>,
    group_prefix: &str,
    suffix: &str,
) {
    let mut group = c.benchmark_group(format!("{group_prefix}/m256i"));

    let iterations: Vec<_> = (0..4).map(|v| (v + 1) * ITERATIONS).collect();

    for iterations in iterations {
        group.throughput(Throughput::Bytes((iterations * mem::size_of::<__m256i>()) as u64));

        #[inline(always)]
        fn execute<RNG: SimdPrng>(rng: &mut RNG, data: &mut __m256i, i: usize) {
            for _ in 0..i {
                rng.next_m256i(black_box(data));
            }
        }

        let name = BenchmarkId::new(format!("Shishua/{suffix}"), iterations);
        group.bench_with_input(name, &iterations, |b, i| unsafe {
            let mut rng = Shishua::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
            let mut data: __m256i = _mm256_setzero_si256();

            b.iter(|| execute(&mut rng, black_box(&mut data), black_box(*i)))
        });

        let name = BenchmarkId::new(format!("Xoshiro256++/{suffix}"), iterations);
        group.bench_with_input(name, &iterations, |b, i| unsafe {
            let mut rng = Xoshiro256PlusPlusX4::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
            let mut data: __m256i = _mm256_setzero_si256();

            b.iter(|| execute(&mut rng, black_box(&mut data), black_box(*i)))
        });

        let name = BenchmarkId::new(format!("Xoshiro256+/{suffix}"), iterations);
        group.bench_with_input(name, &iterations, |b, i| unsafe {
            let mut rng = Xoshiro256PlusX4::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
            let mut data: __m256i = _mm256_setzero_si256();

            b.iter(|| execute(&mut rng, black_box(&mut data), black_box(*i)))
        });
    }

    group.finish();
}

fn add_m256d_benchmarks<M: Measurement, const ITERATIONS: usize>(
    c: &mut Criterion<M>,
    group_prefix: &str,
    suffix: &str,
) {
    let mut group = c.benchmark_group(format!("{group_prefix}/m256d"));

    let iterations: Vec<_> = (0..4).map(|v| (v + 1) * ITERATIONS).collect();

    for iterations in iterations {
        group.throughput(Throughput::Bytes((iterations * mem::size_of::<__m256d>()) as u64));

        #[inline(always)]
        fn execute<RNG: SimdPrng>(rng: &mut RNG, data: &mut __m256d, i: usize) {
            for _ in 0..i {
                rng.next_m256d(black_box(data));
            }
        }

        let name = BenchmarkId::new(format!("Shishua/{suffix}"), iterations);
        group.bench_with_input(name, &iterations, |b, i| unsafe {
            let mut rng = Shishua::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
            let mut data: __m256d = _mm256_setzero_pd();

            b.iter(|| execute(&mut rng, black_box(&mut data), black_box(*i)))
        });

        let name = BenchmarkId::new(format!("Xoshiro256++/{suffix}"), iterations);
        group.bench_with_input(name, &iterations, |b, i| unsafe {
            let mut rng = Xoshiro256PlusPlusX4::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
            let mut data: __m256d = _mm256_setzero_pd();

            b.iter(|| execute(&mut rng, black_box(&mut data), black_box(*i)))
        });

        let name = BenchmarkId::new(format!("Xoshiro256+/{suffix}"), iterations);
        group.bench_with_input(name, &iterations, |b, i| unsafe {
            let mut rng = Xoshiro256PlusX4::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
            let mut data: __m256d = _mm256_setzero_pd();

            b.iter(|| execute(&mut rng, black_box(&mut data), black_box(*i)))
        });
    }

    group.finish();
}
