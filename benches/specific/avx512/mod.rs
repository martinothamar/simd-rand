use std::{arch::x86_64::*, mem};

use criterion::{BenchmarkId, Criterion, Throughput, measurement::Measurement};
use rand_core::SeedableRng;
use simd_rand::specific::avx512::*;
use std::hint::black_box;

pub fn add_benchmarks<M: Measurement, const ITERATIONS: usize>(c: &mut Criterion<M>, suffix: &str) {
    let group_prefix = "AVX512";
    add_m512i_benchmarks::<_, ITERATIONS>(c, group_prefix, suffix);
    add_m512d_benchmarks::<_, ITERATIONS>(c, group_prefix, suffix);
}

fn add_m512i_benchmarks<M: Measurement, const ITERATIONS: usize>(
    c: &mut Criterion<M>,
    group_prefix: &str,
    suffix: &str,
) {
    let mut group = c.benchmark_group(format!("{group_prefix}/m512i"));

    let iterations: Vec<_> = (0..4).map(|v| (v + 1) * ITERATIONS).collect();

    for iterations in iterations {
        group.throughput(Throughput::Bytes((iterations * mem::size_of::<__m512i>()) as u64));

        #[inline(always)]
        fn execute<RNG: SimdRand>(rng: &mut RNG, data: &mut __m512i, i: usize) {
            for _ in 0..i {
                *black_box(&mut *data) = rng.next_m512i();
            }
        }

        let name = BenchmarkId::new(format!("Xoshiro256++/{suffix}"), iterations);
        group.bench_with_input(name, &iterations, |b, i| unsafe {
            let mut rng = Xoshiro256PlusPlusX8::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
            let mut data: __m512i = _mm512_setzero_si512();

            b.iter(|| execute(&mut rng, black_box(&mut data), black_box(*i)))
        });

        let name = BenchmarkId::new(format!("Xoshiro256+/{suffix}"), iterations);
        group.bench_with_input(name, &iterations, |b, i| unsafe {
            let mut rng = Xoshiro256PlusX8::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
            let mut data: __m512i = _mm512_setzero_si512();

            b.iter(|| execute(&mut rng, black_box(&mut data), black_box(*i)))
        });
    }

    group.finish();
}

fn add_m512d_benchmarks<M: Measurement, const ITERATIONS: usize>(
    c: &mut Criterion<M>,
    group_prefix: &str,
    suffix: &str,
) {
    let mut group = c.benchmark_group(format!("{group_prefix}/m512d"));

    let iterations: Vec<_> = (0..4).map(|v| (v + 1) * ITERATIONS).collect();

    for iterations in iterations {
        group.throughput(Throughput::Bytes((iterations * mem::size_of::<__m512d>()) as u64));

        #[inline(always)]
        fn execute<RNG: SimdRand>(rng: &mut RNG, data: &mut __m512d, i: usize) {
            for _ in 0..i {
                *black_box(&mut *data) = rng.next_m512d();
            }
        }

        let name = BenchmarkId::new(format!("Xoshiro256++/{suffix}"), iterations);
        group.bench_with_input(name, &iterations, |b, i| unsafe {
            let mut rng = Xoshiro256PlusPlusX8::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
            let mut data: __m512d = _mm512_setzero_pd();

            b.iter(|| execute(&mut rng, black_box(&mut data), black_box(*i)))
        });

        let name = BenchmarkId::new(format!("Xoshiro256+/{suffix}"), iterations);
        group.bench_with_input(name, &iterations, |b, i| unsafe {
            let mut rng = Xoshiro256PlusX8::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
            let mut data: __m512d = _mm512_setzero_pd();

            b.iter(|| execute(&mut rng, black_box(&mut data), black_box(*i)))
        });
    }

    group.finish();
}
