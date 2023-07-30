use std::{mem, simd::u64x4, simd::u64x8};

use criterion::{black_box, measurement::Measurement, BenchmarkId, Criterion, Throughput};
use rand_core::SeedableRng;
use simd_rand::portable::{SimdRandX4, SimdRandX8, Xoshiro256PlusX4, Xoshiro256PlusX8};

pub fn add_benchmarks<M: Measurement, const ITERATIONS: usize>(c: &mut Criterion<M>, suffix: &str) {
    let group_prefix = "Portable";
    add_u64x4_benchmarks::<_, ITERATIONS>(c, group_prefix, suffix);
    add_u64x8_benchmarks::<_, ITERATIONS>(c, group_prefix, suffix);
}

fn add_u64x4_benchmarks<M: Measurement, const ITERATIONS: usize>(
    c: &mut Criterion<M>,
    group_prefix: &str,
    suffix: &str,
) {
    let mut group = c.benchmark_group(format!("{group_prefix}/u64x4"));

    let iterations: Vec<_> = (0..4).map(|v| (v + 1) * ITERATIONS).collect();

    for iterations in iterations {
        group.throughput(Throughput::Bytes((iterations * mem::size_of::<u64x4>()) as u64));

        #[inline(always)]
        fn execute<RNG: SimdRandX4>(rng: &mut RNG, data: &mut u64x4, i: usize) {
            for _ in 0..i {
                *black_box(&mut *data) = rng.next_u64x4();
            }
        }

        let name = BenchmarkId::new(format!("Xoshiro256+/{suffix}"), iterations);
        group.bench_with_input(name, &iterations, |b, i| {
            let mut rng = Xoshiro256PlusX4::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
            let mut data = Default::default();

            b.iter(|| execute(&mut rng, black_box(&mut data), black_box(*i)))
        });
    }

    group.finish();
}

fn add_u64x8_benchmarks<M: Measurement, const ITERATIONS: usize>(
    c: &mut Criterion<M>,
    group_prefix: &str,
    suffix: &str,
) {
    let mut group = c.benchmark_group(format!("{group_prefix}/u64x8"));

    let iterations: Vec<_> = (0..4).map(|v| (v + 1) * ITERATIONS).collect();

    for iterations in iterations {
        group.throughput(Throughput::Bytes((iterations * mem::size_of::<u64x8>()) as u64));

        #[inline(always)]
        fn execute<RNG: SimdRandX8>(rng: &mut RNG, data: &mut u64x8, i: usize) {
            for _ in 0..i {
                *black_box(&mut *data) = rng.next_u64x8();
            }
        }

        let name = BenchmarkId::new(format!("Xoshiro256+/{suffix}"), iterations);
        group.bench_with_input(name, &iterations, |b, i| {
            let mut rng = Xoshiro256PlusX8::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
            let mut data = Default::default();

            b.iter(|| execute(&mut rng, black_box(&mut data), black_box(*i)))
        });
    }

    group.finish();
}
