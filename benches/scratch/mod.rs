use std::arch::x86_64::*;
use std::{mem, simd::u64x4};

use criterion::{black_box, measurement::Measurement, BenchmarkId, Criterion, Throughput};
use rand_core::{SeedableRng, RngCore};
use rand_xoshiro::Xoshiro256Plus;
use simd_rand::portable;
use simd_rand::portable::SimdRand as PortableSimdRand;
use simd_rand::specific;
use simd_rand::specific::avx2::SimdRand as SpecificSimdRand;

#[inline(always)]
fn do_u64_baseline(rng: &mut Xoshiro256Plus) -> u64x4 {
    u64x4::from_array([
        rng.next_u64(),
        rng.next_u64(),
        rng.next_u64(),
        rng.next_u64(),
    ])
}

#[inline(always)]
fn do_u64_portable<RNG: PortableSimdRand>(rng: &mut RNG) -> u64x4 {
    rng.next_u64x4()
}

#[inline(always)]
fn do_u64_specific<RNG: SpecificSimdRand>(rng: &mut RNG) -> __m256i {
    rng.next_m256i()
}

pub fn add_benchmarks<M: Measurement>(c: &mut Criterion<M>, suffix: &str) {
    let mut group = c.benchmark_group("Scratch");

    group.throughput(Throughput::Bytes((1 * mem::size_of::<u64x4>()) as u64));

    let name = BenchmarkId::new(format!("Baseline/Xoshiro256+/{suffix}"), "1");
    group.bench_with_input(name, &1, |b, _| {
        let mut rng = Xoshiro256Plus::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);

        b.iter(|| do_u64_baseline(&mut rng))
    });
    let name = BenchmarkId::new(format!("Portable/Xoshiro256+/{suffix}"), "1");
    group.bench_with_input(name, &1, |b, _| {
        let mut rng = portable::Xoshiro256PlusX4::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);

        b.iter(|| do_u64_portable(&mut rng))
    });
    let name = BenchmarkId::new(format!("Specific/Xoshiro256+/{suffix}"), "1");
    group.bench_with_input(name, &1, |b, _| {
        let mut rng = specific::avx2::Xoshiro256PlusX4::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);

        b.iter(|| do_u64_specific(&mut rng))
    });

    group.finish();
}
