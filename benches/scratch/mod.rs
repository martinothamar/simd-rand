use std::arch::x86_64::*;
use std::{mem, simd::u64x8};

use criterion::{black_box, measurement::Measurement, BenchmarkId, Criterion, Throughput};
use rand_core::{SeedableRng, RngCore};
use rand_xoshiro::Xoshiro256Plus;
use simd_rand::portable;
use simd_rand::portable::SimdRandX8 as PortableSimdRand;
use simd_rand::specific;
use simd_rand::specific::avx512::SimdRand as SpecificSimdRand;

#[inline(always)]
fn do_u64_baseline(rng: &mut Xoshiro256Plus) -> u64x8 {
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

#[inline(always)]
fn do_u64_portable<RNG: PortableSimdRand>(rng: &mut RNG) -> u64x8 {
    rng.next_u64x8()
}

#[inline(always)]
fn do_u64_specific<RNG: SpecificSimdRand>(rng: &mut RNG) -> __m512i {
    rng.next_m512i()
}

pub fn add_benchmarks<M: Measurement>(c: &mut Criterion<M>, suffix: &str) {
    let mut group = c.benchmark_group("Scratch");

    group.throughput(Throughput::Bytes((1 * mem::size_of::<u64x8>()) as u64));

    let name = BenchmarkId::new(format!("Baseline/Xoshiro256+/{suffix}"), "1");
    group.bench_with_input(name, &1, |b, _| {
        let mut rng = Xoshiro256Plus::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);

        b.iter(|| do_u64_baseline(&mut rng))
    });
    let name = BenchmarkId::new(format!("Portable/Xoshiro256+/{suffix}"), "1");
    group.bench_with_input(name, &1, |b, _| {
        let mut rng = portable::Xoshiro256PlusX8::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);

        b.iter(|| do_u64_portable(&mut rng))
    });
    let name = BenchmarkId::new(format!("Specific/Xoshiro256+/{suffix}"), "1");
    group.bench_with_input(name, &1, |b, _| {
        let mut rng = specific::avx512::Xoshiro256PlusX8::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);

        b.iter(|| do_u64_specific(&mut rng))
    });

    group.finish();
}
