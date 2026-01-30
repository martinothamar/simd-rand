use std::arch::x86_64::*;
use std::{mem, simd::u64x8};

use criterion::{black_box, measurement::Measurement, BenchmarkId, Criterion, Throughput};
use rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use simd_rand::portable;
use simd_rand::portable::*;

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
fn do_u64_portable_x4<RNG: SimdRandX4>(rng: &mut RNG) -> u64x8 {
    let a = rng.next_u64x4();
    let b = rng.next_u64x4();
    u64x8::from_array([a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]])
}

#[inline(always)]
fn do_u64_portable_x8<RNG: SimdRandX8>(rng: &mut RNG) -> u64x8 {
    rng.next_u64x8()
}

pub fn add_benchmarks<M: Measurement>(c: &mut Criterion<M>, suffix: &str) {
    let mut group = c.benchmark_group("Scratch");

    group.throughput(Throughput::Bytes(mem::size_of::<u64x8>() as u64));

    let name = BenchmarkId::new(format!("Baseline/Xoshiro256+/{suffix}"), "1");
    group.bench_with_input(name, &1, |b, _| {
        let mut rng = Xoshiro256Plus::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);

        b.iter(|| do_u64_baseline(&mut rng))
    });
    let name = BenchmarkId::new(format!("Portable/Xoshiro256+X4/{suffix}"), "1");
    group.bench_with_input(name, &1, |b, _| {
        let mut rng = portable::Xoshiro256PlusX4::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);

        b.iter(|| do_u64_portable_x4(&mut rng))
    });
    let name = BenchmarkId::new(format!("Portable/Xoshiro256+X8/{suffix}"), "1");
    group.bench_with_input(name, &1, |b, _| {
        let mut rng = portable::Xoshiro256PlusX8::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);

        b.iter(|| do_u64_portable_x8(&mut rng))
    });

    group.finish();
}
