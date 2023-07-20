#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

use std::mem;

use criterion::measurement::Measurement;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use criterion_perf_events::Perf;
use perfcnt::linux::HardwareEventType as Hardware;
use perfcnt::linux::PerfCounterBuilderLinux as Builder;
use rand::rngs::SmallRng;
use rand::Rng;
use rand_core::{RngCore, SeedableRng};
use simd_rand::specific::avx2::*;

const ITERATIONS: usize = 16;

#[inline(always)]
fn do_shishua_u64(rng: &mut Shishua, data: &mut U64x4) {
    for _ in 0..ITERATIONS {
        rng.next_u64x4(black_box(data));
    }
}
#[inline(always)]
fn do_small_rng_u64(rng: &mut SmallRng, data: &mut U64x4) {
    for _ in 0..ITERATIONS {
        let data = black_box(&mut *data);
        data[0] = rng.next_u64();
        data[1] = rng.next_u64();
        data[2] = rng.next_u64();
        data[3] = rng.next_u64();
    }
}

#[inline(always)]
fn do_shishua_f64(rng: &mut Shishua, data: &mut F64x4) {
    for _ in 0..ITERATIONS {
        rng.next_f64x4(black_box(data));
    }
}
#[inline(always)]
fn do_small_rng_f64(rng: &mut SmallRng, data: &mut F64x4) {
    for _ in 0..ITERATIONS {
        let data = black_box(&mut *data);
        data[0] = rng.gen_range(0.0..1.0);
        data[1] = rng.gen_range(0.0..1.0);
        data[2] = rng.gen_range(0.0..1.0);
        data[3] = rng.gen_range(0.0..1.0);
    }
}

fn bench<M: Measurement, const T: u8>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("shishua");

    group.throughput(Throughput::Bytes((ITERATIONS * mem::size_of::<u64>()) as u64));

    let suffix = match T {
        Type::TIME => "Time",
        Type::INST => "Instructions",
        Type::CYCLES => "Cycles",
        _ => unreachable!(),
    };

    let shishua_u64_name = format!("Shishua u64x4 - {suffix}");
    let small_rng_u64_name = format!("SmallRng u64x4 - {suffix}");
    let shishua_f64_name = format!("Shishua f64x4 - {suffix}");
    let small_rng_f64_name = format!("SmallRng f64x4 - {suffix}");

    group.bench_function(shishua_u64_name, |b| {
        let mut rng: Shishua = Shishua::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
        let mut data: U64x4 = Default::default();

        b.iter(|| do_shishua_u64(&mut rng, &mut data))
    });
    group.bench_function(small_rng_u64_name, |b| {
        let mut rng: SmallRng = SmallRng::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
        let mut data: U64x4 = Default::default();

        b.iter(|| do_small_rng_u64(&mut rng, &mut data))
    });
    group.bench_function(shishua_f64_name, |b| {
        let mut rng: Shishua = Shishua::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
        let mut data: F64x4 = Default::default();

        b.iter(|| do_shishua_f64(&mut rng, &mut data))
    });
    group.bench_function(small_rng_f64_name, |b| {
        let mut rng: SmallRng = SmallRng::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
        let mut data: F64x4 = Default::default();

        b.iter(|| do_small_rng_f64(&mut rng, &mut data))
    });

    group.finish();
}

#[non_exhaustive]
struct Type;

impl Type {
    pub const TIME: u8 = 1;
    pub const INST: u8 = 2;
    pub const CYCLES: u8 = 3;
}

criterion_group!(
    name = time;
    config = Criterion::default();
    targets = bench::<_, { Type::TIME }>
);
// criterion_group!(
//     name = instructions;
//     config = Criterion::default().with_measurement(Perf::new(Builder::from_hardware_event(Hardware::Instructions)));
//     targets = bench::<_, { Type::INST }>
// );
// criterion_group!(
//     name = cycles;
//     config = Criterion::default().with_measurement(Perf::new(Builder::from_hardware_event(Hardware::RefCPUCycles)));
//     targets = bench::<_, { Type::CYCLES }>
// );
criterion_main!(time);
