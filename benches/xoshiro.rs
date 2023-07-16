#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

use std::mem;

use criterion::measurement::Measurement;
use criterion::{criterion_group, criterion_main, Criterion, black_box, Throughput};
use rand::Rng;
use criterion_perf_events::Perf;
use perfcnt::linux::HardwareEventType as Hardware;
use perfcnt::linux::PerfCounterBuilderLinux as Builder;
use rand_core::{SeedableRng, RngCore};
use rand_xoshiro::Xoshiro256PlusPlus;
use shishua::{U64x4, F64x4};
use shishua::xoshiro256plusplus::Xoshiro256PlusPlusX4;

const ITERATIONS: usize = 16;

fn do_xoshiro_u64(rng: &mut Xoshiro256PlusPlus, data: &mut U64x4) {
    for _ in 0..ITERATIONS {
        let data = black_box(&mut *data);
        data[0] = rng.next_u64();
        data[1] = rng.next_u64();
        data[2] = rng.next_u64();
        data[3] = rng.next_u64();
    }
}
// This is 10x faster on my machine... somethings not right
fn do_xoshiro_x4_u64(rng: &mut Xoshiro256PlusPlusX4, data: &mut U64x4) {
    for _ in 0..ITERATIONS {
        rng.next_u64x4(black_box(data));
    }
}

fn do_xoshiro_f64(rng: &mut Xoshiro256PlusPlus, data: &mut F64x4) {
    for _ in 0..ITERATIONS {
        let data = black_box(&mut *data);
        data[0] = rng.gen_range(0.0..1.0);
        data[1] = rng.gen_range(0.0..1.0);
        data[2] = rng.gen_range(0.0..1.0);
        data[3] = rng.gen_range(0.0..1.0);
    }
}
// fn do_xoshiro_x4_f64(rng: &mut Xoshiro256PlusPlusX4, data: &mut F64x4) {
//     for _ in 0..ITERATIONS {
//         rng.next_f64x4(black_box(data));
//     }
// }
fn do_xoshiro_x4_f64(rng: &mut Xoshiro256PlusPlusX4, data: &mut F64x4) {
    for _ in 0..ITERATIONS {
        rng.next_f64x4(black_box(data));
    }
}

fn bench<M: Measurement, const T: u8>(c: &mut Criterion<M>) {
    
    let mut group = c.benchmark_group("xoshiro");

    group.throughput(Throughput::Bytes((ITERATIONS * mem::size_of::<U64x4>()) as u64));

    let suffix = match T {
        Type::TIME => "Time",
        Type::INST => "Instructions",
        Type::CYCLES => "Cycles",
        _ => unreachable!(),
    };

    let xoshiro_u64_name = format!("Xoshiro256PlusPlus u64x4 - {suffix}");
    let xoshiro_x4_u64_name = format!("Xoshiro256PlusPlusX4 u64x4 - {suffix}");
    let xoshiro_f64_name = format!("Xoshiro256PlusPlus f64x4 - {suffix}");
    // let xoshiro_x4_f64_name = format!("Xoshiro256PlusPlusX4 f64x4 - {suffix}");
    let xoshiro_x4_f64_name = format!("Xoshiro256PlusPlusX4 f64x4 - {suffix}");

    // group.bench_function(xoshiro_u64_name, |b| {
    //     let mut rng: Xoshiro256PlusPlus = Xoshiro256PlusPlus::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
    //     let mut data: U64x4 = Default::default();

    //     b.iter(|| do_xoshiro_u64(&mut rng, black_box(&mut data)))
    // });
    // group.bench_function(xoshiro_x4_u64_name, |b| {
    //     let mut rng: Xoshiro256PlusPlusX4 = Xoshiro256PlusPlusX4::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
    //     let mut data: U64x4 = Default::default();

    //     b.iter(|| do_xoshiro_x4_u64(&mut rng, black_box(&mut data)))
    // });

    group.bench_function(xoshiro_f64_name, |b| {
        let mut rng: Xoshiro256PlusPlus = Xoshiro256PlusPlus::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
        let mut data: F64x4 = Default::default();

        b.iter(|| do_xoshiro_f64(&mut rng, black_box(&mut data)))
    });
    // group.bench_function(xoshiro_x4_f64_name, |b| {
    //     let mut rng: Xoshiro256PlusPlusX4 = Xoshiro256PlusPlusX4::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
    //     let mut data: F64x4 = Default::default();

    //     b.iter(|| do_xoshiro_x4_f64(&mut rng, black_box(&mut data)))
    // });
    group.bench_function(xoshiro_x4_f64_name, |b| {
        let mut rng: Xoshiro256PlusPlusX4 = Xoshiro256PlusPlusX4::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
        let mut data: F64x4 = Default::default();

        b.iter(|| do_xoshiro_x4_f64(&mut rng, black_box(&mut data)))
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
