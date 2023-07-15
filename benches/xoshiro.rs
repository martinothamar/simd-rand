use std::mem;

use criterion::measurement::Measurement;
use criterion::{criterion_group, criterion_main, Criterion, black_box, Throughput};
// use criterion_perf_events::Perf;
// use perfcnt::linux::HardwareEventType as Hardware;
// use perfcnt::linux::PerfCounterBuilderLinux as Builder;
use rand_core::{SeedableRng, RngCore};
use shishua::U64x4;
use shishua::xoshiro256plusplus::Xoshiro256PlusPlusX4;
use rand::rngs::SmallRng;

const ITERATIONS: usize = 16;

fn do_small_rng(rng: &mut SmallRng, data: &mut U64x4) {
    for _ in 0..ITERATIONS {
        for i in 0..4 {
            black_box(data.0)[i] = rng.next_u64();
        }
    }
}
// This is 10x faster on my machine... somethings not right
fn do_xoshiro_x4(rng: &mut Xoshiro256PlusPlusX4, data: &mut U64x4) {
    for _ in 0..ITERATIONS {
        rng.next_u64x4(black_box(data));
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

    let small_rng_name = format!("SmallRng u64x4 - {suffix}");
    let xoshiro_vec_name = format!("Xoshiro256PlusPlusX4 u64x4 - {suffix}");

    group.bench_function(small_rng_name, |b| {
        let mut rng: SmallRng = SmallRng::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
        let mut data: U64x4 = Default::default();

        b.iter(|| do_small_rng(&mut rng, black_box(&mut data)))
    });
    group.bench_function(xoshiro_vec_name, |b| {
        let mut rng: Xoshiro256PlusPlusX4 = Xoshiro256PlusPlusX4::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
        let mut data: U64x4 = Default::default();

        b.iter(|| do_xoshiro_x4(&mut rng, black_box(&mut data)))
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
