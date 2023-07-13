use std::mem;

use criterion::measurement::Measurement;
use criterion::{criterion_group, criterion_main, Criterion, black_box, Throughput};
use criterion_perf_events::Perf;
use perfcnt::linux::HardwareEventType as Hardware;
use perfcnt::linux::PerfCounterBuilderLinux as Builder;
use rand::rngs::SmallRng;
use rand_core::{RngCore, SeedableRng};
use shishua::Shishua;

const ITERATIONS: usize = 64;

fn work<T: RngCore>(rng: &mut T) {
    for _ in 0..64 {
        black_box(rng.next_u64());
    }
}

fn bench<M: Measurement, const T: u8>(c: &mut Criterion<M>) {
    
    let mut group = c.benchmark_group("RNG");

    group.throughput(Throughput::Bytes((ITERATIONS * mem::size_of::<u64>()) as u64));

    let suffix = match T {
        Type::TIME => "Time",
        Type::INST => "Instructions",
        Type::CYCLES => "Cycles",
        _ => unreachable!(),
    };

    let shishua_name = format!("Shishua u64 - {suffix}");
    let small_rng_name = format!("SmallRng u64 - {suffix}");

    group.bench_function(shishua_name, |b| {
        let mut rng: Shishua = Shishua::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);

        b.iter(|| work(&mut rng))
    });
    group.bench_function(small_rng_name, |b| {
        let mut rng: SmallRng = SmallRng::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);

        b.iter(|| work(&mut rng))
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
criterion_group!(
    name = instructions;
    config = Criterion::default().with_measurement(Perf::new(Builder::from_hardware_event(Hardware::Instructions)));
    targets = bench::<_, { Type::INST }>
);
criterion_group!(
    name = cycles;
    config = Criterion::default().with_measurement(Perf::new(Builder::from_hardware_event(Hardware::RefCPUCycles)));
    targets = bench::<_, { Type::CYCLES }>
);
criterion_main!(time, instructions, cycles);
