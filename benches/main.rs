#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![feature(portable_simd)]

use std::arch::x86_64::*;
use std::mem;

use criterion::measurement::Measurement;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use rand_core::SeedableRng;
use std::hint::black_box;

mod portable;
mod scratch;
mod specific;
mod top;

const ITERATIONS: usize = 8;

fn bench<M: Measurement, const T: u8>(c: &mut Criterion<M>) {
    let suffix = match T {
        Type::TIME => "Time",
        Type::INST => "Instructions",
        Type::CYCLES => "Cycles",
        _ => unreachable!(),
    };

    crate::scratch::add_benchmarks(c, suffix);

    crate::portable::add_benchmarks::<_, ITERATIONS>(c, suffix);

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    crate::specific::avx2::add_benchmarks::<_, ITERATIONS>(c, suffix);

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512dq",
        target_feature = "avx512vl"
    ))]
    crate::specific::avx512::add_benchmarks::<_, ITERATIONS>(c, suffix);

    crate::top::add_top_benchmark::<_, ITERATIONS>(c);
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
