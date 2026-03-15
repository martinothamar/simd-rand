use std::{mem, simd::u64x8};

use criterion::{BenchmarkGroup, Criterion, Throughput, measurement::Measurement};
use frand::Rand;
use rand_core::{RngCore, SeedableRng};
use simd_rand::portable::SimdRandX8;
#[cfg(feature = "specific")]
use simd_rand::specific;
#[cfg(all(feature = "specific", target_arch = "x86_64"))]
use std::arch::x86_64::*;

macro_rules! for_each_top_scalar_case {
    ($m:ident) => {
        $m!(bench_scalar_rngcore, "rand/Xoshiro256+", |seed| {
            rand_xoshiro::Xoshiro256Plus::seed_from_u64(seed)
        });
        $m!(bench_scalar_frand, "frand", frand::Rand::with_seed);
        $m!(bench_scalar_rngcore, "biski64", |seed| {
            biski64::Biski64Rng::from_seed_for_stream(seed, 0, 1)
        });
    };
}

macro_rules! for_each_top_portable_x8_case {
    ($m:ident) => {
        $m!(bench_portable_x8, "simd_rand/Portable/Xoshiro256+X8", |seed| {
            simd_rand::portable::Xoshiro256PlusX8::seed_from_u64(seed)
        });
        $m!(bench_portable_x8, "simd_rand/Portable/FrandX8", |seed| {
            simd_rand::portable::FrandX8::seed_from_u64(seed)
        });
        $m!(bench_portable_x8, "simd_rand/Portable/Biski64X8", |seed| {
            simd_rand::portable::Biski64X8::seed_from_u64(seed)
        });
    };
}

macro_rules! for_each_top_specific_x8_case {
    ($m:ident) => {
        #[cfg(all(
            feature = "specific",
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512dq",
            target_feature = "avx512vl"
        ))]
        $m!(bench_specific_x8, "simd_rand/Specific/Xoshiro256+X8", |seed| {
            simd_rand::specific::avx512::Xoshiro256PlusX8::seed_from_u64(seed)
        });
        #[cfg(all(
            feature = "specific",
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512dq",
            target_feature = "avx512vl"
        ))]
        $m!(bench_specific_x8, "simd_rand/Specific/FrandX8", |seed| {
            simd_rand::specific::avx512::FrandX8::seed_from_u64(seed)
        });
        #[cfg(all(
            feature = "specific",
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512dq",
            target_feature = "avx512vl"
        ))]
        $m!(bench_specific_x8, "simd_rand/Specific/Biski64X8", |seed| {
            simd_rand::specific::avx512::Biski64X8::seed_from_u64(seed)
        });
    };
}

fn bench_scalar_rngcore<M: Measurement, const ITERATIONS: usize, R: RngCore>(
    group: &mut BenchmarkGroup<'_, M>,
    label: &str,
    init: u64,
    make_rng: impl Fn(u64) -> R + Copy,
) {
    group.bench_function(label, |b| {
        let mut rng = make_rng(init);

        b.iter(|| {
            let mut data = u64x8::splat(init);

            for _ in 0..ITERATIONS {
                for i in 0..8 {
                    data[i] += rng.next_u64();
                }
            }

            data
        });
    });
}

fn bench_scalar_frand<M: Measurement, const ITERATIONS: usize>(
    group: &mut BenchmarkGroup<'_, M>,
    label: &str,
    init: u64,
    make_rng: impl Fn(u64) -> Rand + Copy,
) {
    group.bench_function(label, |b| {
        let mut rng = make_rng(init);

        b.iter(|| {
            let mut data = u64x8::splat(init);

            for _ in 0..ITERATIONS {
                for i in 0..8 {
                    data[i] += rng.r#gen::<u64>();
                }
            }

            data
        });
    });
}

fn bench_portable_x8<M: Measurement, const ITERATIONS: usize, R: SimdRandX8>(
    group: &mut BenchmarkGroup<'_, M>,
    label: &str,
    init: u64,
    make_rng: impl Fn(u64) -> R + Copy,
) {
    group.bench_function(label, |b| {
        let mut rng = make_rng(init);

        b.iter(|| {
            let mut data = u64x8::splat(init);

            for _ in 0..ITERATIONS {
                data += rng.next_u64x8();
            }

            data
        });
    });
}

#[cfg(all(
    feature = "specific",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl"
))]
fn bench_specific_x8<M: Measurement, const ITERATIONS: usize, R: specific::avx512::SimdRand>(
    group: &mut BenchmarkGroup<'_, M>,
    label: &str,
    init: u64,
    init_i: i64,
    make_rng: impl Fn(u64) -> R + Copy,
) {
    group.bench_function(label, |b| unsafe {
        let mut rng = make_rng(init);

        b.iter(|| {
            let mut data = _mm512_set1_epi64(init_i);

            for _ in 0..ITERATIONS {
                data = _mm512_add_epi64(data, specific::avx512::SimdRand::next_m512i(&mut rng));
            }

            data
        });
    });
}

fn add_scalar_top_benchmarks<M: Measurement, const ITERATIONS: usize>(group: &mut BenchmarkGroup<'_, M>, init: u64) {
    macro_rules! register_scalar_case {
        (bench_scalar_rngcore, $label:literal, $make_rng:expr) => {
            bench_scalar_rngcore::<_, ITERATIONS, _>(group, $label, init, $make_rng);
        };
        (bench_scalar_frand, $label:literal, $make_rng:expr) => {
            bench_scalar_frand::<_, ITERATIONS>(group, $label, init, $make_rng);
        };
    }

    for_each_top_scalar_case!(register_scalar_case);
}

fn add_portable_top_benchmarks<M: Measurement, const ITERATIONS: usize>(group: &mut BenchmarkGroup<'_, M>, init: u64) {
    macro_rules! register_portable_case {
        ($helper:ident, $label:literal, $make_rng:expr) => {
            $helper::<_, ITERATIONS, _>(group, $label, init, $make_rng);
        };
    }

    for_each_top_portable_x8_case!(register_portable_case);
}

#[cfg(all(
    feature = "specific",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl"
))]
fn add_specific_top_benchmarks<M: Measurement, const ITERATIONS: usize>(
    group: &mut BenchmarkGroup<'_, M>,
    init: u64,
    init_i: i64,
) {
    macro_rules! register_specific_case {
        ($helper:ident, $label:literal, $make_rng:expr) => {
            $helper::<_, ITERATIONS, _>(group, $label, init, init_i, $make_rng);
        };
    }

    for_each_top_specific_x8_case!(register_specific_case);
}

pub fn add_top_benchmark<M: Measurement, const ITERATIONS: usize>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Top");

    group.throughput(Throughput::ElementsAndBytes {
        elements: (ITERATIONS * 8) as u64,
        bytes: (ITERATIONS * mem::size_of::<u64x8>()) as u64,
    });
    group.noise_threshold(0.03);

    let mut rng = rand::rng();
    let init = rng.next_u64();

    add_scalar_top_benchmarks::<_, ITERATIONS>(&mut group, init);
    add_portable_top_benchmarks::<_, ITERATIONS>(&mut group, init);

    #[cfg(all(
        feature = "specific",
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512dq",
        target_feature = "avx512vl"
    ))]
    add_specific_top_benchmarks::<_, ITERATIONS>(&mut group, init, init.cast_signed());

    group.finish();
}
