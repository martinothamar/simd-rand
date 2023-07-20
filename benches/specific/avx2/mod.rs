use std::{mem, arch::x86_64::*};

use criterion::{measurement::Measurement, Criterion, Throughput, black_box};
use rand_core::SeedableRng;
use simd_rand::specific::avx2::*;

pub fn add_benchmarks<M: Measurement, const ITERATIONS: usize>(c: &mut Criterion<M>, suffix: &str) {
    let group_prefix = "AVX2";
    add_m256i_benchmarks::<_, ITERATIONS>(c, group_prefix, suffix);
    add_m256d_benchmarks::<_, ITERATIONS>(c, group_prefix, suffix);
}

fn add_m256i_benchmarks<M: Measurement, const ITERATIONS: usize>(c: &mut Criterion<M>, group_prefix: &str, suffix: &str) {
    let mut group = c.benchmark_group(format!("{group_prefix}/m256i"));

    group.throughput(Throughput::Bytes((ITERATIONS * mem::size_of::<U64x4>()) as u64));

    #[inline(always)]
    fn execute<RNG: SimdPrng, const ITERATIONS: usize>(rng: &mut RNG, data: &mut __m256i) {
        for _ in 0..ITERATIONS {
            rng.next_m256i(black_box(data));
        }
    }

    let name = format!("Shishua - {suffix}");
    group.bench_function(name, |b| {
        unsafe { // TODO - better seed?
            let mut rng: Shishua = Shishua::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
            let mut data: __m256i = _mm256_setzero_si256();
    
            b.iter(|| execute::<_, ITERATIONS>(&mut rng, black_box(&mut data)))
        }
    });

    let name = format!("Xoshiro256++ - {suffix}");
    group.bench_function(name, |b| {
        unsafe {
            let mut rng: Xoshiro256PlusPlusX4 = Xoshiro256PlusPlusX4::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
            let mut data: __m256i = _mm256_setzero_si256();

            b.iter(|| execute::<_, ITERATIONS>(&mut rng, black_box(&mut data)))
        }
    });

    let name = format!("Xoshiro256+ - {suffix}");
    group.bench_function(name, |b| {
        unsafe {
            let mut rng: Xoshiro256PlusX4 = Xoshiro256PlusX4::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
            let mut data: __m256i = _mm256_setzero_si256();

            b.iter(|| execute::<_, ITERATIONS>(&mut rng, black_box(&mut data)))
        }
    });
}

fn add_m256d_benchmarks<M: Measurement, const ITERATIONS: usize>(c: &mut Criterion<M>, group_prefix: &str, suffix: &str) {
    let mut group = c.benchmark_group(format!("{group_prefix}/m256d"));

    group.throughput(Throughput::Bytes((ITERATIONS * mem::size_of::<F64x4>()) as u64));

    #[inline(always)]
    fn execute<RNG: SimdPrng, const ITERATIONS: usize>(rng: &mut RNG, data: &mut __m256d) {
        for _ in 0..ITERATIONS {
            rng.next_m256d(black_box(data));
        }
    }

    let name = format!("Shishua - {suffix}");
    group.bench_function(name, |b| {
        unsafe { // TODO - better seed?
            let mut rng: Shishua = Shishua::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
            let mut data: __m256d = _mm256_setzero_pd();
    
            b.iter(|| execute::<_, ITERATIONS>(&mut rng, black_box(&mut data)))
        }
    });

    let name = format!("Xoshiro256++ - {suffix}");
    group.bench_function(name, |b| {
        unsafe {
            let mut rng: Xoshiro256PlusPlusX4 = Xoshiro256PlusPlusX4::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
            let mut data: __m256d = _mm256_setzero_pd();

            b.iter(|| execute::<_, ITERATIONS>(&mut rng, black_box(&mut data)))
        }
    });

    let name = format!("Xoshiro256+ - {suffix}");
    group.bench_function(name, |b| {
        unsafe {
            let mut rng: Xoshiro256PlusX4 = Xoshiro256PlusX4::seed_from_u64(0x0DDB1A5E5BAD5EEDu64);
            let mut data: __m256d = _mm256_setzero_pd();

            b.iter(|| execute::<_, ITERATIONS>(&mut rng, black_box(&mut data)))
        }
    });
}
