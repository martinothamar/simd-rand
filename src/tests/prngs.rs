use crate::frand::test_support::{FrandReference, ref_seed_x4 as ref_seed_frand_x4, ref_seed_x8 as ref_seed_frand_x8};
#[cfg(feature = "portable")]
use crate::portable::{
    Biski64X4, Biski64X4Seed, Biski64X8, Biski64X8Seed, FrandX4, FrandX4Seed, FrandX8, FrandX8Seed, SimdRandX4,
    SimdRandX8, Xoshiro256PlusPlusX4, Xoshiro256PlusPlusX4Seed, Xoshiro256PlusPlusX8, Xoshiro256PlusPlusX8Seed,
    Xoshiro256PlusX4, Xoshiro256PlusX4Seed, Xoshiro256PlusX8, Xoshiro256PlusX8Seed,
};
#[cfg(all(feature = "specific", target_arch = "x86_64", target_feature = "avx2"))]
use crate::specific::avx2::{
    Biski64X4 as SpecificBiski64X4, Biski64X4Seed as SpecificBiski64X4Seed, DEFAULT_BUFFER_SIZE,
    FrandX4 as SpecificFrandX4, FrandX4Seed as SpecificFrandX4Seed, Shishua, SimdRand as SpecificSimdRandX4,
    Xoshiro256PlusPlusX4 as SpecificXoshiro256PlusPlusX4, Xoshiro256PlusPlusX4Seed as SpecificXoshiro256PlusPlusX4Seed,
    Xoshiro256PlusX4 as SpecificXoshiro256PlusX4, Xoshiro256PlusX4Seed as SpecificXoshiro256PlusX4Seed,
    shishua_test_vectors,
};
#[cfg(all(
    feature = "specific",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl"
))]
use crate::specific::avx512::{
    Biski64X8 as SpecificBiski64X8, Biski64X8Seed as SpecificBiski64X8Seed, FrandX8 as SpecificFrandX8,
    FrandX8Seed as SpecificFrandX8Seed, SimdRand as SpecificSimdRandX8,
    Xoshiro256PlusPlusX8 as SpecificXoshiro256PlusPlusX8, Xoshiro256PlusPlusX8Seed as SpecificXoshiro256PlusPlusX8Seed,
    Xoshiro256PlusX8 as SpecificXoshiro256PlusX8, Xoshiro256PlusX8Seed as SpecificXoshiro256PlusX8Seed,
};
use core::{fmt::Debug, fmt::Display, ops::Range};
use num_traits::{Num, NumCast};
use rand_core::{RngCore, SeedableRng};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use rust_decimal_macros::dec;

const DOUBLE_RANGE: Range<f64> = 0.0..1.0;

fn seed_bytes<const BYTES: usize>(values: &[u64]) -> [u8; BYTES] {
    let mut seed = [0u8; BYTES];
    assert_eq!(values.len() * 8, BYTES);

    for (chunk, value) in seed.chunks_exact_mut(8).zip(values) {
        chunk.copy_from_slice(&value.to_le_bytes());
    }

    seed
}

fn repeated_lane_seed<const BYTES: usize>(values: &[u64], lanes: usize) -> [u8; BYTES] {
    let mut seed = [0u8; BYTES];
    let words = BYTES / 8;

    assert_eq!(values.len() * lanes, words);

    for (index, chunk) in seed.chunks_exact_mut(8).enumerate() {
        chunk.copy_from_slice(&values[index / lanes].to_le_bytes());
    }

    seed
}

fn sequential_words<const WORDS: usize>() -> [u64; WORDS] {
    core::array::from_fn(|index| (index + 1) as u64)
}

fn ref_seed_biski64_x4() -> [u8; 32] {
    repeated_lane_seed::<32>(&sequential_words::<1>(), 4)
}

fn ref_seed_biski64_x8() -> [u8; 64] {
    repeated_lane_seed::<64>(&sequential_words::<1>(), 8)
}

fn xoshiro_reference_seed() -> [u8; 32] {
    seed_bytes::<32>(&sequential_words::<4>())
}

fn ref_seed_256() -> [u8; 128] {
    repeated_lane_seed::<128>(&sequential_words::<4>(), 4)
}

fn ref_seed_512() -> [u8; 256] {
    repeated_lane_seed::<256>(&sequential_words::<4>(), 8)
}

fn random_seeded_rng<R>() -> R
where
    R: SeedableRng,
    R::Seed: Default + AsMut<[u8]>,
{
    let mut seed = R::Seed::default();
    rand::rng().fill_bytes(seed.as_mut());
    R::from_seed(seed)
}

#[allow(clippy::items_after_statements, clippy::unwrap_used)]
fn test_uniform_distribution<const SAMPLES: usize, T>(mut f: impl FnMut() -> T, range: Range<T>)
where
    T: Num + NumCast + Display + TryInto<Decimal>,
    <T as TryInto<Decimal>>::Error: Debug,
{
    let mut dist: alloc::vec::Vec<Decimal> = alloc::vec::Vec::with_capacity(SAMPLES);

    let range_start: Decimal = range.start.try_into().unwrap();
    let range_end: Decimal = range.end.try_into().unwrap();
    let range = range_start..range_end;

    let mut sum = Decimal::ZERO;
    for _ in 0..SAMPLES {
        let value: Decimal = f().try_into().unwrap();
        assert!(value >= range.start && value < range.end);
        sum += value;
        dist.push(value);
    }

    let samples_divisor: Decimal = SAMPLES.into();
    let mean: Decimal = sum / samples_divisor;

    let mut squared_diffs = Decimal::ZERO;
    for n in dist {
        let diff = (n - mean).powi(2);
        squared_diffs += diff;
    }

    const DIFF_LIMIT: Decimal = dec!(0.001);

    let expected_mean = (range.start + range.end - DIFF_LIMIT) / dec!(2.0);
    let mean_difference = (mean - expected_mean).abs();

    let variance = squared_diffs / samples_divisor;
    let expected_variance = (range.end - range.start).powi(2) / dec!(12.0);
    let variance_difference = (variance - expected_variance).abs();

    let stddev = variance.sqrt().unwrap();
    let expected_stddev = expected_variance.sqrt().unwrap();
    let stddev_difference = (stddev - expected_stddev).abs();

    assert!(
        mean_difference <= DIFF_LIMIT,
        "Mean difference was more than {DIFF_LIMIT:.5}: {mean_difference:.5}. Expected mean: {expected_mean:.6}, actual mean: {mean:.6}"
    );
    assert!(
        variance_difference <= DIFF_LIMIT,
        "Variance difference was more than {DIFF_LIMIT:.5}: {variance_difference:.5}. Expected variance: {expected_variance:.6}, actual mean: {variance:.6}"
    );
    assert!(
        stddev_difference <= DIFF_LIMIT,
        "Std deviation difference was more than {DIFF_LIMIT:.5}: {stddev_difference:.5}. Expected std deviation: {expected_stddev:.6}, actual mean: {stddev:.6}"
    );
}

fn assert_matches_scalar_reference<const LANES: usize, R>(
    mut rng: R,
    mut next: impl FnMut(&mut R) -> [u64; LANES],
    mut reference: impl FnMut() -> u64,
) {
    for _ in 0..10 {
        assert_eq!(next(&mut rng), [reference(); LANES]);
    }
}

fn assert_nonzero_unique<const LANES: usize>(values: [u64; LANES]) {
    assert!(values.iter().all(|&value| value != 0));

    for left in 0..LANES {
        for right in (left + 1)..LANES {
            assert_ne!(values[left], values[right]);
        }
    }
}

fn assert_nonzero_f64<const LANES: usize>(values: [f64; LANES]) {
    assert!(values.iter().all(|&value| value != 0.0));
}

fn assert_u64_smoke<const LANES: usize, R>(mut rng: R, mut next: impl FnMut(&mut R) -> [u64; LANES]) {
    assert_nonzero_unique(next(&mut rng));
    assert_nonzero_unique(next(&mut rng));
}

fn assert_f64_smoke<const LANES: usize, R>(mut rng: R, mut next: impl FnMut(&mut R) -> [f64; LANES]) {
    assert_nonzero_f64(next(&mut rng));
    assert_nonzero_f64(next(&mut rng));
}

fn assert_f64_distribution<const LANES: usize, R>(mut rng: R, mut next: impl FnMut(&mut R) -> [f64; LANES]) {
    let mut current: Option<[f64; LANES]> = None;
    let mut current_index = 0;

    test_uniform_distribution::<10_000_000, f64>(
        || match current {
            Some(vector) if current_index < LANES => {
                let result = vector[current_index];
                current_index += 1;
                result
            }
            _ => {
                current_index = 1;
                let vector = next(&mut rng);
                let result = vector[0];
                current = Some(vector);
                result
            }
        },
        DOUBLE_RANGE,
    );
}

macro_rules! define_prng_tests {
    (
        $(#[$meta:meta])*
        $module:ident,
        lanes = $lanes:expr,
        rng = $rng_ty:path,
        seed = $seed_ty:path,
        ref_seed = $ref_seed:expr,
        reference_seed = $reference_seed:expr,
        reference_rng = $reference_rng:expr,
        reference_next = $reference_next:expr,
        next_u64 = $next_u64:expr,
        next_f64 = $next_f64:expr
    ) => {
        $(#[$meta])*
        mod $module {
            use super::*;

            #[test]
            fn reference() {
                let rng: $rng_ty = <$rng_ty>::from_seed(<$seed_ty>::from($ref_seed));
                let mut reference = ($reference_rng)($reference_seed);
                let reference_next = $reference_next;
                assert_matches_scalar_reference::<$lanes, _>(rng, $next_u64, || reference_next(&mut reference));
            }

            #[test]
            fn sample_u64() {
                let rng = random_seeded_rng::<$rng_ty>();
                assert_u64_smoke::<$lanes, _>(rng, $next_u64);
            }

            #[test]
            fn sample_f64() {
                let rng = random_seeded_rng::<$rng_ty>();
                assert_f64_smoke::<$lanes, _>(rng, $next_f64);
            }

            #[test]
            #[cfg_attr(any(debug_assertions, miri), ignore = "distribution test requires release mode and real RNG")]
            fn distribution() {
                let rng = random_seeded_rng::<$rng_ty>();
                assert_f64_distribution::<$lanes, _>(rng, $next_f64);
            }
        }
    };
}

#[cfg(feature = "portable")]
define_prng_tests!(
    portable_frand_x4,
    lanes = 4,
    rng = FrandX4,
    seed = FrandX4Seed,
    ref_seed = ref_seed_frand_x4(),
    reference_seed = 1u64,
    reference_rng = FrandReference::new,
    reference_next = |rng: &mut FrandReference| rng.next_u64(),
    next_u64 = |rng: &mut FrandX4| rng.next_u64x4().to_array(),
    next_f64 = |rng: &mut FrandX4| rng.next_f64x4().to_array()
);

#[cfg(feature = "portable")]
define_prng_tests!(
    portable_frand_x8,
    lanes = 8,
    rng = FrandX8,
    seed = FrandX8Seed,
    ref_seed = ref_seed_frand_x8(),
    reference_seed = 1u64,
    reference_rng = FrandReference::new,
    reference_next = |rng: &mut FrandReference| rng.next_u64(),
    next_u64 = |rng: &mut FrandX8| rng.next_u64x8().to_array(),
    next_f64 = |rng: &mut FrandX8| rng.next_f64x8().to_array()
);

#[cfg(feature = "portable")]
define_prng_tests!(
    portable_xoshiro256plus_x4,
    lanes = 4,
    rng = Xoshiro256PlusX4,
    seed = Xoshiro256PlusX4Seed,
    ref_seed = ref_seed_256(),
    reference_seed = xoshiro_reference_seed(),
    reference_rng = rand_xoshiro::Xoshiro256Plus::from_seed,
    reference_next = |rng: &mut rand_xoshiro::Xoshiro256Plus| rng.next_u64(),
    next_u64 = |rng: &mut Xoshiro256PlusX4| rng.next_u64x4().to_array(),
    next_f64 = |rng: &mut Xoshiro256PlusX4| rng.next_f64x4().to_array()
);

#[cfg(feature = "portable")]
define_prng_tests!(
    portable_xoshiro256plus_x8,
    lanes = 8,
    rng = Xoshiro256PlusX8,
    seed = Xoshiro256PlusX8Seed,
    ref_seed = ref_seed_512(),
    reference_seed = xoshiro_reference_seed(),
    reference_rng = rand_xoshiro::Xoshiro256Plus::from_seed,
    reference_next = |rng: &mut rand_xoshiro::Xoshiro256Plus| rng.next_u64(),
    next_u64 = |rng: &mut Xoshiro256PlusX8| rng.next_u64x8().to_array(),
    next_f64 = |rng: &mut Xoshiro256PlusX8| rng.next_f64x8().to_array()
);

#[cfg(feature = "portable")]
define_prng_tests!(
    portable_xoshiro256plusplus_x4,
    lanes = 4,
    rng = Xoshiro256PlusPlusX4,
    seed = Xoshiro256PlusPlusX4Seed,
    ref_seed = ref_seed_256(),
    reference_seed = xoshiro_reference_seed(),
    reference_rng = rand_xoshiro::Xoshiro256PlusPlus::from_seed,
    reference_next = |rng: &mut rand_xoshiro::Xoshiro256PlusPlus| rng.next_u64(),
    next_u64 = |rng: &mut Xoshiro256PlusPlusX4| rng.next_u64x4().to_array(),
    next_f64 = |rng: &mut Xoshiro256PlusPlusX4| rng.next_f64x4().to_array()
);

#[cfg(feature = "portable")]
define_prng_tests!(
    portable_xoshiro256plusplus_x8,
    lanes = 8,
    rng = Xoshiro256PlusPlusX8,
    seed = Xoshiro256PlusPlusX8Seed,
    ref_seed = ref_seed_512(),
    reference_seed = xoshiro_reference_seed(),
    reference_rng = rand_xoshiro::Xoshiro256PlusPlus::from_seed,
    reference_next = |rng: &mut rand_xoshiro::Xoshiro256PlusPlus| rng.next_u64(),
    next_u64 = |rng: &mut Xoshiro256PlusPlusX8| rng.next_u64x8().to_array(),
    next_f64 = |rng: &mut Xoshiro256PlusPlusX8| rng.next_f64x8().to_array()
);

#[cfg(feature = "portable")]
define_prng_tests!(
    portable_biski64_x4,
    lanes = 4,
    rng = Biski64X4,
    seed = Biski64X4Seed,
    ref_seed = ref_seed_biski64_x4(),
    reference_seed = 1u64,
    reference_rng = |seed| biski64::Biski64Rng::from_seed_for_stream(seed, 0, 1),
    reference_next = |rng: &mut biski64::Biski64Rng| rng.next_u64(),
    next_u64 = |rng: &mut Biski64X4| rng.next_u64x4().to_array(),
    next_f64 = |rng: &mut Biski64X4| rng.next_f64x4().to_array()
);

#[cfg(feature = "portable")]
define_prng_tests!(
    portable_biski64_x8,
    lanes = 8,
    rng = Biski64X8,
    seed = Biski64X8Seed,
    ref_seed = ref_seed_biski64_x8(),
    reference_seed = 1u64,
    reference_rng = |seed| biski64::Biski64Rng::from_seed_for_stream(seed, 0, 1),
    reference_next = |rng: &mut biski64::Biski64Rng| rng.next_u64(),
    next_u64 = |rng: &mut Biski64X8| rng.next_u64x8().to_array(),
    next_f64 = |rng: &mut Biski64X8| rng.next_f64x8().to_array()
);

#[cfg(all(feature = "specific", target_arch = "x86_64", target_feature = "avx2"))]
define_prng_tests!(
    specific_avx2_frand_x4,
    lanes = 4,
    rng = SpecificFrandX4,
    seed = SpecificFrandX4Seed,
    ref_seed = ref_seed_frand_x4(),
    reference_seed = 1u64,
    reference_rng = FrandReference::new,
    reference_next = |rng: &mut FrandReference| rng.next_u64(),
    next_u64 = |rng: &mut SpecificFrandX4| *rng.next_u64x4(),
    next_f64 = |rng: &mut SpecificFrandX4| *rng.next_f64x4()
);

#[cfg(all(feature = "specific", target_arch = "x86_64", target_feature = "avx2"))]
define_prng_tests!(
    specific_avx2_xoshiro256plus_x4,
    lanes = 4,
    rng = SpecificXoshiro256PlusX4,
    seed = SpecificXoshiro256PlusX4Seed,
    ref_seed = ref_seed_256(),
    reference_seed = xoshiro_reference_seed(),
    reference_rng = rand_xoshiro::Xoshiro256Plus::from_seed,
    reference_next = |rng: &mut rand_xoshiro::Xoshiro256Plus| rng.next_u64(),
    next_u64 = |rng: &mut SpecificXoshiro256PlusX4| *rng.next_u64x4(),
    next_f64 = |rng: &mut SpecificXoshiro256PlusX4| *rng.next_f64x4()
);

#[cfg(all(feature = "specific", target_arch = "x86_64", target_feature = "avx2"))]
define_prng_tests!(
    specific_avx2_xoshiro256plusplus_x4,
    lanes = 4,
    rng = SpecificXoshiro256PlusPlusX4,
    seed = SpecificXoshiro256PlusPlusX4Seed,
    ref_seed = ref_seed_256(),
    reference_seed = xoshiro_reference_seed(),
    reference_rng = rand_xoshiro::Xoshiro256PlusPlus::from_seed,
    reference_next = |rng: &mut rand_xoshiro::Xoshiro256PlusPlus| rng.next_u64(),
    next_u64 = |rng: &mut SpecificXoshiro256PlusPlusX4| *rng.next_u64x4(),
    next_f64 = |rng: &mut SpecificXoshiro256PlusPlusX4| *rng.next_f64x4()
);

#[cfg(all(feature = "specific", target_arch = "x86_64", target_feature = "avx2"))]
define_prng_tests!(
    specific_avx2_biski64_x4,
    lanes = 4,
    rng = SpecificBiski64X4,
    seed = SpecificBiski64X4Seed,
    ref_seed = ref_seed_biski64_x4(),
    reference_seed = 1u64,
    reference_rng = |seed| biski64::Biski64Rng::from_seed_for_stream(seed, 0, 1),
    reference_next = |rng: &mut biski64::Biski64Rng| rng.next_u64(),
    next_u64 = |rng: &mut SpecificBiski64X4| *rng.next_u64x4(),
    next_f64 = |rng: &mut SpecificBiski64X4| *rng.next_f64x4()
);

#[cfg(all(
    feature = "specific",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl"
))]
define_prng_tests!(
    specific_avx512_frand_x8,
    lanes = 8,
    rng = SpecificFrandX8,
    seed = SpecificFrandX8Seed,
    ref_seed = ref_seed_frand_x8(),
    reference_seed = 1u64,
    reference_rng = FrandReference::new,
    reference_next = |rng: &mut FrandReference| rng.next_u64(),
    next_u64 = |rng: &mut SpecificFrandX8| *rng.next_u64x8(),
    next_f64 = |rng: &mut SpecificFrandX8| *rng.next_f64x8()
);

#[cfg(all(
    feature = "specific",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl"
))]
define_prng_tests!(
    specific_avx512_xoshiro256plus_x8,
    lanes = 8,
    rng = SpecificXoshiro256PlusX8,
    seed = SpecificXoshiro256PlusX8Seed,
    ref_seed = ref_seed_512(),
    reference_seed = xoshiro_reference_seed(),
    reference_rng = rand_xoshiro::Xoshiro256Plus::from_seed,
    reference_next = |rng: &mut rand_xoshiro::Xoshiro256Plus| rng.next_u64(),
    next_u64 = |rng: &mut SpecificXoshiro256PlusX8| *rng.next_u64x8(),
    next_f64 = |rng: &mut SpecificXoshiro256PlusX8| *rng.next_f64x8()
);

#[cfg(all(
    feature = "specific",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl"
))]
define_prng_tests!(
    specific_avx512_xoshiro256plusplus_x8,
    lanes = 8,
    rng = SpecificXoshiro256PlusPlusX8,
    seed = SpecificXoshiro256PlusPlusX8Seed,
    ref_seed = ref_seed_512(),
    reference_seed = xoshiro_reference_seed(),
    reference_rng = rand_xoshiro::Xoshiro256PlusPlus::from_seed,
    reference_next = |rng: &mut rand_xoshiro::Xoshiro256PlusPlus| rng.next_u64(),
    next_u64 = |rng: &mut SpecificXoshiro256PlusPlusX8| *rng.next_u64x8(),
    next_f64 = |rng: &mut SpecificXoshiro256PlusPlusX8| *rng.next_f64x8()
);

#[cfg(all(
    feature = "specific",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl"
))]
define_prng_tests!(
    specific_avx512_biski64_x8,
    lanes = 8,
    rng = SpecificBiski64X8,
    seed = SpecificBiski64X8Seed,
    ref_seed = ref_seed_biski64_x8(),
    reference_seed = 1u64,
    reference_rng = |seed| biski64::Biski64Rng::from_seed_for_stream(seed, 0, 1),
    reference_next = |rng: &mut biski64::Biski64Rng| rng.next_u64(),
    next_u64 = |rng: &mut SpecificBiski64X8| *rng.next_u64x8(),
    next_f64 = |rng: &mut SpecificBiski64X8| *rng.next_f64x8()
);

#[cfg(all(feature = "specific", target_arch = "x86_64", target_feature = "avx2"))]
mod specific_avx2_shishua {
    use alloc::vec::Vec;
    use rand::Rng;
    use rand_core::TryRngCore;

    use super::*;

    type DefaultShishua = Shishua<DEFAULT_BUFFER_SIZE>;

    const FLOAT_RANGE: Range<f32> = 0.0f32..1.0f32;

    fn read_with_next_u64<const N: usize>(rng: &mut Shishua<N>, total_bytes: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(total_bytes);
        let mut remaining = total_bytes;

        while remaining >= 8 {
            bytes.extend_from_slice(&rng.next_u64().to_le_bytes());
            remaining -= 8;
        }

        if remaining > 0 {
            let value = rng.next_u64().to_le_bytes();
            bytes.extend_from_slice(&value[..remaining]);
        }

        bytes
    }

    fn read_with_chunks<const N: usize>(rng: &mut Shishua<N>, total_bytes: usize, chunk_sizes: &[usize]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(total_bytes);
        let mut offset = 0;
        let mut chunk_index = 0;

        while offset < total_bytes {
            let remaining = total_bytes - offset;
            let chunk = chunk_sizes[chunk_index % chunk_sizes.len()].min(remaining).min(N);
            let start = bytes.len();
            bytes.resize(start + chunk, 0);
            rng.fill_bytes(&mut bytes[start..start + chunk]);
            offset += chunk;
            chunk_index += 1;
        }

        bytes
    }

    fn assert_chunked_deterministic<const N: usize>(seed: [u8; 32], total_bytes: usize, chunk_sizes: &[usize]) {
        let mut rng_a = Shishua::<N>::from_seed(seed);
        let mut rng_b = Shishua::<N>::from_seed(seed);

        let bytes_a = read_with_chunks(&mut rng_a, total_bytes, chunk_sizes);
        let bytes_b = read_with_chunks(&mut rng_b, total_bytes, chunk_sizes);

        assert_eq!(bytes_a, bytes_b);
    }

    #[test]
    fn scalar_smoke() {
        let mut rng = DefaultShishua::from_seed([0; 32]);

        assert_eq!(rng.buffer_index(), 0);
        let value_f64 = rng.random_range(DOUBLE_RANGE);
        let value_f32 = rng.random_range(FLOAT_RANGE);

        assert_ne!(rng.next_u32(), 0);
        assert_ne!(rng.next_u64(), 0);
        assert!(DOUBLE_RANGE.contains(&value_f64) && value_f64 != 0.0);
        assert!(FLOAT_RANGE.contains(&value_f32) && value_f32 != 0.0);
    }

    #[test]
    fn vector_smoke() {
        let rng = DefaultShishua::from_seed(shishua_test_vectors::seed_bytes(shishua_test_vectors::SEED_PI));

        assert_u64_smoke::<4, _>(rng, |rng: &mut DefaultShishua| *rng.next_u64x4());

        let rng = DefaultShishua::from_seed(shishua_test_vectors::seed_bytes(shishua_test_vectors::SEED_PI));
        assert_f64_smoke::<4, _>(rng, |rng: &mut DefaultShishua| *rng.next_f64x4());
    }

    #[test]
    fn fill_bytes_matches_next_u64_stream() {
        const REF_BYTES: usize = 512;
        let seed = shishua_test_vectors::seed_bytes(shishua_test_vectors::SEED_PI);
        let mut rng_words = DefaultShishua::from_seed(seed);
        let mut rng_bytes = DefaultShishua::from_seed(seed);

        let from_words = read_with_next_u64(&mut rng_words, REF_BYTES);
        let mut from_bytes = vec![0u8; REF_BYTES];
        rng_bytes.fill_bytes(&mut from_bytes);

        assert_eq!(from_words, from_bytes);
    }

    #[test]
    fn fill_bytes_matches_reference_prefix() {
        let mut rng_zero = DefaultShishua::from_seed(shishua_test_vectors::seed_bytes(shishua_test_vectors::SEED_ZERO));
        let mut zero = [0u8; 64];
        rng_zero.fill_bytes(&mut zero);
        assert_eq!(&zero[..], &shishua_test_vectors::SEED_ZERO_EXPECTED[..zero.len()]);

        let mut rng_pi = DefaultShishua::from_seed(shishua_test_vectors::seed_bytes(shishua_test_vectors::SEED_PI));
        let mut pi = [0u8; 64];
        rng_pi.fill_bytes(&mut pi);
        assert_eq!(&pi[..], &shishua_test_vectors::SEED_PI_EXPECTED[..pi.len()]);
    }

    #[test]
    fn fill_bytes_chunking_is_deterministic() {
        let chunk_sizes = [1, 7, 8, 13, 32, 64, 100, 255, 1000];
        let total_bytes = DEFAULT_BUFFER_SIZE + 500;

        assert_chunked_deterministic::<DEFAULT_BUFFER_SIZE>([0; 32], total_bytes, &chunk_sizes);
    }

    #[test]
    fn non_default_buffer_sizes_are_deterministic() {
        let chunk_sizes = [1, 7, 8, 13, 32, 64, 100, 255];
        let total_bytes = 2048;
        let seed = shishua_test_vectors::seed_bytes(shishua_test_vectors::SEED_PI);

        assert_chunked_deterministic::<256>(seed, total_bytes, &chunk_sizes);
        assert_chunked_deterministic::<512>(seed, total_bytes, &chunk_sizes);
        assert_chunked_deterministic::<1024>(seed, total_bytes, &chunk_sizes);
    }

    #[test]
    fn try_fill_bytes_matches_fill_bytes() {
        const LEN: usize = 256;
        let mut rng_try = DefaultShishua::from_seed([0; 32]);
        let mut rng_fill = DefaultShishua::from_seed([0; 32]);

        let mut try_buf = [0u8; LEN];
        let mut fill_buf = [0u8; LEN];
        assert!(rng_try.try_fill_bytes(&mut try_buf).is_ok());
        rng_fill.fill_bytes(&mut fill_buf);

        assert_eq!(try_buf, fill_buf);
    }

    #[test]
    #[cfg_attr(
        any(debug_assertions, miri),
        ignore = "distribution test requires release mode and real RNG"
    )]
    fn sample_f64_distribution() {
        let mut rng = DefaultShishua::from_seed([0; 32]);

        test_uniform_distribution::<10_000_000, f64>(|| rng.random_range(DOUBLE_RANGE), DOUBLE_RANGE);
    }

    #[test]
    #[cfg_attr(
        any(debug_assertions, miri),
        ignore = "distribution test requires release mode and real RNG"
    )]
    fn sample_f64x4_distribution() {
        let rng = DefaultShishua::from_seed([0; 32]);
        assert_f64_distribution::<4, _>(rng, |rng: &mut DefaultShishua| *rng.next_f64x4());
    }

    #[test]
    #[cfg_attr(
        any(debug_assertions, miri),
        ignore = "distribution test requires release mode and real RNG"
    )]
    fn sample_f32_distribution() {
        let mut rng = DefaultShishua::from_seed([0; 32]);

        test_uniform_distribution::<10_000_000, f32>(|| rng.random_range(FLOAT_RANGE), FLOAT_RANGE);
    }
}
