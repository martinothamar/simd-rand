#![cfg(all(feature = "portable", feature = "specific"))]
#![cfg_attr(feature = "portable", feature(portable_simd))]

use rand_core::SeedableRng;

#[path = "../src/testutil/fixed_u64_rng.rs"]
mod fixed_u64_rng;

use fixed_u64_rng::FixedBytesRng;

fn assert_same_vectors<const LANES: usize>(
    mut lhs: impl FnMut() -> [u64; LANES],
    mut rhs: impl FnMut() -> [u64; LANES],
) {
    for _ in 0..3 {
        assert_eq!(lhs(), rhs());
    }
}

fn fill_seed_32(words: &[u64; 4]) -> [u8; 32] {
    let mut seed = [0u8; 32];

    for (index, word) in words.iter().enumerate() {
        seed[(index * 8)..((index + 1) * 8)].copy_from_slice(&word.to_le_bytes());
    }

    seed
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl"
))]
fn fill_seed_64(words: &[u64; 8]) -> [u8; 64] {
    let mut seed = [0u8; 64];

    for (index, word) in words.iter().enumerate() {
        seed[(index * 8)..((index + 1) * 8)].copy_from_slice(&word.to_le_bytes());
    }

    seed
}

fn fill_seed_128(words: &[u64; 16]) -> [u8; 128] {
    let mut seed = [0u8; 128];

    for (index, word) in words.iter().enumerate() {
        seed[(index * 8)..((index + 1) * 8)].copy_from_slice(&word.to_le_bytes());
    }

    seed
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl"
))]
fn fill_seed_256(words: &[u64; 32]) -> [u8; 256] {
    let mut seed = [0u8; 256];

    for (index, word) in words.iter().enumerate() {
        seed[(index * 8)..((index + 1) * 8)].copy_from_slice(&word.to_le_bytes());
    }

    seed
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[test]
fn avx2_matches_portable_for_asymmetric_seeds() {
    use simd_rand::portable::{
        Biski64X4 as PortableBiski64X4, Biski64X4Seed as PortableBiski64X4Seed, FrandX4 as PortableFrandX4,
        FrandX4Seed as PortableFrandX4Seed, SimdRandX4, Xoshiro256PlusX4 as PortableXoshiro256PlusX4,
        Xoshiro256PlusX4Seed as PortableXoshiro256PlusX4Seed,
    };
    use simd_rand::specific::avx2::{
        Biski64X4 as SpecificBiski64X4, Biski64X4Seed as SpecificBiski64X4Seed, FrandX4 as SpecificFrandX4,
        FrandX4Seed as SpecificFrandX4Seed, SimdRand, Xoshiro256PlusX4 as SpecificXoshiro256PlusX4,
        Xoshiro256PlusX4Seed as SpecificXoshiro256PlusX4Seed,
    };

    let lane_seed = fill_seed_32(&[
        0x0123_4567_89AB_CDEF,
        0x1112_1314_1516_1718,
        0x2122_2324_2526_2728,
        0x3132_3334_3536_3738,
    ]);
    let mut portable_frand = PortableFrandX4::from_seed(PortableFrandX4Seed::from(lane_seed));
    let mut specific_frand = SpecificFrandX4::from_seed(SpecificFrandX4Seed::from(lane_seed));

    assert_same_vectors(
        || portable_frand.next_u64x4().to_array(),
        || *specific_frand.next_u64x4(),
    );

    let mut portable_biski = PortableBiski64X4::from_seed(PortableBiski64X4Seed::from(lane_seed));
    let mut specific_biski = SpecificBiski64X4::from_seed(SpecificBiski64X4Seed::from(lane_seed));

    assert_same_vectors(
        || portable_biski.next_u64x4().to_array(),
        || *specific_biski.next_u64x4(),
    );

    let mut portable_biski_from_u64 = PortableBiski64X4::seed_from_u64(42);
    let mut specific_biski_from_u64 = SpecificBiski64X4::seed_from_u64(42);

    assert_same_vectors(
        || portable_biski_from_u64.next_u64x4().to_array(),
        || *specific_biski_from_u64.next_u64x4(),
    );

    let mut portable_biski_from_bytes = PortableBiski64X4::from_rng(&mut FixedBytesRng::new(lane_seed));
    let mut specific_biski_from_bytes = SpecificBiski64X4::from_rng(&mut FixedBytesRng::new(lane_seed));

    assert_same_vectors(
        || portable_biski_from_bytes.next_u64x4().to_array(),
        || *specific_biski_from_bytes.next_u64x4(),
    );

    let xoshiro_seed = fill_seed_128(&[
        0x0001_0002_0003_0004,
        0x0101_0102_0103_0104,
        0x0201_0202_0203_0204,
        0x0301_0302_0303_0304,
        0x1001_1002_1003_1004,
        0x1101_1102_1103_1104,
        0x1201_1202_1203_1204,
        0x1301_1302_1303_1304,
        0x2001_2002_2003_2004,
        0x2101_2102_2103_2104,
        0x2201_2202_2203_2204,
        0x2301_2302_2303_2304,
        0x3001_3002_3003_3004,
        0x3101_3102_3103_3104,
        0x3201_3202_3203_3204,
        0x3301_3302_3303_3304,
    ]);
    let mut portable_xoshiro = PortableXoshiro256PlusX4::from_seed(PortableXoshiro256PlusX4Seed::from(xoshiro_seed));
    let mut specific_xoshiro = SpecificXoshiro256PlusX4::from_seed(SpecificXoshiro256PlusX4Seed::from(xoshiro_seed));

    assert_same_vectors(
        || portable_xoshiro.next_u64x4().to_array(),
        || *specific_xoshiro.next_u64x4(),
    );
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl"
))]
#[test]
fn avx512_matches_portable_for_asymmetric_seeds() {
    use simd_rand::portable::{
        Biski64X8 as PortableBiski64X8, Biski64X8Seed as PortableBiski64X8Seed, FrandX8 as PortableFrandX8,
        FrandX8Seed as PortableFrandX8Seed, SimdRandX8, Xoshiro256PlusX8 as PortableXoshiro256PlusX8,
        Xoshiro256PlusX8Seed as PortableXoshiro256PlusX8Seed,
    };
    use simd_rand::specific::avx512::{
        Biski64X8 as SpecificBiski64X8, Biski64X8Seed as SpecificBiski64X8Seed, FrandX8 as SpecificFrandX8,
        FrandX8Seed as SpecificFrandX8Seed, SimdRand, Xoshiro256PlusX8 as SpecificXoshiro256PlusX8,
        Xoshiro256PlusX8Seed as SpecificXoshiro256PlusX8Seed,
    };

    let lane_seed = fill_seed_64(&[
        0x0123_4567_89AB_CDEF,
        0x1112_1314_1516_1718,
        0x2122_2324_2526_2728,
        0x3132_3334_3536_3738,
        0x4142_4344_4546_4748,
        0x5152_5354_5556_5758,
        0x6162_6364_6566_6768,
        0x7172_7374_7576_7778,
    ]);
    let mut portable_frand = PortableFrandX8::from_seed(PortableFrandX8Seed::from(lane_seed));
    let mut specific_frand = SpecificFrandX8::from_seed(SpecificFrandX8Seed::from(lane_seed));

    assert_same_vectors(
        || portable_frand.next_u64x8().to_array(),
        || *specific_frand.next_u64x8(),
    );

    let mut portable_biski = PortableBiski64X8::from_seed(PortableBiski64X8Seed::from(lane_seed));
    let mut specific_biski = SpecificBiski64X8::from_seed(SpecificBiski64X8Seed::from(lane_seed));

    assert_same_vectors(
        || portable_biski.next_u64x8().to_array(),
        || *specific_biski.next_u64x8(),
    );

    let mut portable_biski_from_u64 = PortableBiski64X8::seed_from_u64(42);
    let mut specific_biski_from_u64 = SpecificBiski64X8::seed_from_u64(42);

    assert_same_vectors(
        || portable_biski_from_u64.next_u64x8().to_array(),
        || *specific_biski_from_u64.next_u64x8(),
    );

    let mut portable_biski_from_bytes = PortableBiski64X8::from_rng(&mut FixedBytesRng::new(lane_seed));
    let mut specific_biski_from_bytes = SpecificBiski64X8::from_rng(&mut FixedBytesRng::new(lane_seed));

    assert_same_vectors(
        || portable_biski_from_bytes.next_u64x8().to_array(),
        || *specific_biski_from_bytes.next_u64x8(),
    );

    let xoshiro_seed = fill_seed_256(&[
        0x0001_0002_0003_0004,
        0x0101_0102_0103_0104,
        0x0201_0202_0203_0204,
        0x0301_0302_0303_0304,
        0x0401_0402_0403_0404,
        0x0501_0502_0503_0504,
        0x0601_0602_0603_0604,
        0x0701_0702_0703_0704,
        0x1001_1002_1003_1004,
        0x1101_1102_1103_1104,
        0x1201_1202_1203_1204,
        0x1301_1302_1303_1304,
        0x1401_1402_1403_1404,
        0x1501_1502_1503_1504,
        0x1601_1602_1603_1604,
        0x1701_1702_1703_1704,
        0x2001_2002_2003_2004,
        0x2101_2102_2103_2104,
        0x2201_2202_2203_2204,
        0x2301_2302_2303_2304,
        0x2401_2402_2403_2404,
        0x2501_2502_2503_2504,
        0x2601_2602_2603_2604,
        0x2701_2702_2703_2704,
        0x3001_3002_3003_3004,
        0x3101_3102_3103_3104,
        0x3201_3202_3203_3204,
        0x3301_3302_3303_3304,
        0x3401_3402_3403_3404,
        0x3501_3502_3503_3504,
        0x3601_3602_3603_3604,
        0x3701_3702_3703_3704,
    ]);
    let mut portable_xoshiro = PortableXoshiro256PlusX8::from_seed(PortableXoshiro256PlusX8Seed::from(xoshiro_seed));
    let mut specific_xoshiro = SpecificXoshiro256PlusX8::from_seed(SpecificXoshiro256PlusX8Seed::from(xoshiro_seed));

    assert_same_vectors(
        || portable_xoshiro.next_u64x8().to_array(),
        || *specific_xoshiro.next_u64x8(),
    );
}
