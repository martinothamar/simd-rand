use rand_core::SeedableRng;

struct FixedBytesRng<const N: usize> {
    bytes: [u8; N],
    offset: usize,
}

impl<const N: usize> FixedBytesRng<N> {
    const fn new(bytes: [u8; N]) -> Self {
        Self { bytes, offset: 0 }
    }
}

impl<const N: usize> rand_core::RngCore for FixedBytesRng<N> {
    fn next_u32(&mut self) -> u32 {
        rand_core::impls::next_u32_via_fill(self)
    }

    fn next_u64(&mut self) -> u64 {
        rand_core::impls::next_u64_via_fill(self)
    }

    fn fill_bytes(&mut self, dst: &mut [u8]) {
        let end = self.offset + dst.len();
        assert!(end <= self.bytes.len());
        dst.copy_from_slice(&self.bytes[self.offset..end]);
        self.offset = end;
    }
}

macro_rules! for_each_avx2_from_seed_case {
    ($m:ident) => {
        $m!(
            asymmetric_seed_32(),
            crate::portable::FrandX4,
            crate::portable::FrandX4Seed,
            |rng: &mut crate::portable::FrandX4| rng.next_u64x4().to_array(),
            crate::specific::avx2::FrandX4,
            crate::specific::avx2::FrandX4Seed,
            |rng: &mut crate::specific::avx2::FrandX4| *rng.next_u64x4()
        );
        $m!(
            asymmetric_seed_32(),
            crate::portable::Biski64X4,
            crate::portable::Biski64X4Seed,
            |rng: &mut crate::portable::Biski64X4| rng.next_u64x4().to_array(),
            crate::specific::avx2::Biski64X4,
            crate::specific::avx2::Biski64X4Seed,
            |rng: &mut crate::specific::avx2::Biski64X4| *rng.next_u64x4()
        );
        $m!(
            asymmetric_seed_128(),
            crate::portable::Xoshiro256PlusX4,
            crate::portable::Xoshiro256PlusX4Seed,
            |rng: &mut crate::portable::Xoshiro256PlusX4| rng.next_u64x4().to_array(),
            crate::specific::avx2::Xoshiro256PlusX4,
            crate::specific::avx2::Xoshiro256PlusX4Seed,
            |rng: &mut crate::specific::avx2::Xoshiro256PlusX4| *rng.next_u64x4()
        );
        $m!(
            asymmetric_seed_128(),
            crate::portable::Xoshiro256PlusPlusX4,
            crate::portable::Xoshiro256PlusPlusX4Seed,
            |rng: &mut crate::portable::Xoshiro256PlusPlusX4| rng.next_u64x4().to_array(),
            crate::specific::avx2::Xoshiro256PlusPlusX4,
            crate::specific::avx2::Xoshiro256PlusPlusX4Seed,
            |rng: &mut crate::specific::avx2::Xoshiro256PlusPlusX4| *rng.next_u64x4()
        );
    };
}

macro_rules! for_each_avx512_from_seed_case {
    ($m:ident) => {
        $m!(
            asymmetric_seed_64(),
            crate::portable::FrandX8,
            crate::portable::FrandX8Seed,
            |rng: &mut crate::portable::FrandX8| rng.next_u64x8().to_array(),
            crate::specific::avx512::FrandX8,
            crate::specific::avx512::FrandX8Seed,
            |rng: &mut crate::specific::avx512::FrandX8| *rng.next_u64x8()
        );
        $m!(
            asymmetric_seed_64(),
            crate::portable::Biski64X8,
            crate::portable::Biski64X8Seed,
            |rng: &mut crate::portable::Biski64X8| rng.next_u64x8().to_array(),
            crate::specific::avx512::Biski64X8,
            crate::specific::avx512::Biski64X8Seed,
            |rng: &mut crate::specific::avx512::Biski64X8| *rng.next_u64x8()
        );
        $m!(
            asymmetric_seed_256(),
            crate::portable::Xoshiro256PlusX8,
            crate::portable::Xoshiro256PlusX8Seed,
            |rng: &mut crate::portable::Xoshiro256PlusX8| rng.next_u64x8().to_array(),
            crate::specific::avx512::Xoshiro256PlusX8,
            crate::specific::avx512::Xoshiro256PlusX8Seed,
            |rng: &mut crate::specific::avx512::Xoshiro256PlusX8| *rng.next_u64x8()
        );
        $m!(
            asymmetric_seed_256(),
            crate::portable::Xoshiro256PlusPlusX8,
            crate::portable::Xoshiro256PlusPlusX8Seed,
            |rng: &mut crate::portable::Xoshiro256PlusPlusX8| rng.next_u64x8().to_array(),
            crate::specific::avx512::Xoshiro256PlusPlusX8,
            crate::specific::avx512::Xoshiro256PlusPlusX8Seed,
            |rng: &mut crate::specific::avx512::Xoshiro256PlusPlusX8| *rng.next_u64x8()
        );
    };
}

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

fn asymmetric_seed_words<const WORDS: usize>() -> [u64; WORDS] {
    let mut words = [0u64; WORDS];

    for (index, word) in words.iter_mut().enumerate() {
        // Keep every 64-bit word distinct so lane-order and state-word mixups are visible.
        let block = (index / 4) as u64 + 1;
        let lane = (index % 4) as u64 + 1;
        *word = (block << 48) | (lane << 32) | (block << 16) | lane;
    }

    words
}

fn asymmetric_seed_32() -> [u8; 32] {
    fill_seed_32(&asymmetric_seed_words::<4>())
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

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl"
))]
fn asymmetric_seed_64() -> [u8; 64] {
    fill_seed_64(&asymmetric_seed_words::<8>())
}

fn fill_seed_128(words: &[u64; 16]) -> [u8; 128] {
    let mut seed = [0u8; 128];

    for (index, word) in words.iter().enumerate() {
        seed[(index * 8)..((index + 1) * 8)].copy_from_slice(&word.to_le_bytes());
    }

    seed
}

fn asymmetric_seed_128() -> [u8; 128] {
    fill_seed_128(&asymmetric_seed_words::<16>())
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

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl"
))]
fn asymmetric_seed_256() -> [u8; 256] {
    fill_seed_256(&asymmetric_seed_words::<32>())
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[test]
fn avx2_matches_portable_for_asymmetric_seeds() {
    use crate::portable::{Biski64X4 as PortableBiski64X4, SimdRandX4};
    use crate::specific::avx2::{Biski64X4 as SpecificBiski64X4, SimdRand};

    macro_rules! assert_from_seed_case {
        ($seed:expr, $portable_ty:path, $portable_seed:path, $portable_next:expr, $specific_ty:path, $specific_seed:path, $specific_next:expr) => {{
            let seed = $seed;
            let mut portable = <$portable_ty>::from_seed(<$portable_seed>::from(seed));
            let mut specific = <$specific_ty>::from_seed(<$specific_seed>::from(seed));
            let portable_next = $portable_next;
            let specific_next = $specific_next;

            assert_same_vectors(|| portable_next(&mut portable), || specific_next(&mut specific));
        }};
    }

    for_each_avx2_from_seed_case!(assert_from_seed_case);

    let mut portable_biski_from_u64 = PortableBiski64X4::seed_from_u64(42);
    let mut specific_biski_from_u64 = SpecificBiski64X4::seed_from_u64(42);

    assert_same_vectors(
        || portable_biski_from_u64.next_u64x4().to_array(),
        || *specific_biski_from_u64.next_u64x4(),
    );

    let lane_seed = asymmetric_seed_32();
    let mut portable_biski_from_bytes = PortableBiski64X4::from_rng(&mut FixedBytesRng::new(lane_seed));
    let mut specific_biski_from_bytes = SpecificBiski64X4::from_rng(&mut FixedBytesRng::new(lane_seed));

    assert_same_vectors(
        || portable_biski_from_bytes.next_u64x4().to_array(),
        || *specific_biski_from_bytes.next_u64x4(),
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
    use crate::portable::{Biski64X8 as PortableBiski64X8, SimdRandX8};
    use crate::specific::avx512::{Biski64X8 as SpecificBiski64X8, SimdRand};

    macro_rules! assert_from_seed_case {
        ($seed:expr, $portable_ty:path, $portable_seed:path, $portable_next:expr, $specific_ty:path, $specific_seed:path, $specific_next:expr) => {{
            let seed = $seed;
            let mut portable = <$portable_ty>::from_seed(<$portable_seed>::from(seed));
            let mut specific = <$specific_ty>::from_seed(<$specific_seed>::from(seed));
            let portable_next = $portable_next;
            let specific_next = $specific_next;

            assert_same_vectors(|| portable_next(&mut portable), || specific_next(&mut specific));
        }};
    }

    for_each_avx512_from_seed_case!(assert_from_seed_case);

    let mut portable_biski_from_u64 = PortableBiski64X8::seed_from_u64(42);
    let mut specific_biski_from_u64 = SpecificBiski64X8::seed_from_u64(42);

    assert_same_vectors(
        || portable_biski_from_u64.next_u64x8().to_array(),
        || *specific_biski_from_u64.next_u64x8(),
    );

    let lane_seed = asymmetric_seed_64();
    let mut portable_biski_from_bytes = PortableBiski64X8::from_rng(&mut FixedBytesRng::new(lane_seed));
    let mut specific_biski_from_bytes = SpecificBiski64X8::from_rng(&mut FixedBytesRng::new(lane_seed));

    assert_same_vectors(
        || portable_biski_from_bytes.next_u64x8().to_array(),
        || *specific_biski_from_bytes.next_u64x8(),
    );
}
