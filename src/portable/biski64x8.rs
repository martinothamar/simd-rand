use core::{
    mem,
    ops::{Deref, DerefMut},
    simd::u64x8,
};

use rand_core::{RngCore, SeedableRng, TryRngCore};

use crate::biski64::{FAST_LOOP_INCREMENT, seed_from_bytes, seed_state, seed_stream_states};

use super::{SimdRandX8, read_u64_array, rotate_left};

const INCREMENT: u64x8 = u64x8::from_array([FAST_LOOP_INCREMENT; 8]);

#[derive(Clone)]
pub struct Biski64X8Seed([u8; 64]);

impl Biski64X8Seed {
    #[must_use]
    pub const fn new(seed: [u8; 64]) -> Self {
        Self(seed)
    }
}

impl From<[u8; 64]> for Biski64X8Seed {
    fn from(val: [u8; 64]) -> Self {
        Self::new(val)
    }
}

impl From<&[u8]> for Biski64X8Seed {
    fn from(val: &[u8]) -> Self {
        assert_eq!(val.len(), 64);
        let mut seed = [0u8; 64];
        seed.copy_from_slice(val);
        Self::new(seed)
    }
}

impl Deref for Biski64X8Seed {
    type Target = [u8; 64];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Biski64X8Seed {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Default for Biski64X8Seed {
    fn default() -> Self {
        Self([0; 64])
    }
}

impl AsRef<[u8]> for Biski64X8Seed {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl AsMut<[u8]> for Biski64X8Seed {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

pub struct Biski64X8 {
    fast_loop: u64x8,
    mix: u64x8,
    loop_mix: u64x8,
}

impl SeedableRng for Biski64X8 {
    type Seed = Biski64X8Seed;

    fn from_seed(seed: Self::Seed) -> Self {
        const SIZE: usize = mem::size_of::<u64>();
        const LEN: usize = u64x8::LEN;
        assert!(seed.len() == SIZE * LEN);

        let seed_words = read_u64_array::<8>(&seed[..]);
        let seeded_state = seed_words.map(seed_state);

        Self {
            fast_loop: u64x8::from_array(seeded_state.map(|state| state[0])),
            mix: u64x8::from_array(seeded_state.map(|state| state[1])),
            loop_mix: u64x8::from_array(seeded_state.map(|state| state[2])),
        }
    }

    fn seed_from_u64(seed: u64) -> Self {
        let seeded_state = seed_stream_states::<8>(seed);

        Self {
            fast_loop: u64x8::from_array(seeded_state.map(|state| state[0])),
            mix: u64x8::from_array(seeded_state.map(|state| state[1])),
            loop_mix: u64x8::from_array(seeded_state.map(|state| state[2])),
        }
    }

    fn from_rng(rng: &mut impl RngCore) -> Self {
        let mut seed = Self::Seed::default();
        rng.fill_bytes(seed.as_mut());
        Self::seed_from_u64(seed_from_bytes(seed.as_ref()))
    }

    fn try_from_rng<R: TryRngCore>(rng: &mut R) -> Result<Self, R::Error> {
        let mut seed = Self::Seed::default();
        rng.try_fill_bytes(seed.as_mut())?;
        Ok(Self::seed_from_u64(seed_from_bytes(seed.as_ref())))
    }
}

impl SimdRandX8 for Biski64X8 {
    fn next_u64x8(&mut self) -> u64x8 {
        let fast_loop = self.fast_loop;
        let mix = self.mix;
        let loop_mix = self.loop_mix;

        self.fast_loop = fast_loop + INCREMENT;
        self.mix = rotate_left(mix, 16) + rotate_left(loop_mix, 40);
        self.loop_mix = fast_loop ^ mix;

        mix + loop_mix
    }
}

#[cfg(test)]
mod tests {
    use core::simd::*;
    use itertools::Itertools;
    use rand_core::SeedableRng;

    use crate::{
        biski64::seed_from_bytes,
        testutil::{
            DOUBLE_RANGE, REF_SEED_BISKI64_X8, biski64_parallel_reference_vectors, biski64_reference_sequence,
            fixed_u64_rng::FixedBytesRng, test_uniform_distribution,
        },
    };

    use super::*;

    type RngSeed = Biski64X8Seed;
    type RngImpl = Biski64X8;

    #[test]
    fn reference() {
        let seed: RngSeed = REF_SEED_BISKI64_X8.into();
        let mut rng = RngImpl::from_seed(seed);

        for expected in biski64_reference_sequence::<10>(1) {
            let mem = rng.next_u64x8();
            for &value in mem.as_array() {
                assert_eq!(value, expected);
            }
        }
    }

    #[test]
    fn sample_u64x8() {
        let mut seed = RngSeed::default();
        rand::rng().fill_bytes(&mut *seed);
        let mut rng = RngImpl::from_seed(seed);

        let values = *rng.next_u64x8().as_array();

        assert!(values.iter().all(|&value| value != 0));
        assert!(values.iter().unique().count() == values.len());
        println!("{values:?}");

        let values = *rng.next_u64x8().as_array();

        assert!(values.iter().all(|&value| value != 0));
        assert!(values.iter().unique().count() == values.len());
        println!("{values:?}");
    }

    #[test]
    fn sample_f64x8() {
        let mut seed = RngSeed::default();
        rand::rng().fill_bytes(&mut *seed);
        let mut rng = RngImpl::from_seed(seed);

        let values = *rng.next_f64x8().as_array();

        assert!(values.iter().all(|&value| value != 0.0));
        println!("{values:?}");

        let values = *rng.next_f64x8().as_array();

        assert!(values.iter().all(|&value| value != 0.0));
        println!("{values:?}");
    }

    #[test]
    fn seed_from_u64_matches_upstream_parallel_streams() {
        let mut rng = RngImpl::seed_from_u64(42);

        for expected in biski64_parallel_reference_vectors::<8, 10>(42) {
            assert_eq!(rng.next_u64x8().to_array(), expected);
        }
    }

    #[test]
    fn from_rng_matches_upstream_parallel_streams() {
        let seed = [
            0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x28, 0x27,
            0x26, 0x25, 0x24, 0x23, 0x22, 0x21, 0x38, 0x37, 0x36, 0x35, 0x34, 0x33, 0x32, 0x31, 0x48, 0x47, 0x46, 0x45,
            0x44, 0x43, 0x42, 0x41, 0x58, 0x57, 0x56, 0x55, 0x54, 0x53, 0x52, 0x51, 0x68, 0x67, 0x66, 0x65, 0x64, 0x63,
            0x62, 0x61, 0x78, 0x77, 0x76, 0x75, 0x74, 0x73, 0x72, 0x71,
        ];
        let mut seed_rng = FixedBytesRng::new(seed);
        let mut rng = RngImpl::from_rng(&mut seed_rng);
        let master_seed = seed_from_bytes(&seed);

        for expected in biski64_parallel_reference_vectors::<8, 10>(master_seed) {
            assert_eq!(rng.next_u64x8().to_array(), expected);
        }
    }

    #[test]
    #[cfg_attr(
        any(debug_assertions, miri),
        ignore = "distribution test requires release mode and real RNG"
    )]
    fn sample_f64x8_distribution() {
        let mut seed = RngSeed::default();
        rand::rng().fill_bytes(&mut *seed);
        let mut rng = RngImpl::from_seed(seed);

        let mut current: Option<f64x8> = None;
        let mut current_index: usize = 0;

        test_uniform_distribution::<10_000_000, f64>(
            || match &current {
                Some(vector) if current_index < 8 => {
                    let result = vector[current_index];
                    current_index += 1;
                    result
                }
                _ => {
                    current_index = 0;
                    let vector = rng.next_f64x8();
                    let result = vector[current_index];
                    current = Some(vector);
                    current_index += 1;
                    result
                }
            },
            DOUBLE_RANGE,
        );
    }
}
