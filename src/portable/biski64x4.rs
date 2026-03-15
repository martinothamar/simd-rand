use core::{
    mem,
    ops::{Deref, DerefMut},
    simd::u64x4,
};

use rand_core::SeedableRng;

use crate::biski64::{FAST_LOOP_INCREMENT, seed_state};

use super::{SimdRandX4, read_u64_array, rotate_left};

const INCREMENT: u64x4 = u64x4::from_array([FAST_LOOP_INCREMENT; 4]);

#[derive(Clone, Default)]
pub struct Biski64X4Seed([u8; 32]);

impl Biski64X4Seed {
    #[must_use]
    pub const fn new(seed: [u8; 32]) -> Self {
        Self(seed)
    }
}

impl From<[u8; 32]> for Biski64X4Seed {
    fn from(val: [u8; 32]) -> Self {
        Self::new(val)
    }
}

impl From<&[u8]> for Biski64X4Seed {
    fn from(val: &[u8]) -> Self {
        assert_eq!(val.len(), 32);
        let mut seed = [0u8; 32];
        seed.copy_from_slice(val);
        Self::new(seed)
    }
}

impl Deref for Biski64X4Seed {
    type Target = [u8; 32];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Biski64X4Seed {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl AsRef<[u8]> for Biski64X4Seed {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl AsMut<[u8]> for Biski64X4Seed {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

pub struct Biski64X4 {
    fast_loop: u64x4,
    mix: u64x4,
    loop_mix: u64x4,
}

impl SeedableRng for Biski64X4 {
    type Seed = Biski64X4Seed;

    fn from_seed(seed: Self::Seed) -> Self {
        const SIZE: usize = mem::size_of::<u64>();
        const LEN: usize = u64x4::LEN;
        assert!(seed.len() == SIZE * LEN);

        let seed_words = read_u64_array::<4>(&seed[..]);
        let seeded_state = seed_words.map(seed_state);

        Self {
            fast_loop: u64x4::from_array(seeded_state.map(|state| state[0])),
            mix: u64x4::from_array(seeded_state.map(|state| state[1])),
            loop_mix: u64x4::from_array(seeded_state.map(|state| state[2])),
        }
    }
}

impl SimdRandX4 for Biski64X4 {
    fn next_u64x4(&mut self) -> u64x4 {
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
    use rand_core::{RngCore, SeedableRng};

    use crate::testutil::{DOUBLE_RANGE, REF_SEED_BISKI64_X4, biski64_reference_sequence, test_uniform_distribution};

    use super::*;

    type RngSeed = Biski64X4Seed;
    type RngImpl = Biski64X4;

    #[test]
    fn reference() {
        let seed: RngSeed = REF_SEED_BISKI64_X4.into();
        let mut rng = RngImpl::from_seed(seed);

        for expected in biski64_reference_sequence::<10>(1) {
            let mem = rng.next_u64x4();
            for &value in mem.as_array() {
                assert_eq!(value, expected);
            }
        }
    }

    #[test]
    fn sample_u64x4() {
        let mut seed = RngSeed::default();
        rand::rng().fill_bytes(&mut *seed);
        let mut rng = RngImpl::from_seed(seed);

        let values = *rng.next_u64x4().as_array();

        assert!(values.iter().all(|&value| value != 0));
        assert!(values.iter().unique().count() == values.len());
        println!("{values:?}");

        let values = *rng.next_u64x4().as_array();

        assert!(values.iter().all(|&value| value != 0));
        assert!(values.iter().unique().count() == values.len());
        println!("{values:?}");
    }

    #[test]
    fn sample_f64x4() {
        let mut seed = RngSeed::default();
        rand::rng().fill_bytes(&mut *seed);
        let mut rng = RngImpl::from_seed(seed);

        let values = *rng.next_f64x4().as_array();

        assert!(values.iter().all(|&value| value != 0.0));
        println!("{values:?}");

        let values = *rng.next_f64x4().as_array();

        assert!(values.iter().all(|&value| value != 0.0));
        println!("{values:?}");
    }

    #[test]
    #[cfg_attr(
        any(debug_assertions, miri),
        ignore = "distribution test requires release mode and real RNG"
    )]
    fn sample_f64x4_distribution() {
        let mut seed = RngSeed::default();
        rand::rng().fill_bytes(&mut *seed);
        let mut rng = RngImpl::from_seed(seed);

        let mut current: Option<f64x4> = None;
        let mut current_index: usize = 0;

        test_uniform_distribution::<10_000_000, f64>(
            || match &current {
                Some(vector) if current_index < 4 => {
                    let result = vector[current_index];
                    current_index += 1;
                    result
                }
                _ => {
                    current_index = 0;
                    let vector = rng.next_f64x4();
                    let result = vector[current_index];
                    current = Some(vector);
                    current_index += 1;
                    result
                }
            },
            DOUBLE_RANGE,
        );
    }

    #[test]
    fn asymmetric_seeds_match_scalar_reference() {
        let seed_words = [
            0x0000000000000000_u64,
            0x0123456789ABCDEF_u64,
            0x1112131415161718_u64,
            0xFFFFFFFFFFFFFFFF_u64,
        ];
        let mut seed = [0u8; 32];
        for (index, word) in seed_words.iter().enumerate() {
            seed[(index * 8)..((index + 1) * 8)].copy_from_slice(&word.to_le_bytes());
        }

        let mut rng = RngImpl::from_seed(seed.into());
        let vectors = [
            rng.next_u64x4().to_array(),
            rng.next_u64x4().to_array(),
            rng.next_u64x4().to_array(),
            rng.next_u64x4().to_array(),
        ];

        for (lane, seed_word) in seed_words.into_iter().enumerate() {
            let expected = biski64_reference_sequence::<4>(seed_word);
            let actual = vectors.map(|vector| vector[lane]);
            assert_eq!(actual, expected);
        }
    }
}
