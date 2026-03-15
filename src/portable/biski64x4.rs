use core::{
    mem,
    ops::{Deref, DerefMut},
    simd::u64x4,
};

use rand_core::{RngCore, SeedableRng, TryRngCore};

use crate::biski64::{FAST_LOOP_INCREMENT, seed_from_bytes, seed_state, seed_stream_states};

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

    fn seed_from_u64(seed: u64) -> Self {
        let seeded_state = seed_stream_states::<4>(seed);

        Self {
            fast_loop: u64x4::from_array(seeded_state.map(|state| state[0])),
            mix: u64x4::from_array(seeded_state.map(|state| state[1])),
            loop_mix: u64x4::from_array(seeded_state.map(|state| state[2])),
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
    use rand_core::SeedableRng;

    use super::{Biski64X4, Biski64X4Seed, SimdRandX4};
    use crate::biski64::{FixedBytesRng, parallel_reference_vectors, reference_sequence, seed_from_bytes};

    #[test]
    fn seed_from_u64_matches_upstream_parallel_streams() {
        let mut rng = Biski64X4::seed_from_u64(42);

        for expected in parallel_reference_vectors::<4, 10>(42) {
            assert_eq!(rng.next_u64x4().to_array(), expected);
        }
    }

    #[test]
    fn from_rng_matches_upstream_parallel_streams() {
        let seed = [
            0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x28, 0x27,
            0x26, 0x25, 0x24, 0x23, 0x22, 0x21, 0x38, 0x37, 0x36, 0x35, 0x34, 0x33, 0x32, 0x31,
        ];
        let mut rng = Biski64X4::from_rng(&mut FixedBytesRng::new(seed));
        let master_seed = seed_from_bytes(&seed);

        for expected in parallel_reference_vectors::<4, 10>(master_seed) {
            assert_eq!(rng.next_u64x4().to_array(), expected);
        }
    }

    #[test]
    fn from_rng_uses_seed_bytes_beyond_first_word() {
        let seed_a = [0u8; 32];
        let mut seed_b = seed_a;
        seed_b[31] = 1;

        let mut rng_a = Biski64X4::from_rng(&mut FixedBytesRng::new(seed_a));
        let mut rng_b = Biski64X4::from_rng(&mut FixedBytesRng::new(seed_b));

        assert_ne!(rng_a.next_u64x4().to_array(), rng_b.next_u64x4().to_array());
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

        let mut rng = Biski64X4::from_seed(Biski64X4Seed::from(seed));
        let vectors = [
            rng.next_u64x4().to_array(),
            rng.next_u64x4().to_array(),
            rng.next_u64x4().to_array(),
            rng.next_u64x4().to_array(),
        ];

        for (lane, seed_word) in seed_words.into_iter().enumerate() {
            assert_eq!(vectors.map(|vector| vector[lane]), reference_sequence::<4>(seed_word));
        }
    }
}
