use core::{
    arch::x86_64::*,
    mem,
    ops::{Deref, DerefMut},
};

use rand_core::{RngCore, SeedableRng, TryRngCore};

use crate::biski64::{FAST_LOOP_INCREMENT, seed_from_bytes, seed_state, seed_stream_states};

use super::{rotate_left, simdrand::*};

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

#[repr(align(32))]
pub struct Biski64X4 {
    fast_loop: __m256i,
    mix: __m256i,
    loop_mix: __m256i,
}

impl SeedableRng for Biski64X4 {
    type Seed = Biski64X4Seed;

    fn from_seed(seed: Self::Seed) -> Self {
        const SIZE: usize = mem::size_of::<u64>();
        const LEN: usize = 4;
        assert!(seed.len() == SIZE * LEN);

        let (chunks, remainder) = seed[..].as_chunks::<8>();
        assert!(remainder.is_empty());
        assert_eq!(chunks.len(), LEN);

        let mut seed_words = [0; LEN];
        for (dst, chunk) in seed_words.iter_mut().zip(chunks) {
            *dst = u64::from_le_bytes(*chunk);
        }
        let seeded_state = seed_words.map(seed_state);

        Self {
            fast_loop: pack_u64x4(seeded_state.map(|state| state[0])),
            mix: pack_u64x4(seeded_state.map(|state| state[1])),
            loop_mix: pack_u64x4(seeded_state.map(|state| state[2])),
        }
    }

    fn seed_from_u64(seed: u64) -> Self {
        let seeded_state = seed_stream_states::<4>(seed);

        Self {
            fast_loop: pack_u64x4(seeded_state.map(|state| state[0])),
            mix: pack_u64x4(seeded_state.map(|state| state[1])),
            loop_mix: pack_u64x4(seeded_state.map(|state| state[2])),
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

impl SimdRand for Biski64X4 {
    #[inline(always)]
    fn next_m256i(&mut self) -> __m256i {
        unsafe {
            let fast_loop = self.fast_loop;
            let mix = self.mix;
            let loop_mix = self.loop_mix;

            self.fast_loop = _mm256_add_epi64(fast_loop, _mm256_set1_epi64x(FAST_LOOP_INCREMENT.cast_signed()));
            self.mix = _mm256_add_epi64(rotate_left::<16>(mix), rotate_left::<40>(loop_mix));
            self.loop_mix = _mm256_xor_si256(fast_loop, mix);

            _mm256_add_epi64(mix, loop_mix)
        }
    }
}

#[inline(always)]
fn pack_u64x4(values: [u64; 4]) -> __m256i {
    unsafe {
        _mm256_set_epi64x(
            values[3].cast_signed(),
            values[2].cast_signed(),
            values[1].cast_signed(),
            values[0].cast_signed(),
        )
    }
}

#[cfg(test)]
mod tests {
    use rand_core::SeedableRng;

    use super::{Biski64X4, SimdRand};
    use crate::biski64::{FixedBytesRng, assert_rngs_match};

    #[test]
    fn try_from_rng_matches_from_rng() {
        let seed = [7u8; 32];

        assert_rngs_match::<4, _>(
            Biski64X4::from_rng(&mut FixedBytesRng::new(seed)),
            Biski64X4::try_from_rng(&mut FixedBytesRng::new(seed)).unwrap(),
            |rng| *rng.next_u64x4(),
        );
    }
}
