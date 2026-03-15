use core::{
    arch::x86_64::*,
    mem,
    ops::{Deref, DerefMut},
};

use rand_core::{RngCore, SeedableRng, TryRngCore};

use crate::biski64::{FAST_LOOP_INCREMENT, seed_state, seed_stream_states};

use super::simdrand::*;

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

#[repr(align(64))]
pub struct Biski64X8 {
    fast_loop: __m512i,
    mix: __m512i,
    loop_mix: __m512i,
}

impl SeedableRng for Biski64X8 {
    type Seed = Biski64X8Seed;

    fn from_seed(seed: Self::Seed) -> Self {
        const SIZE: usize = mem::size_of::<u64>();
        const LEN: usize = 8;
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
            fast_loop: pack_u64x8(seeded_state.map(|state| state[0])),
            mix: pack_u64x8(seeded_state.map(|state| state[1])),
            loop_mix: pack_u64x8(seeded_state.map(|state| state[2])),
        }
    }

    fn seed_from_u64(seed: u64) -> Self {
        let seeded_state = seed_stream_states::<8>(seed);

        Self {
            fast_loop: pack_u64x8(seeded_state.map(|state| state[0])),
            mix: pack_u64x8(seeded_state.map(|state| state[1])),
            loop_mix: pack_u64x8(seeded_state.map(|state| state[2])),
        }
    }

    fn from_rng(rng: &mut impl RngCore) -> Self {
        Self::seed_from_u64(rng.next_u64())
    }

    fn try_from_rng<R: TryRngCore>(rng: &mut R) -> Result<Self, R::Error> {
        Ok(Self::seed_from_u64(rng.try_next_u64()?))
    }
}

impl SimdRand for Biski64X8 {
    #[inline(always)]
    fn next_m512i(&mut self) -> __m512i {
        unsafe {
            let fast_loop = self.fast_loop;
            let mix = self.mix;
            let loop_mix = self.loop_mix;

            self.fast_loop = _mm512_add_epi64(fast_loop, _mm512_set1_epi64(FAST_LOOP_INCREMENT.cast_signed()));
            self.mix = _mm512_add_epi64(_mm512_rol_epi64::<16>(mix), _mm512_rol_epi64::<40>(loop_mix));
            self.loop_mix = _mm512_xor_si512(fast_loop, mix);

            _mm512_add_epi64(mix, loop_mix)
        }
    }
}

#[inline(always)]
fn pack_u64x8(values: [u64; 8]) -> __m512i {
    unsafe {
        _mm512_set_epi64(
            values[7].cast_signed(),
            values[6].cast_signed(),
            values[5].cast_signed(),
            values[4].cast_signed(),
            values[3].cast_signed(),
            values[2].cast_signed(),
            values[1].cast_signed(),
            values[0].cast_signed(),
        )
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand_core::{RngCore, SeedableRng};

    use crate::testutil::{DOUBLE_RANGE, REF_SEED_BISKI64_X8, biski64_reference_sequence, test_uniform_distribution};

    use super::super::vecs::*;
    use super::*;

    type RngSeed = Biski64X8Seed;
    type RngImpl = Biski64X8;

    #[test]
    fn reference() {
        let seed: RngSeed = REF_SEED_BISKI64_X8.into();
        let mut rng = RngImpl::from_seed(seed);

        for &expected in &biski64_reference_sequence::<10>(1) {
            let mem = rng.next_u64x8();
            for &value in &*mem {
                assert_eq!(value, expected);
            }
        }
    }

    #[test]
    fn sample_u64x8() {
        let mut seed = RngSeed::default();
        rand::rng().fill_bytes(&mut *seed);
        let mut rng = RngImpl::from_seed(seed);

        let values = rng.next_u64x8();

        assert!(values.iter().all(|&value| value != 0));
        assert!(values.iter().unique().count() == values.len());
        println!("{values:?}");

        let values = rng.next_u64x8();

        assert!(values.iter().all(|&value| value != 0));
        assert!(values.iter().unique().count() == values.len());
        println!("{values:?}");
    }

    #[test]
    fn sample_f64x8() {
        let mut seed = RngSeed::default();
        rand::rng().fill_bytes(&mut *seed);
        let mut rng = RngImpl::from_seed(seed);

        let values = rng.next_f64x8();

        assert!(values.iter().all(|&value| value != 0.0));
        println!("{values:?}");

        let values = rng.next_f64x8();

        assert!(values.iter().all(|&value| value != 0.0));
        println!("{values:?}");
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore = "distribution test requires release mode")]
    fn sample_f64x8_distribution() {
        let mut seed = RngSeed::default();
        rand::rng().fill_bytes(&mut *seed);
        let mut rng = RngImpl::from_seed(seed);

        let mut current: Option<F64x8> = None;
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
