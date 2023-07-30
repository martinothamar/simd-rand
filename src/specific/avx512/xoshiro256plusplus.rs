use std::{
    arch::x86_64::*,
    mem,
    ops::{Deref, DerefMut},
};

use rand_core::SeedableRng;

use crate::specific::avx512::read_u64_into_vec;

use super::{rotate_left, simdrand::*};

pub struct Xoshiro256PlusPlusX8Seed([u8; 256]);

impl Xoshiro256PlusPlusX8Seed {
    pub fn new(seed: [u8; 256]) -> Self {
        Self(seed)
    }
}

impl Into<Xoshiro256PlusPlusX8Seed> for [u8; 256] {
    fn into(self) -> Xoshiro256PlusPlusX8Seed {
        Xoshiro256PlusPlusX8Seed::new(self)
    }
}

impl Into<Xoshiro256PlusPlusX8Seed> for Vec<u8> {
    fn into(self) -> Xoshiro256PlusPlusX8Seed {
        assert!(self.len() == 256);
        Xoshiro256PlusPlusX8Seed::new(self.try_into().unwrap())
    }
}

impl Deref for Xoshiro256PlusPlusX8Seed {
    type Target = [u8; 256];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for Xoshiro256PlusPlusX8Seed {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[repr(align(64))]
pub struct Xoshiro256PlusPlusX8 {
    s0: __m512i,
    s1: __m512i,
    s2: __m512i,
    s3: __m512i,
}
impl Default for Xoshiro256PlusPlusX8Seed {
    fn default() -> Xoshiro256PlusPlusX8Seed {
        Xoshiro256PlusPlusX8Seed([0; 256])
    }
}

impl AsMut<[u8]> for Xoshiro256PlusPlusX8Seed {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl SeedableRng for Xoshiro256PlusPlusX8 {
    type Seed = Xoshiro256PlusPlusX8Seed;

    fn from_seed(seed: Self::Seed) -> Self {
        const SIZE: usize = mem::size_of::<u64>();
        const LEN: usize = 8;
        const VECSIZE: usize = SIZE * LEN;
        // TODO: implement "jumps" between lanes?
        
        let s0 = read_u64_into_vec(&seed[(VECSIZE * 0)..(VECSIZE * 1)]);
        let s1 = read_u64_into_vec(&seed[(VECSIZE * 1)..(VECSIZE * 2)]);
        let s2 = read_u64_into_vec(&seed[(VECSIZE * 2)..(VECSIZE * 3)]);
        let s3 = read_u64_into_vec(&seed[(VECSIZE * 3)..(VECSIZE * 4)]);

        Self { s0, s1, s2, s3 }
    }
}

impl SimdRand for Xoshiro256PlusPlusX8 {
    #[inline(always)]
    fn next_m512i(&mut self) -> __m512i {
        unsafe {
            // const uint64_t result = rotl(s[0] + s[3], 23) + s[0];
            let vector = _mm512_add_epi64(rotate_left::<23>(_mm512_add_epi64(self.s0, self.s3)), self.s0);

            // const uint64_t t = s[1] << 17;
            let t = _mm512_slli_epi64::<17>(self.s1);

            // s[2] ^= s[0];
            // s[3] ^= s[1];
            // s[1] ^= s[2];
            // s[0] ^= s[3];
            self.s2 = _mm512_xor_si512(self.s2, self.s0);
            self.s3 = _mm512_xor_si512(self.s3, self.s1);
            self.s1 = _mm512_xor_si512(self.s1, self.s2);
            self.s0 = _mm512_xor_si512(self.s0, self.s3);

            // s[2] ^= t;
            self.s2 = _mm512_xor_si512(self.s2, t);

            // s[3] = rotl(s[3], 45);
            self.s3 = rotate_left::<45>(self.s3);

            vector
        }
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand_core::{RngCore, SeedableRng};
    use serial_test::parallel;

    use crate::testutil::{test_uniform_distribution, DOUBLE_RANGE, REF_SEED_512};

    use super::super::vecs::*;
    use super::*;

    type RngSeed = Xoshiro256PlusPlusX8Seed;
    type RngImpl = Xoshiro256PlusPlusX8;

    #[test]
    #[parallel]
    fn reference() {
        let seed: RngSeed = REF_SEED_512.into();
        let mut rng = RngImpl::from_seed(seed);
        // These values were produced with the reference implementation:
        // http://xoshiro.di.unimi.it/xoshiro256plusplus.c
        #[rustfmt::skip]
        let expected = [
            41943041, 58720359, 3588806011781223, 3591011842654386,
            9228616714210784205, 9973669472204895162, 14011001112246962877,
            12406186145184390807, 15849039046786891736, 10450023813501588000,
        ];
        for &e in &expected {
            let mem = rng.next_u64x8();
            for v in mem.into_iter() {
                assert_eq!(v, e);
            }
        }
    }

    #[test]
    #[parallel]
    fn sample_u64x8() {
        let mut seed: RngSeed = Default::default();
        rand::thread_rng().fill_bytes(&mut *seed);
        let mut rng = RngImpl::from_seed(seed);

        let values = rng.next_u64x8();

        assert!(values.iter().all(|&v| v != 0));
        assert!(values.iter().unique().count() == values.len());
        println!("{values:?}");

        let values = rng.next_u64x8();

        assert!(values.iter().all(|&v| v != 0));
        assert!(values.iter().unique().count() == values.len());
        println!("{values:?}");
    }

    #[test]
    #[parallel]
    fn sample_f64x8() {
        let mut seed: RngSeed = Default::default();
        rand::thread_rng().fill_bytes(&mut *seed);
        let mut rng = RngImpl::from_seed(seed);

        let values = rng.next_f64x8();

        assert!(values.iter().all(|&v| v != 0.0));
        println!("{values:?}");

        let values = rng.next_f64x8();

        assert!(values.iter().all(|&v| v != 0.0));
        println!("{values:?}");
    }

    #[test]
    #[parallel]
    fn sample_f64x8_distribution() {
        let mut seed: RngSeed = Default::default();
        rand::thread_rng().fill_bytes(&mut *seed);
        let mut rng = RngImpl::from_seed(seed);

        let mut current: Option<F64x8> = None;
        let mut current_index: usize = 0;

        test_uniform_distribution::<10_000_000, f64>(
            || match &current {
                Some(vector) if current_index < 8 => {
                    let result = vector[current_index];
                    current_index += 1;
                    return result;
                }
                _ => {
                    current_index = 0;
                    let vector = rng.next_f64x8();
                    let result = vector[current_index];
                    current = Some(vector);
                    current_index += 1;
                    return result;
                }
            },
            DOUBLE_RANGE,
        );
    }
}
