use std::{
    arch::x86_64::*,
    mem,
    ops::{Deref, DerefMut},
};

use rand_core::SeedableRng;

use crate::specific::avx512::read_u64_into_vec;

use super::{simdprng::*, rotate_left};
use super::vecs::*;

pub struct Xoshiro256PlusX8Seed([u8; 256]);

impl Xoshiro256PlusX8Seed {
    pub fn new(seed: [u8; 256]) -> Self {
        Self(seed)
    }
}

impl Into<Xoshiro256PlusX8Seed> for [u8; 256] {
    fn into(self) -> Xoshiro256PlusX8Seed {
        Xoshiro256PlusX8Seed::new(self)
    }
}

impl Into<Xoshiro256PlusX8Seed> for Vec<u8> {
    fn into(self) -> Xoshiro256PlusX8Seed {
        assert!(self.len() == 256);
        Xoshiro256PlusX8Seed::new(self.try_into().unwrap())
    }
}

impl Deref for Xoshiro256PlusX8Seed {
    type Target = [u8; 256];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for Xoshiro256PlusX8Seed {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[repr(align(64))]
pub struct Xoshiro256PlusX8 {
    s0: __m512i,
    s1: __m512i,
    s2: __m512i,
    s3: __m512i,
}
impl Default for Xoshiro256PlusX8Seed {
    fn default() -> Xoshiro256PlusX8Seed {
        Xoshiro256PlusX8Seed([0; 256])
    }
}

impl AsMut<[u8]> for Xoshiro256PlusX8Seed {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl SeedableRng for Xoshiro256PlusX8 {
    type Seed = Xoshiro256PlusX8Seed;

    fn from_seed(seed: Self::Seed) -> Self {
        const SIZE: usize = mem::size_of::<u64>();
        const LEN: usize = 8;
        const VECSIZE: usize = SIZE * LEN;
        // TODO: implement "jumps" between lanes?
        unsafe {
            let mut s0: __m512i = _mm512_setzero_si512();
            let mut s1: __m512i = _mm512_setzero_si512();
            let mut s2: __m512i = _mm512_setzero_si512();
            let mut s3: __m512i = _mm512_setzero_si512();
            read_u64_into_vec(&seed[(VECSIZE * 0)..(VECSIZE * 1)], &mut s0);
            read_u64_into_vec(&seed[(VECSIZE * 1)..(VECSIZE * 2)], &mut s1);
            read_u64_into_vec(&seed[(VECSIZE * 2)..(VECSIZE * 3)], &mut s2);
            read_u64_into_vec(&seed[(VECSIZE * 3)..(VECSIZE * 4)], &mut s3);

            Self { s0, s1, s2, s3 }
        }
    }
}

impl SimdPrng for Xoshiro256PlusX8 {
    #[inline(always)]
    fn next_m512i(&mut self, vector: &mut __m512i) {
        unsafe {
            // const uint64_t result = s[0] + s[3];
            *vector = _mm512_add_epi64(self.s0, self.s3);

            // const uint64_t t = s[1] << 17;
            let t = _mm512_sll_epi64(self.s1, _mm_cvtsi32_si128(17));
        
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
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem;

    use itertools::Itertools;
    use num_traits::PrimInt;
    use rand_core::{SeedableRng, RngCore};
    use serial_test::parallel;

    use crate::testutil::{test_uniform_distribution, DOUBLE_RANGE, REF_SEED_512};

    use super::*;

    #[test]
    #[parallel]
    fn reference() {
        let seed: Xoshiro256PlusX8Seed = REF_SEED_512.into();
        let mut rng = Xoshiro256PlusX8::from_seed(seed);
        // These values were produced with the reference implementation:
        // http://xoshiro.di.unimi.it/xoshiro256plusplus.c
        #[rustfmt::skip]
        let expected = [
            5,
            211106232532999,
            211106635186183,
            9223759065350669058,
            9250833439874351877,
            13862484359527728515,
            2346507365006083650,
            1168864526675804870,
            34095955243042024,
            3466914240207415127,
        ];
        for &e in &expected {
            let mut mem = Default::default();
            rng.next_u64x8(&mut mem);
            for v in mem.into_iter() {
                assert_eq!(v, e);
            }
        }
    }
}