use core::{
    arch::x86_64::*,
    mem,
    ops::{Deref, DerefMut},
};

use rand_core::SeedableRng;

use crate::specific::avx512::read_u64_into_vec;

use super::simdrand::*;

#[derive(Clone)]
pub struct Xoshiro256PlusX8Seed([u8; 256]);

impl Xoshiro256PlusX8Seed {
    #[must_use]
    pub const fn new(seed: [u8; 256]) -> Self {
        Self(seed)
    }
}

impl From<[u8; 256]> for Xoshiro256PlusX8Seed {
    fn from(val: [u8; 256]) -> Self {
        Self::new(val)
    }
}

impl From<&[u8]> for Xoshiro256PlusX8Seed {
    fn from(val: &[u8]) -> Self {
        assert_eq!(val.len(), 256);
        let mut seed = [0u8; 256];
        seed.copy_from_slice(val);
        Self::new(seed)
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
    fn default() -> Self {
        Self([0; 256])
    }
}

impl AsRef<[u8]> for Xoshiro256PlusX8Seed {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl AsMut<[u8]> for Xoshiro256PlusX8Seed {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl SeedableRng for Xoshiro256PlusX8 {
    type Seed = Xoshiro256PlusX8Seed;

    #[allow(clippy::identity_op, clippy::erasing_op)]
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

impl SimdRand for Xoshiro256PlusX8 {
    #[inline(always)]
    fn next_m512i(&mut self) -> __m512i {
        unsafe {
            // const uint64_t result = s[0] + s[3];
            let vector = _mm512_add_epi64(self.s0, self.s3);

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
            self.s3 = _mm512_rol_epi64::<45>(self.s3);

            vector
        }
    }
}
