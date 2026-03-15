use core::{
    arch::x86_64::*,
    mem,
    ops::{Deref, DerefMut},
};

use rand_core::SeedableRng;

use crate::specific::avx2::read_u64_into_vec;

use super::{rotate_left, simdrand::*};

#[derive(Clone)]
pub struct Xoshiro256PlusX4Seed([u8; 128]);

impl Xoshiro256PlusX4Seed {
    #[must_use]
    pub const fn new(seed: [u8; 128]) -> Self {
        Self(seed)
    }
}

impl From<[u8; 128]> for Xoshiro256PlusX4Seed {
    fn from(val: [u8; 128]) -> Self {
        Self::new(val)
    }
}

impl From<&[u8]> for Xoshiro256PlusX4Seed {
    fn from(val: &[u8]) -> Self {
        assert_eq!(val.len(), 128);
        let mut seed = [0u8; 128];
        seed.copy_from_slice(val);
        Self::new(seed)
    }
}

impl Deref for Xoshiro256PlusX4Seed {
    type Target = [u8; 128];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for Xoshiro256PlusX4Seed {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Default for Xoshiro256PlusX4Seed {
    fn default() -> Self {
        Self([0; 128])
    }
}

impl AsRef<[u8]> for Xoshiro256PlusX4Seed {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl AsMut<[u8]> for Xoshiro256PlusX4Seed {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

#[repr(align(32))]
pub struct Xoshiro256PlusX4 {
    s0: __m256i,
    s1: __m256i,
    s2: __m256i,
    s3: __m256i,
}

impl SeedableRng for Xoshiro256PlusX4 {
    type Seed = Xoshiro256PlusX4Seed;

    #[allow(clippy::identity_op, clippy::erasing_op)]
    fn from_seed(seed: Self::Seed) -> Self {
        const SIZE: usize = mem::size_of::<u64>();
        const LEN: usize = 4;
        const VECSIZE: usize = SIZE * LEN;

        let s0 = read_u64_into_vec(&seed[(VECSIZE * 0)..(VECSIZE * 1)]);
        let s1 = read_u64_into_vec(&seed[(VECSIZE * 1)..(VECSIZE * 2)]);
        let s2 = read_u64_into_vec(&seed[(VECSIZE * 2)..(VECSIZE * 3)]);
        let s3 = read_u64_into_vec(&seed[(VECSIZE * 3)..(VECSIZE * 4)]);

        Self { s0, s1, s2, s3 }
    }
}

impl SimdRand for Xoshiro256PlusX4 {
    #[inline(always)]
    fn next_m256i(&mut self) -> __m256i {
        unsafe {
            let vector = _mm256_add_epi64(self.s0, self.s3);

            let t = _mm256_slli_epi64::<17>(self.s1);

            self.s2 = _mm256_xor_si256(self.s2, self.s0);
            self.s3 = _mm256_xor_si256(self.s3, self.s1);
            self.s1 = _mm256_xor_si256(self.s1, self.s2);
            self.s0 = _mm256_xor_si256(self.s0, self.s3);

            self.s2 = _mm256_xor_si256(self.s2, t);

            self.s3 = rotate_left::<45>(self.s3);

            vector
        }
    }
}
