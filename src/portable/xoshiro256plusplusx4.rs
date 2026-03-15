use core::{
    mem,
    ops::{Deref, DerefMut},
    simd::u64x4,
};

use rand_core::SeedableRng;

use super::{SimdRandX4, read_u64_into_vec, rotate_left};

#[derive(Clone)]
pub struct Xoshiro256PlusPlusX4Seed([u8; 128]);

impl Xoshiro256PlusPlusX4Seed {
    #[must_use]
    pub const fn new(seed: [u8; 128]) -> Self {
        Self(seed)
    }
}

impl From<[u8; 128]> for Xoshiro256PlusPlusX4Seed {
    fn from(val: [u8; 128]) -> Self {
        Self::new(val)
    }
}

impl From<&[u8]> for Xoshiro256PlusPlusX4Seed {
    fn from(val: &[u8]) -> Self {
        assert_eq!(val.len(), 128);
        let mut seed = [0u8; 128];
        seed.copy_from_slice(val);
        Self::new(seed)
    }
}

impl Deref for Xoshiro256PlusPlusX4Seed {
    type Target = [u8; 128];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for Xoshiro256PlusPlusX4Seed {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Default for Xoshiro256PlusPlusX4Seed {
    fn default() -> Self {
        Self([0; 128])
    }
}

impl AsRef<[u8]> for Xoshiro256PlusPlusX4Seed {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl AsMut<[u8]> for Xoshiro256PlusPlusX4Seed {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

pub struct Xoshiro256PlusPlusX4 {
    s0: u64x4,
    s1: u64x4,
    s2: u64x4,
    s3: u64x4,
}

impl SeedableRng for Xoshiro256PlusPlusX4 {
    type Seed = Xoshiro256PlusPlusX4Seed;

    #[allow(clippy::identity_op, clippy::erasing_op)]
    fn from_seed(seed: Self::Seed) -> Self {
        const SIZE: usize = mem::size_of::<u64>();
        const LEN: usize = u64x4::LEN;
        const VECSIZE: usize = SIZE * LEN;

        let s0 = read_u64_into_vec(&seed[(VECSIZE * 0)..(VECSIZE * 1)]);
        let s1 = read_u64_into_vec(&seed[(VECSIZE * 1)..(VECSIZE * 2)]);
        let s2 = read_u64_into_vec(&seed[(VECSIZE * 2)..(VECSIZE * 3)]);
        let s3 = read_u64_into_vec(&seed[(VECSIZE * 3)..(VECSIZE * 4)]);

        Self { s0, s1, s2, s3 }
    }
}

impl SimdRandX4 for Xoshiro256PlusPlusX4 {
    fn next_u64x4(&mut self) -> u64x4 {
        let result = rotate_left(self.s0 + self.s3, 23) + self.s0;

        let t = self.s1 << u64x4::splat(17);

        self.s2 ^= self.s0;
        self.s3 ^= self.s1;
        self.s1 ^= self.s2;
        self.s0 ^= self.s3;

        self.s2 ^= t;

        self.s3 = rotate_left(self.s3, 45);

        result
    }
}
