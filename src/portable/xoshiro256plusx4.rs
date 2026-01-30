use std::{
    mem,
    ops::{Deref, DerefMut},
    simd::u64x4,
};

use rand_core::SeedableRng;

use super::{SimdRandX4, read_u64_into_vec, rotate_left};

#[derive(Clone)]
pub struct Xoshiro256PlusX4Seed([u8; 128]);

impl Xoshiro256PlusX4Seed {
    pub fn new(seed: [u8; 128]) -> Self {
        Self(seed)
    }
}

impl From<[u8; 128]> for Xoshiro256PlusX4Seed {
    fn from(val: [u8; 128]) -> Self {
        Xoshiro256PlusX4Seed::new(val)
    }
}

impl From<Vec<u8>> for Xoshiro256PlusX4Seed {
    fn from(val: Vec<u8>) -> Self {
        assert!(val.len() == 128);
        Xoshiro256PlusX4Seed::new(val.try_into().unwrap())
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
    fn default() -> Xoshiro256PlusX4Seed {
        Xoshiro256PlusX4Seed([0; 128])
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

pub struct Xoshiro256PlusX4 {
    s0: u64x4,
    s1: u64x4,
    s2: u64x4,
    s3: u64x4,
}

impl SeedableRng for Xoshiro256PlusX4 {
    type Seed = Xoshiro256PlusX4Seed;

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

impl SimdRandX4 for Xoshiro256PlusX4 {
    fn next_u64x4(&mut self) -> u64x4 {
        let result = self.s0 + self.s3;

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

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand_core::{RngCore, SeedableRng};
    use serial_test::parallel;
    use std::simd::*;

    use crate::testutil::{DOUBLE_RANGE, REF_SEED_256, test_uniform_distribution};

    use super::*;

    type RngSeed = Xoshiro256PlusX4Seed;
    type RngImpl = Xoshiro256PlusX4;

    #[test]
    #[parallel]
    fn reference() {
        let seed: RngSeed = REF_SEED_256.into();
        let mut rng = RngImpl::from_seed(seed);
        // These values were produced with the reference implementation:
        // http://xoshiro.di.unimi.it/xoshiro256plus.c
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
        for e in expected {
            let mem = rng.next_u64x4();
            for &v in mem.as_array().iter() {
                assert_eq!(v, e);
            }
        }
    }

    #[test]
    #[parallel]
    fn sample_u64x4() {
        let mut seed: RngSeed = Default::default();
        rand::rng().fill_bytes(&mut *seed);
        let mut rng = RngImpl::from_seed(seed);

        let values = *rng.next_u64x4().as_array();

        assert!(values.iter().all(|&v| v != 0));
        assert!(values.iter().unique().count() == values.len());
        println!("{values:?}");

        let values = *rng.next_u64x4().as_array();

        assert!(values.iter().all(|&v| v != 0));
        assert!(values.iter().unique().count() == values.len());
        println!("{values:?}");
    }

    #[test]
    #[parallel]
    fn sample_f64x4() {
        let mut seed: RngSeed = Default::default();
        rand::rng().fill_bytes(&mut *seed);
        let mut rng = RngImpl::from_seed(seed);

        let values = *rng.next_f64x4().as_array();

        assert!(values.iter().all(|&v| v != 0.0));
        println!("{values:?}");

        let values = *rng.next_f64x4().as_array();

        assert!(values.iter().all(|&v| v != 0.0));
        println!("{values:?}");
    }

    #[test]
    #[parallel]
    #[cfg_attr(miri, ignore)]
    fn sample_f64x4_distribution() {
        let mut seed: RngSeed = Default::default();
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
}
