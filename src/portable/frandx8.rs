use core::{
    mem,
    ops::{Deref, DerefMut},
    simd::u64x8,
};

use rand_core::SeedableRng;

use super::{SimdRandX8, read_u64_into_vec};

const INCREMENT: u64x8 = u64x8::from_array([12964901029718341801; 8]);
const MUL_XOR: u64x8 = u64x8::from_array([149988720821803190; 8]);
const SHIFT: u64x8 = u64x8::from_array([32; 8]);

#[derive(Clone)]
pub struct FrandX8Seed([u8; 64]);

impl FrandX8Seed {
    #[must_use]
    pub const fn new(seed: [u8; 64]) -> Self {
        Self(seed)
    }
}

impl From<[u8; 64]> for FrandX8Seed {
    fn from(val: [u8; 64]) -> Self {
        Self::new(val)
    }
}

impl From<&[u8]> for FrandX8Seed {
    fn from(val: &[u8]) -> Self {
        assert_eq!(val.len(), 64);
        let mut seed = [0u8; 64];
        seed.copy_from_slice(val);
        Self::new(seed)
    }
}

impl Deref for FrandX8Seed {
    type Target = [u8; 64];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for FrandX8Seed {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Default for FrandX8Seed {
    fn default() -> Self {
        Self([0; 64])
    }
}

impl AsRef<[u8]> for FrandX8Seed {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl AsMut<[u8]> for FrandX8Seed {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

pub struct FrandX8 {
    seed: u64x8,
}

impl SeedableRng for FrandX8 {
    type Seed = FrandX8Seed;

    fn from_seed(seed: Self::Seed) -> Self {
        const SIZE: usize = mem::size_of::<u64>();
        const LEN: usize = u64x8::LEN;
        assert!(seed.len() == SIZE * LEN);

        let s = read_u64_into_vec(&seed[..]);

        Self { seed: s }
    }
}

impl SimdRandX8 for FrandX8 {
    fn next_u64x8(&mut self) -> u64x8 {
        let value = self.seed + INCREMENT;
        self.seed = value;
        let value = value * (MUL_XOR ^ value);
        value ^ (value >> SHIFT)
    }
}

#[cfg(test)]
mod tests {
    use core::simd::*;
    use itertools::Itertools;
    use rand_core::{RngCore, SeedableRng};

    use crate::testutil::{DOUBLE_RANGE, REF_SEED_FRAND_X8, test_uniform_distribution};

    use super::*;

    type RngSeed = FrandX8Seed;
    type RngImpl = FrandX8;

    #[test]
    fn reference() {
        let seed: RngSeed = REF_SEED_FRAND_X8.into();
        let mut rng = RngImpl::from_seed(seed);
        #[rustfmt::skip]
        let expected = [
            14822886784369691238,
            16363031081693429805,
            853694022644052200,
            12415232109135804396,
            11186080767534476449,
            3540825981221545167,
            2242727420787380596,
            10506211554217625353,
            2569786415164250473,
            7749215101137833260,
        ];
        for e in expected {
            let mem = rng.next_u64x8();
            for &v in mem.as_array() {
                assert_eq!(v, e);
            }
        }
    }

    #[test]
    fn sample_u64x8() {
        let mut seed = RngSeed::default();
        rand::rng().fill_bytes(&mut *seed);
        let mut rng = RngImpl::from_seed(seed);

        let values = *rng.next_u64x8().as_array();

        assert!(values.iter().all(|&v| v != 0));
        assert!(values.iter().unique().count() == values.len());
        println!("{values:?}");

        let values = *rng.next_u64x8().as_array();

        assert!(values.iter().all(|&v| v != 0));
        assert!(values.iter().unique().count() == values.len());
        println!("{values:?}");
    }

    #[test]
    fn sample_f64x8() {
        let mut seed = RngSeed::default();
        rand::rng().fill_bytes(&mut *seed);
        let mut rng = RngImpl::from_seed(seed);

        let values = *rng.next_f64x8().as_array();

        assert!(values.iter().all(|&v| v != 0.0));
        println!("{values:?}");

        let values = *rng.next_f64x8().as_array();

        assert!(values.iter().all(|&v| v != 0.0));
        println!("{values:?}");
    }

    #[test]
    #[cfg_attr(
        any(debug_assertions, miri),
        ignore = "distribution test requires release mode and real RNG"
    )]
    fn sample_f64x8_distribution() {
        let mut seed = RngSeed::default();
        rand::rng().fill_bytes(&mut *seed);
        let mut rng = RngImpl::from_seed(seed);

        let mut current: Option<f64x8> = None;
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
