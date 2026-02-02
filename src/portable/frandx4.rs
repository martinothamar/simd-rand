use core::{
    mem,
    ops::{Deref, DerefMut},
    simd::u64x4,
};

use rand_core::SeedableRng;

use super::{SimdRandX4, read_u64_into_vec};

const INCREMENT: u64x4 = u64x4::from_array([12964901029718341801; 4]);
const MUL_XOR: u64x4 = u64x4::from_array([149988720821803190; 4]);
const SHIFT: u64x4 = u64x4::from_array([32; 4]);

#[derive(Clone, Default)]
pub struct FrandX4Seed([u8; 32]);

impl FrandX4Seed {
    #[must_use]
    pub const fn new(seed: [u8; 32]) -> Self {
        Self(seed)
    }
}

impl From<[u8; 32]> for FrandX4Seed {
    fn from(val: [u8; 32]) -> Self {
        Self::new(val)
    }
}

impl From<&[u8]> for FrandX4Seed {
    fn from(val: &[u8]) -> Self {
        assert_eq!(val.len(), 32);
        let mut seed = [0u8; 32];
        seed.copy_from_slice(val);
        Self::new(seed)
    }
}

impl Deref for FrandX4Seed {
    type Target = [u8; 32];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for FrandX4Seed {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl AsRef<[u8]> for FrandX4Seed {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl AsMut<[u8]> for FrandX4Seed {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

pub struct FrandX4 {
    seed: u64x4,
}

impl SeedableRng for FrandX4 {
    type Seed = FrandX4Seed;

    fn from_seed(seed: Self::Seed) -> Self {
        const SIZE: usize = mem::size_of::<u64>();
        const LEN: usize = u64x4::LEN;
        assert!(seed.len() == SIZE * LEN);

        let s = read_u64_into_vec(&seed[..]);

        Self { seed: s }
    }
}

impl SimdRandX4 for FrandX4 {
    fn next_u64x4(&mut self) -> u64x4 {
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

    use crate::testutil::{DOUBLE_RANGE, REF_SEED_FRAND_X4, test_uniform_distribution};

    use super::*;

    type RngSeed = FrandX4Seed;
    type RngImpl = FrandX4;

    #[test]
    fn reference() {
        let seed: RngSeed = REF_SEED_FRAND_X4.into();
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
            let mem = rng.next_u64x4();
            for &v in mem.as_array() {
                assert_eq!(v, e);
            }
        }
    }

    #[test]
    fn sample_u64x4() {
        let mut seed = RngSeed::default();
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
    fn sample_f64x4() {
        let mut seed = RngSeed::default();
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
    #[cfg_attr(
        any(debug_assertions, miri),
        ignore = "distribution test requires release mode and real RNG"
    )]
    fn sample_f64x4_distribution() {
        let mut seed = RngSeed::default();
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
