use core::{
    mem,
    ops::{Deref, DerefMut},
    simd::u64x8,
};

use rand_core::SeedableRng;

use crate::frand::{hash_seed_bytes, repeated_seed_bytes};

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

        let seed = hash_seed_bytes::<64>(&seed[..]);
        let s = read_u64_into_vec(&seed);

        Self { seed: s }
    }

    fn seed_from_u64(seed: u64) -> Self {
        Self::from_seed(Self::Seed::from(repeated_seed_bytes::<64>(seed)))
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
    use rand_core::SeedableRng;

    use super::{FrandX8, SimdRandX8};
    use crate::frand::test_support::assert_seed_from_u64_matches_upstream;

    #[test]
    fn seed_from_u64_matches_upstream() {
        assert_seed_from_u64_matches_upstream::<8, _>(42, FrandX8::seed_from_u64(42), |rng| {
            rng.next_u64x8().to_array()
        });
    }
}
