use core::{
    mem,
    ops::{Deref, DerefMut},
    simd::u64x4,
};

use rand_core::SeedableRng;

use crate::frand::{hash_seed_bytes, repeated_seed_bytes};

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

        let seed = hash_seed_bytes::<32>(&seed[..]);
        let s = read_u64_into_vec(&seed);

        Self { seed: s }
    }

    fn seed_from_u64(seed: u64) -> Self {
        Self::from_seed(Self::Seed::from(repeated_seed_bytes::<32>(seed)))
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
    use rand_core::SeedableRng;

    use super::{FrandX4, SimdRandX4};
    use crate::frand::test_support::assert_seed_from_u64_matches_upstream;

    #[test]
    fn seed_from_u64_matches_upstream() {
        assert_seed_from_u64_matches_upstream::<4, _>(42, FrandX4::seed_from_u64(42), |rng| {
            rng.next_u64x4().to_array()
        });
    }
}
