#[cfg(test)]
pub mod test_support {
    const INCREMENT: u64 = 12964901029718341801;
    const MUL_XOR: u64 = 149988720821803190;

    fn repeated_lane_seed<const BYTES: usize>(value: u64, lanes: usize) -> [u8; BYTES] {
        let mut seed = [0u8; BYTES];
        let words = BYTES / 8;

        assert_eq!(words % lanes, 0);

        for (index, chunk) in seed.chunks_exact_mut(8).enumerate() {
            let repeated_index = index / lanes;
            assert_eq!(repeated_index, 0);
            chunk.copy_from_slice(&value.to_le_bytes());
        }

        seed
    }

    pub fn ref_seed_x4() -> [u8; 32] {
        repeated_lane_seed::<32>(1, 4)
    }

    pub fn ref_seed_x8() -> [u8; 64] {
        repeated_lane_seed::<64>(1, 8)
    }

    pub struct FrandReference {
        seed: u64,
    }

    impl FrandReference {
        pub const fn new(seed: u64) -> Self {
            Self { seed }
        }

        pub const fn next_u64(&mut self) -> u64 {
            let value = self.seed.wrapping_add(INCREMENT);
            self.seed = value;
            let value = value.wrapping_mul(MUL_XOR ^ value);
            value ^ (value >> 32)
        }
    }
}
