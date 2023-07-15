use std::{arch::x86_64::*, mem::{transmute, self}};

use rand_core::{SeedableRng, le::read_u64_into};

pub struct Xoshiro256PlusPlusx4Seed(pub [u8; 128]);

pub struct Xoshiro256PlusPlusx4 {
    states: [[u64; 4]; 4],
}
impl Default for Xoshiro256PlusPlusx4Seed {
    fn default() -> Xoshiro256PlusPlusx4Seed {
        Xoshiro256PlusPlusx4Seed([0; 128])
    }
}

impl AsMut<[u8]> for Xoshiro256PlusPlusx4Seed {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl SeedableRng for Xoshiro256PlusPlusx4 {
    type Seed = Xoshiro256PlusPlusx4Seed;

    fn from_seed(seed: Self::Seed) -> Self {
        const SIZE: usize = mem::size_of::<u64>();
        const LEN: usize = 4;
        let mut states: [[u64; LEN]; LEN] = [[0; 4]; 4];
        read_u64_into(&seed.0[..(SIZE * LEN * 1)], &mut states[0][..]);
        read_u64_into(&seed.0[(SIZE * LEN * 1)..(SIZE * LEN * 2)], &mut states[1][..]);
        read_u64_into(&seed.0[(SIZE * LEN * 2)..(SIZE * LEN * 3)], &mut states[2][..]);
        read_u64_into(&seed.0[(SIZE * LEN * 3)..], &mut states[3][..]);

        Self {
            states
        }
    }
}

impl Xoshiro256PlusPlusx4 {
    pub fn next_m256i(&mut self) -> __m256i {
        unsafe {
            let s0 = _mm256_set_epi64x(
                transmute::<_, i64>(self.states[0][0]),
                transmute::<_, i64>(self.states[1][0]),
                transmute::<_, i64>(self.states[2][0]),
                transmute::<_, i64>(self.states[3][0])
            );
            let s3 = _mm256_set_epi64x(
                transmute::<_, i64>(self.states[0][3]),
                transmute::<_, i64>(self.states[1][3]),
                transmute::<_, i64>(self.states[2][3]),
                transmute::<_, i64>(self.states[3][3])
            );

            // s[0] + s[3]
            let sadd = _mm256_add_epi64(s0, s3);

            // rotl(s[0] + s[3], 23)
            // rotl: (x << k) | (x >> (64 - k)), k = 23
            const K: i32 = 23; 
            let rotl = _mm256_sll_epi64(sadd, _mm_set1_epi32(K));
            let rotl = _mm256_or_si256(rotl, _mm256_srl_epi64(sadd, _mm_set1_epi32(64 - K)));

            // rotl(...) + s[0]
            let result = _mm256_add_epi64(rotl, s0);

            //         let t = self.s[1] << 17;
            let s1 = _mm256_set_epi64x(
                transmute::<_, i64>(self.states[0][1]),
                transmute::<_, i64>(self.states[1][1]),
                transmute::<_, i64>(self.states[2][1]),
                transmute::<_, i64>(self.states[3][1])
            );
            let t = _mm256_sll_epi64(s1, _mm_set1_epi32(17));

            //         self.s[2] ^= self.s[0];
            //         self.s[3] ^= self.s[1];
            //         self.s[1] ^= self.s[2];
            //         self.s[0] ^= self.s[3];

            //         self.s[2] ^= t;

            //         self.s[3] = self.s[3].rotate_left(45);

            //         result_plusplus

            result
        }
    }

    pub fn next_u64s(&mut self, mem: &mut U64x4) {
        assert!(mem::align_of_val(mem) % 32 == 0, "mem needs to be aligned to 32 bytes");

        let vec = self.next_m256i();
        unsafe {   
            _mm256_store_si256(transmute::<_, *mut __m256i>(&mut mem.0), vec);
        }
    }
}

#[repr(align(32))]
pub struct U64x4([u64; 4]);

// impl RngCore for Xoshiro256PlusPlus {
//     #[inline]
//     fn next_u64(&mut self) -> u64 {
//         let result_plusplus = self.s[0]
//             .wrapping_add(self.s[3])
//             .rotate_left(23)
//             .wrapping_add(self.s[0]);

//         let t = self.s[1] << 17;

//         self.s[2] ^= self.s[0];
//         self.s[3] ^= self.s[1];
//         self.s[1] ^= self.s[2];
//         self.s[0] ^= self.s[3];

//         self.s[2] ^= t;

//         self.s[3] = self.s[3].rotate_left(45);

//         result_plusplus
//     }

//     #[inline]
//     fn next_u32(&mut self) -> u32 {
//         // The lowest bits have some linear dependencies, so we use the
//         // upper bits instead.
//         (self.next_u64() >> 32) as u32
//     }

//     #[inline]
//     fn fill_bytes(&mut self, dest: &mut [u8]) {
//         fill_bytes_via_next(self, dest);
//     }

//     #[inline]
//     fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
//         self.fill_bytes(dest);
//         Ok(())
//     }
// }

#[cfg(test)]
mod tests {
    use std::mem;

    use num_traits::PrimInt;
    use rand::rngs::SmallRng;
    use rand_core::{SeedableRng, RngCore};

    use super::*;

    #[test]
    fn bitfiddling() {
        let v = 0b00000000_00000000_00000000_000000001u32;
        print(v);
    }

    #[test]
    fn rng() {
        let mut seeder = SmallRng::seed_from_u64(0);

        let mut seed: Xoshiro256PlusPlusx4Seed = Default::default();
        seeder.fill_bytes(&mut seed.0[..]);

        let mut rng = Xoshiro256PlusPlusx4::from_seed(seed);

        let mut values = U64x4([0; 4]);
        rng.next_u64s(&mut values);

        assert!(values.0.iter().all(|&v| v != 0));
    }
    
    fn print<T>(v: T) where T: PrimInt + ToString {
        let size = mem::size_of::<T>();
        let bit_size = size * 8;

        const PREFIX: &str = "0b";

        let mut output = String::with_capacity(PREFIX.len() + bit_size + (size - 1));
        output.push_str(PREFIX);
        let one = T::one();
        for n in (0..bit_size).rev() {
            let bit = (v >> n) & one;
            output.push_str(&bit.to_string());
            if n != 0 && n % 8 == 0 {
                output.push('_');
            }
        }
        println!("{output}");
    }
}
