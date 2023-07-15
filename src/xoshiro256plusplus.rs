use std::{arch::x86_64::*, mem::{transmute, self}};

use rand_core::SeedableRng;

pub struct Xoshiro256PlusPlusX4Seed(pub [u8; 128]);

#[repr(align(32))]
pub struct Xoshiro256PlusPlusX4 {
    s0: __m256i,
    s1: __m256i,
    s2: __m256i,
    s3: __m256i,
}
impl Default for Xoshiro256PlusPlusX4Seed {
    fn default() -> Xoshiro256PlusPlusX4Seed {
        Xoshiro256PlusPlusX4Seed([0; 128])
    }
}

impl AsMut<[u8]> for Xoshiro256PlusPlusX4Seed {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

impl SeedableRng for Xoshiro256PlusPlusX4 {
    type Seed = Xoshiro256PlusPlusX4Seed;

    fn from_seed(seed: Self::Seed) -> Self {
        const SIZE: usize = mem::size_of::<u64>();
        const LEN: usize = 4;
        const VECSIZE: usize = SIZE * LEN;
        unsafe {
            let mut s0: __m256i = _mm256_setzero_si256();
            let mut s1: __m256i = _mm256_setzero_si256();
            let mut s2: __m256i = _mm256_setzero_si256();
            let mut s3: __m256i = _mm256_setzero_si256();
            read_u64_into_vec(&seed.0[(VECSIZE * 0)..(VECSIZE * 1)], &mut s0);
            read_u64_into_vec(&seed.0[(VECSIZE * 1)..(VECSIZE * 2)], &mut s1);
            read_u64_into_vec(&seed.0[(VECSIZE * 2)..(VECSIZE * 3)], &mut s2);
            read_u64_into_vec(&seed.0[(VECSIZE * 3)..(VECSIZE * 4)], &mut s3);
    
            Self {
                s0, s1, s2, s3
            }
        }
    }
}

#[inline]
pub fn read_u64_into_vec(src: &[u8], dst: &mut __m256i) {
    const SIZE: usize = mem::size_of::<u64>();
    assert!(src.len() == SIZE * 4);
    unsafe {
        *dst = _mm256_set_epi64x(
            transmute::<_, i64>(u64::from_le_bytes(src[(SIZE * 0)..(SIZE * 1)].try_into().unwrap())),
            transmute::<_, i64>(u64::from_le_bytes(src[(SIZE * 1)..(SIZE * 2)].try_into().unwrap())),
            transmute::<_, i64>(u64::from_le_bytes(src[(SIZE * 2)..(SIZE * 3)].try_into().unwrap())),
            transmute::<_, i64>(u64::from_le_bytes(src[(SIZE * 3)..(SIZE * 4)].try_into().unwrap())),
        )
    }
}

impl Xoshiro256PlusPlusX4 {
    pub fn next_m256i(&mut self) -> __m256i {
        unsafe {
            let s0 = _mm256_load_si256(
                transmute::<_, *const __m256i>(&self.s0)
            );
            let s3 = _mm256_load_si256(
                transmute::<_, *const __m256i>(&self.s3)
            );

            // s[0] + s[3]
            let sadd = _mm256_add_epi64(s0, s3);

            // rotl(s[0] + s[3], 23)
            // rotl: (x << k) | (x >> (64 - k)), k = 23
            let rotl = rotate_left::<23>(sadd);

            // rotl(...) + s[0]
            let result = _mm256_add_epi64(rotl, s0);

            //         let t = self.s[1] << 17;
            let s1 = _mm256_load_si256(
                transmute::<_, *const __m256i>(&self.s1)
            );
            let t = _mm256_sll_epi64(s1, _mm_set1_epi32(17));

            //         self.s[2] ^= self.s[0];
            //         self.s[3] ^= self.s[1];
            //         self.s[1] ^= self.s[2];
            //         self.s[0] ^= self.s[3];
            self.s2 = _mm256_xor_si256(self.s2, self.s0);
            self.s3 = _mm256_xor_si256(self.s3, self.s1);
            self.s1 = _mm256_xor_si256(self.s1, self.s2);
            self.s0 = _mm256_xor_si256(self.s0, self.s3);

            //         self.s[2] ^= t;
            self.s2 = _mm256_xor_si256(self.s2, t);

            //         self.s[3] = self.s[3].rotate_left(45);
            self.s3 = rotate_left::<45>(self.s3);

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

fn rotate_left<const K: i32>(x: __m256i) -> __m256i {
    unsafe { 
        // rotl: (x << k) | (x >> (64 - k)), k = 23
        let rotl = _mm256_sll_epi64(x, _mm_set1_epi32(K));
        _mm256_or_si256(rotl, _mm256_srl_epi64(x, _mm_set1_epi32(64 - K)))
    }
}

#[cfg(test)]
mod tests {
    use std::mem;

    use num_traits::PrimInt;
    use rand::rngs::SmallRng;
    use rand_core::{SeedableRng, RngCore};
    use itertools::Itertools;

    use super::*;

    #[test]
    fn bitfiddling() {
        let v = 0b00000000_00000000_00000000_000000001u32;
        print(v);
    }

    #[test]
    fn rng() {
        let mut seeder = SmallRng::seed_from_u64(0);

        let mut seed: Xoshiro256PlusPlusX4Seed = Default::default();
        seeder.fill_bytes(&mut seed.0[..]);

        let mut rng = Xoshiro256PlusPlusX4::from_seed(seed);

        let mut values = U64x4([0; 4]);
        rng.next_u64s(&mut values);

        let data = values.0;
        assert!(data.iter().all(|&v| v != 0));
        assert!(data.iter().unique().count() == data.len());
        println!("{data:?}");

        let mut values = U64x4([0; 4]);
        rng.next_u64s(&mut values);

        let data = values.0;
        assert!(data.iter().all(|&v| v != 0));
        assert!(data.iter().unique().count() == data.len());
        println!("{data:?}");
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
