use std::{arch::x86_64::*, mem::{transmute, self}};

use rand_core::{RngCore, impls::fill_bytes_via_next, Error, SeedableRng, le::read_u64_into};

pub struct Xoshiro256PlusPlus {
    s: [u64; 4],
}

pub struct Xoshiro256PlusPlusx4 {
    rng: [Xoshiro256PlusPlus; 4],
}

impl SeedableRng for Xoshiro256PlusPlus {
    type Seed = [u8; 32];

    /// Create a new `Xoshiro256PlusPlus`.  If `seed` is entirely 0, it will be
    /// mapped to a different seed.
    #[inline]
    fn from_seed(seed: [u8; 32]) -> Xoshiro256PlusPlus {
        if seed.iter().all(|&x| x == 0) {
            return Self::seed_from_u64(0);
        }
        let mut state = [0; 4];
        read_u64_into(&seed, &mut state);
        Xoshiro256PlusPlus { s: state }
    }

    /// Create a new `Xoshiro256PlusPlus` from a `u64` seed.
    ///
    /// This uses the SplitMix64 generator internally.
    fn seed_from_u64(mut state: u64) -> Self {
        const PHI: u64 = 0x9e3779b97f4a7c15;
        let mut seed = Self::Seed::default();
        for chunk in seed.as_mut().chunks_mut(8) {
            state = state.wrapping_add(PHI);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z = z ^ (z >> 31);
            chunk.copy_from_slice(&z.to_le_bytes());
        }
        Self::from_seed(seed)
    }
}

impl Xoshiro256PlusPlusx4 {
    pub fn next_vecs(&mut self) -> __m256i {
        unsafe {
            let s0 = _mm256_set_epi64x(
                transmute::<_, i64>(self.rng[0].s[0]),
                transmute::<_, i64>(self.rng[1].s[0]),
                transmute::<_, i64>(self.rng[2].s[0]),
                transmute::<_, i64>(self.rng[3].s[0])
            );
            let s3 = _mm256_set_epi64x(
                transmute::<_, i64>(self.rng[0].s[3]),
                transmute::<_, i64>(self.rng[1].s[3]),
                transmute::<_, i64>(self.rng[2].s[3]),
                transmute::<_, i64>(self.rng[3].s[3])
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

            result
        }
    }

    pub fn next_u64s(&mut self, mem: &mut U64x4) {
        assert!(mem::align_of_val(mem) % 32 == 0, "mem needs to be aligned to 32 bytes");

        let vec = self.next_vecs();
        unsafe {   
            _mm256_store_si256(transmute::<_, *mut __m256i>(&mut mem.0), vec);
        }
    }
}

#[repr(align(32))]
pub struct U64x4([u64; 4]);

impl RngCore for Xoshiro256PlusPlus {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result_plusplus = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);

        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;

        self.s[3] = self.s[3].rotate_left(45);

        result_plusplus
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        // The lowest bits have some linear dependencies, so we use the
        // upper bits instead.
        (self.next_u64() >> 32) as u32
    }

    #[inline]
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        fill_bytes_via_next(self, dest);
    }

    #[inline]
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

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

        let rngs: [Xoshiro256PlusPlus; 4] = [
            Xoshiro256PlusPlus::seed_from_u64(seeder.next_u64()),
            Xoshiro256PlusPlus::seed_from_u64(seeder.next_u64()),
            Xoshiro256PlusPlus::seed_from_u64(seeder.next_u64()),
            Xoshiro256PlusPlus::seed_from_u64(seeder.next_u64()),
        ];

        let mut rng = Xoshiro256PlusPlusx4 {
            rng: rngs,
        };

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
