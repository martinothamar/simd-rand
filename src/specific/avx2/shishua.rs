use std::mem::transmute;
use std::ptr::NonNull;
use std::{arch::x86_64::*, mem::size_of};

use rand_core::le::read_u64_into;
use rand_core::{RngCore, SeedableRng};
use std::alloc::{Layout, LayoutError};
use std::iter::Iterator;
use std::{alloc, mem};

use super::simdprng::*;
use super::vecs::*;

pub const DEFAULT_BUFFER_SIZE: usize = 1024 * 32;

/// Shishua is a fast, vectorized, buffered PRNG.
/// When initialized, it will seed its state of size `BUFFER_SIZE` (32k by default).
/// When sampling, if enough randomness is buffered, it will just extract your T from the buffered bytes.
/// When all the buffered randomness is spent, it will rebuffer using vectorized instructions
///
/// # Performance
///
/// Performance varies slightly by what `BUFFER_SIZE` is set.
/// Ideal conditions are of course when the entire state can reside in L1 cache.
/// Wether or not this happens is dependent on _how_ the generator is used,
/// so I recommend you benchmark your specific workload and use something like
/// perf stat to observe cache misses (L1-dcache-load-misses:u).
///
/// # Safety
///
/// As this is a performance-oriented library, there is some unsafe code here.
/// One example is elision of bounds check in the hotpath of extracting random bytes
/// from the buffer. The library is in a single file so feel free to check and
/// provide feedback.
///

pub struct Shishua<const BUFFER_SIZE: usize = DEFAULT_BUFFER_SIZE> {
    state: NonNull<BufferedState<BUFFER_SIZE>>,
}

const BUFFERED_STATE_ALIGNMENT: usize = 128;
const fn get_buffered_state_layout_unchecked<const BUFFER_SIZE: usize>() -> Layout {
    unsafe { Layout::from_size_align_unchecked(size_of::<BufferedState<BUFFER_SIZE>>(), BUFFERED_STATE_ALIGNMENT) }
}
const fn get_buffered_state_layout<const BUFFER_SIZE: usize>() -> Result<Layout, LayoutError> {
    Layout::from_size_align(size_of::<BufferedState<BUFFER_SIZE>>(), BUFFERED_STATE_ALIGNMENT)
}

impl<const BUFFER_SIZE: usize> Shishua<BUFFER_SIZE> {
    pub const LAYOUT: Layout = get_buffered_state_layout_unchecked::<BUFFER_SIZE>();

    // For 'dasm' cfg, our intention is to analyze the generated assembly,
    // so suggest that the function shouldn't be inlined
    #[cfg_attr(dasm, inline(never))]
    // For any other cfg this should be inlined, as it is in hotpath for Rng and RngCore traits
    #[cfg_attr(not(dasm), inline(always))]
    fn fill_bytes_arr<const N: usize>(&mut self, dest: &mut [u8; N]) {
        unsafe {
            let state = self.state.as_mut();

            state.ensure_buffered(N);

            let src = state
                .buffer
                .0
                .as_slice()
                .get_unchecked(state.buffer_index..state.buffer_index + N);
            dest.copy_from_slice(src);
            state.buffer_index += N;
        };
    }

    #[inline(always)]
    pub fn buffer_index(&self) -> usize {
        let state = unsafe { self.state.as_ref() };
        state.buffer_index
    }

    #[cfg_attr(dasm, inline(never))]
    #[cfg_attr(not(dasm), inline(always))]
    pub fn next_f32(&mut self) -> f32 {
        let v = self.next_u32();
        (v >> 8) as f32 * (1.0f32 / (1u32 << 24) as f32)
    }

    #[cfg_attr(dasm, inline(never))]
    #[cfg_attr(not(dasm), inline(always))]
    pub fn next_f64(&mut self) -> f64 {
        let v = self.next_u64();
        (v >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

impl<const BUFFER_SIZE: usize> SimdPrng for Shishua<BUFFER_SIZE> {
    #[cfg_attr(dasm, inline(never))]
    #[cfg_attr(not(dasm), inline(always))]
    fn next_m256i(&mut self, vector: &mut __m256i) {
        const SIZE: usize = mem::size_of::<__m256i>();
        unsafe {
            let state = self.state.as_mut();

            state.ensure_buffered(SIZE);

            let src = &state.buffer.0[state.buffer_index];

            *vector = _mm256_load_si256(transmute::<_, *const __m256i>(src));

            state.buffer_index += SIZE;
        }
    }

    #[cfg_attr(dasm, inline(never))]
    #[cfg_attr(not(dasm), inline(always))]
    fn next_u64x4(&mut self, vector: &mut U64x4) {
        unsafe {
            let mut result = _mm256_set1_epi64x(0);
            self.next_m256i(&mut result);
            _mm256_store_si256(transmute::<_, *mut __m256i>(vector), result);
        }
    }

    #[cfg_attr(dasm, inline(never))]
    #[cfg_attr(not(dasm), inline(always))]
    fn next_f64x4(&mut self, vector: &mut F64x4) {
        let mut v = Default::default();
        self.next_u64x4(&mut v);

        vector[0] = (v[0] >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        vector[1] = (v[1] >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        vector[2] = (v[2] >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        vector[3] = (v[3] >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
    }
}

impl<const BUFFER_SIZE: usize> SeedableRng for Shishua<BUFFER_SIZE> {
    type Seed = [u8; 32];

    #[cfg_attr(dasm, inline(never))]
    fn from_seed(seed: Self::Seed) -> Self {
        assert!(
            get_buffered_state_layout::<BUFFER_SIZE>().is_ok(),
            "couldnt construct memory layout of Shishua buffered state"
        );
        assert!(
            BUFFER_SIZE.is_power_of_two(),
            "The Shishua buffer size must be a power of 2"
        );
        assert!(
            BUFFER_SIZE >= 256,
            "The Shishua buffer size must be >= 256 (and power of 2)"
        );

        let ptr = unsafe {
            let ptr = alloc::alloc(Self::LAYOUT) as *mut BufferedState<BUFFER_SIZE>;

            let buffered_state = ptr.as_mut().expect("Failed to allocate state for Shishua");

            let mut iseed = [0; 4];
            read_u64_into(&seed[..], iseed.as_mut_slice());

            buffered_state.state.prng_init(mem::transmute::<_, &[_; 4]>(&iseed));
            buffered_state.rebuffer();

            NonNull::new_unchecked(ptr)
        };

        Self { state: ptr }
    }
}

impl<const BUFFER_SIZE: usize> Drop for Shishua<BUFFER_SIZE> {
    #[cfg_attr(dasm, inline(never))]
    fn drop(&mut self) {
        let ptr = self.state.as_ptr();
        unsafe {
            alloc::dealloc(ptr as *mut u8, Self::LAYOUT);
        }
    }
}

impl<const BUFFER_SIZE: usize> RngCore for Shishua<BUFFER_SIZE> {
    #[cfg_attr(dasm, inline(never))]
    #[cfg_attr(not(dasm), inline(always))]
    fn next_u32(&mut self) -> u32 {
        let mut result: u32 = 0;
        let mut bytes = unsafe { mem::transmute::<_, &mut [u8; 4]>(&mut result) };
        self.fill_bytes_arr(&mut bytes);
        result
    }

    #[cfg_attr(dasm, inline(never))]
    #[cfg_attr(not(dasm), inline(always))]
    fn next_u64(&mut self) -> u64 {
        let mut result: u64 = 0;
        let mut bytes = unsafe { mem::transmute::<_, &mut [u8; 8]>(&mut result) };
        self.fill_bytes_arr(&mut bytes);
        result
    }

    #[cfg_attr(dasm, inline(never))]
    #[cfg_attr(not(dasm), inline(always))]
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let size = dest.len();

        unsafe {
            let state = self.state.as_mut();

            state.ensure_buffered(size);

            let src = state
                .buffer
                .0
                .as_slice()
                .get_unchecked(state.buffer_index..state.buffer_index + size);
            dest.copy_from_slice(src);
            state.buffer_index += size;
        };
    }

    #[cfg_attr(dasm, inline(never))]
    #[cfg_attr(not(dasm), inline(always))]
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

struct BufferedState<const BUFFER_SIZE: usize> {
    state: RawState,
    buffer: BufferedStateBuffer<BUFFER_SIZE>,
    buffer_index: usize,
}

#[repr(align(32))]
struct BufferedStateBuffer<const BUFFER_SIZE: usize>([u8; BUFFER_SIZE]);

impl<const BUFFER_SIZE: usize> BufferedState<BUFFER_SIZE> {
    #[inline(always)] // This should be inlined, this branch will be checked every time we sample
    fn ensure_buffered(&mut self, size: usize) {
        if BUFFER_SIZE - self.buffer_index < size {
            self.rebuffer();
        }
    }

    #[cold] // This attribute seems to make LLVM organize jumps/branches better
    #[inline(never)] // This should not be inlined, as entering the branch above is the rare case
    fn rebuffer(&mut self) {
        unsafe {
            self.state.prng_gen(&mut self.buffer.0[..]);
        }
        self.buffer_index = 0;
    }
}

struct RawState {
    state: [__m256i; 4],
    output: [__m256i; 4],
    counter: __m256i,
}

impl RawState {
    #[inline(never)] // This should be called rarely (only when rebuffering), so prefer not to inline
    unsafe fn prng_init(&mut self, seed: &[u64; 4]) {
        const STEPS: usize = 1;
        const ROUNDS: usize = 13;

        *self = mem::zeroed();
        let mut buf: [u8; 128 * STEPS] = [0; 128 * STEPS];

        self.state[0] = _mm256_set_epi64x(
            mem::transmute::<u64, i64>(PHI[3]),
            mem::transmute::<u64, i64>(PHI[2]) ^ seed[1] as i64,
            mem::transmute::<u64, i64>(PHI[1]),
            mem::transmute::<u64, i64>(PHI[0]) ^ seed[0] as i64,
        );
        self.state[1] = _mm256_set_epi64x(
            mem::transmute::<u64, i64>(PHI[7]),
            mem::transmute::<u64, i64>(PHI[6]) ^ seed[3] as i64,
            mem::transmute::<u64, i64>(PHI[5]),
            mem::transmute::<u64, i64>(PHI[4]) ^ seed[2] as i64,
        );
        self.state[2] = _mm256_set_epi64x(
            mem::transmute::<u64, i64>(PHI[11]),
            mem::transmute::<u64, i64>(PHI[10]) ^ seed[3] as i64,
            mem::transmute::<u64, i64>(PHI[9]),
            mem::transmute::<u64, i64>(PHI[8]) ^ seed[2] as i64,
        );
        self.state[3] = _mm256_set_epi64x(
            mem::transmute::<u64, i64>(PHI[15]),
            mem::transmute::<u64, i64>(PHI[14]) ^ seed[1] as i64,
            mem::transmute::<u64, i64>(PHI[13]),
            mem::transmute::<u64, i64>(PHI[12]) ^ seed[0] as i64,
        );
        for _ in 0..ROUNDS {
            Self::prng_gen(self, &mut buf[..]);
            self.state[0] = self.output[3];
            self.state[1] = self.output[2];
            self.state[2] = self.output[1];
            self.state[3] = self.output[0];
        }
    }

    #[inline(never)] // This should be called rarely (only when rebuffering), so prefer not to inline
    unsafe fn prng_gen(&mut self, buf: &mut [u8]) {
        let mut o0 = self.output[0];
        let mut o1 = self.output[1];
        let mut o2 = self.output[2];
        let mut o3 = self.output[3];
        let mut s0 = self.state[0];
        let mut s1 = self.state[1];
        let mut s2 = self.state[2];
        let mut s3 = self.state[3];
        let mut t0: __m256i;
        let mut t1: __m256i;
        let mut t2: __m256i;
        let mut t3: __m256i;
        let mut u0: __m256i;
        let mut u1: __m256i;
        let mut u2: __m256i;
        let mut u3: __m256i;
        let mut counter = self.counter;

        let shu0 = _mm256_set_epi32(4, 3, 2, 1, 0, 7, 6, 5);
        let shu1 = _mm256_set_epi32(2, 1, 0, 7, 6, 5, 4, 3);

        let increment = _mm256_set_epi64x(1, 3, 5, 7);

        assert!(buf.len() % 128 == 0);

        let buf_ptr = buf.as_mut_ptr();
        for i in (0..buf.len()).step_by(128) {
            _mm256_storeu_si256(buf_ptr.add(i + 0) as *mut __m256i, o0);
            _mm256_storeu_si256(buf_ptr.add(i + 32) as *mut __m256i, o1);
            _mm256_storeu_si256(buf_ptr.add(i + 64) as *mut __m256i, o2);
            _mm256_storeu_si256(buf_ptr.add(i + 96) as *mut __m256i, o3);

            s1 = _mm256_add_epi64(s1, counter);
            s3 = _mm256_add_epi64(s3, counter);
            counter = _mm256_add_epi64(counter, increment);

            u0 = _mm256_srli_epi64(s0, 1);
            u1 = _mm256_srli_epi64(s1, 3);
            u2 = _mm256_srli_epi64(s2, 1);
            u3 = _mm256_srli_epi64(s3, 3);
            t0 = _mm256_permutevar8x32_epi32(s0, shu0);
            t1 = _mm256_permutevar8x32_epi32(s1, shu1);
            t2 = _mm256_permutevar8x32_epi32(s2, shu0);
            t3 = _mm256_permutevar8x32_epi32(s3, shu1);

            s0 = _mm256_add_epi64(t0, u0);
            s1 = _mm256_add_epi64(t1, u1);
            s2 = _mm256_add_epi64(t2, u2);
            s3 = _mm256_add_epi64(t3, u3);

            // Two orthogonally grown pieces evolving independently, XORed.
            o0 = _mm256_xor_si256(u0, t1);
            o1 = _mm256_xor_si256(u2, t3);
            o2 = _mm256_xor_si256(s0, s3);
            o3 = _mm256_xor_si256(s2, s1);
        }

        self.output[0] = o0;
        self.output[1] = o1;
        self.output[2] = o2;
        self.output[3] = o3;
        self.state[0] = s0;
        self.state[1] = s1;
        self.state[2] = s2;
        self.state[3] = s3;
        self.counter = counter;
    }
}

#[rustfmt::skip]
const PHI: [u64; 16] = [
    0x9E3779B97F4A7C15, 0xF39CC0605CEDC834, 0x1082276BF3A27251, 0xF86C6A11D0C18E95,
    0x2767F0B153D27B7F, 0x0347045B5BF1827F, 0x01886F0928403002, 0xC1D64BA40F335E36,
    0xF06AD7AE9717877E, 0x85839D6EFFBD7DC6, 0x64D325D1C5371682, 0xCADD0CCCFDFFBBE1,
    0x626E33B8D04B4331, 0xBBF73C790D94F79D, 0x471C4AB3ED3D82A5, 0xFEC507705E4AE6E5,
];

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::Rng;
    use serial_test::parallel;

    type Shishua = super::Shishua<DEFAULT_BUFFER_SIZE>;

    use crate::testutil::{test_uniform_distribution, DOUBLE_RANGE, FLOAT_RANGE};

    use super::*;

    #[test]
    #[parallel]
    fn alignment() {
        assert!(mem::align_of::<BufferedState<DEFAULT_BUFFER_SIZE>>() % 32 == 0);

        let rng = create_with_zero_seed();
        let state = unsafe { rng.state.as_ref() };
        let buf_alignment = mem::align_of_val(&state.buffer);
        assert!(buf_alignment % 32 == 0);
    }

    #[test]
    #[parallel]
    fn reference_zero() {
        unsafe {
            let mut state: RawState = mem::zeroed();
            state.prng_init(&SEED_ZERO);
            let mut buf: [u8; 512] = [0; 512];
            state.prng_gen(&mut buf[..]);

            assert_eq!(&buf, &SEED_ZERO_EXPECTED);
        }
    }

    #[test]
    #[parallel]
    fn reference_pi() {
        unsafe {
            let mut state: RawState = mem::zeroed();
            state.prng_init(&SEED_PI);
            let mut buf: [u8; 512] = [0; 512];
            state.prng_gen(&mut buf[..]);

            assert_eq!(&buf, &SEED_PI_EXPECTED);
        }
    }

    #[test]
    #[parallel]
    fn construction_zero_seed() {
        let mut rng = create_with_zero_seed();
        assert!(rng.buffer_index() == 0);
        let v = rng.gen_range(DOUBLE_RANGE);
        assert!(DOUBLE_RANGE.contains(&v) && v != 0.0);
    }

    #[test]
    #[parallel]
    fn construction_predefined_seed() {
        let mut rng = create_with_predefined_seed();
        assert!(rng.buffer_index() == 0);
        let v = rng.gen_range(DOUBLE_RANGE);
        assert!(DOUBLE_RANGE.contains(&v) && v != 0.0);
    }

    #[test]
    #[parallel]
    #[should_panic]
    fn construction_invalid_size_power() {
        let seed = get_predefined_seed();
        let rng = super::Shishua::<127>::from_seed(*seed);
        assert!(rng.buffer_index() == 0);
    }

    #[test]
    #[parallel]
    #[should_panic]
    fn construction_invalid_size_small() {
        let seed = get_predefined_seed();
        let rng = super::Shishua::<128>::from_seed(*seed);
        assert!(rng.buffer_index() == 0);
    }

    #[test]
    #[parallel]
    fn sample_u32() {
        let mut rng = create_with_zero_seed();

        let n = rng.next_u32();
        assert!(n >= u32::MIN && n <= u32::MAX);
    }

    #[test]
    #[parallel]
    fn sample_u64() {
        let mut rng = create_with_zero_seed();

        let n = rng.next_u64();
        assert!(n >= u64::MIN && n <= u64::MAX);
    }

    #[test]
    #[parallel]
    fn sample_f64() {
        let mut rng = create_with_zero_seed();

        let n = rng.gen_range(DOUBLE_RANGE);
        assert!(DOUBLE_RANGE.contains(&n));
    }

    #[test]
    #[parallel]
    fn sample_f32() {
        let mut rng = create_with_zero_seed();

        let n = rng.gen_range(FLOAT_RANGE);
        assert!(FLOAT_RANGE.contains(&n));
    }

    #[test]
    #[parallel]
    fn sample_u64x4() {
        let mut rng = create_with_predefined_seed();

        let mut values = U64x4::new([0; 4]);
        rng.next_u64x4(&mut values);

        assert!(values.iter().all(|&v| v != 0));
        assert!(values.iter().unique().count() == values.len());
        println!("{values:?}");

        let mut values = U64x4::new([0; 4]);
        rng.next_u64x4(&mut values);

        assert!(values.iter().all(|&v| v != 0));
        assert!(values.iter().unique().count() == values.len());
        println!("{values:?}");
    }

    #[test]
    #[parallel]
    fn sample_f64x4() {
        let mut rng = create_with_predefined_seed();

        let mut values = F64x4::new([0.0; 4]);
        rng.next_f64x4(&mut values);

        assert!(values.iter().all(|&v| v != 0.0));
        println!("{values:?}");

        let mut values = F64x4::new([0.0; 4]);
        rng.next_f64x4(&mut values);

        assert!(values.iter().all(|&v| v != 0.0));
        println!("{values:?}");
    }

    #[test]
    #[parallel]
    fn sample_f64_distribution() {
        let mut rng = create_with_zero_seed();

        test_uniform_distribution::<100_000_000, f64>(|| rng.gen_range(DOUBLE_RANGE), DOUBLE_RANGE);
    }

    #[test]
    #[parallel]
    fn sample_f64x4_distribution() {
        let mut rng = create_with_zero_seed();

        let mut current: Option<F64x4> = None;
        let mut current_index: usize = 0;

        test_uniform_distribution::<100_000_000, f64>(
            || match &current {
                Some(vector) if current_index < 4 => {
                    let result = vector[current_index];
                    current_index += 1;
                    return result;
                }
                _ => {
                    let mut vector = Default::default();
                    current_index = 0;
                    rng.next_f64x4(&mut vector);
                    let result = vector[current_index];
                    current = Some(vector);
                    current_index += 1;
                    return result;
                }
            },
            DOUBLE_RANGE,
        );
    }

    #[test]
    #[parallel]
    fn sample_f32_distribution() {
        let mut rng = create_with_zero_seed();

        tests::test_uniform_distribution::<100_000_000, f32>(|| rng.gen_range(FLOAT_RANGE), DOUBLE_RANGE);
    }

    fn get_zero_seed() -> &'static [u8; 32] {
        unsafe { std::mem::transmute::<_, &[u8; 4 * 8]>(&SEED_ZERO) }
    }

    fn get_predefined_seed() -> &'static [u8; 32] {
        unsafe { std::mem::transmute::<_, &[u8; 4 * 8]>(&SEED_PI) }
    }

    fn create_with_zero_seed() -> Shishua {
        let seed = get_zero_seed();
        Shishua::from_seed(*seed)
    }

    fn create_with_predefined_seed() -> Shishua {
        let seed = get_predefined_seed();
        Shishua::from_seed(*seed)
    }

    const SEED_ZERO: [u64; 4] = [
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000,
    ];

    const SEED_PI: [u64; 4] = [
        0x243f6a8885a308d3,
        0x13198a2e03707344,
        0xa409382229f31d00,
        0x82efa98ec4e6c894,
    ];

    const SEED_ZERO_EXPECTED: [u8; 512] = [
        0x95, 0x5d, 0x96, 0xf9, 0x0f, 0xb4, 0xaa, 0x53, 0x09, 0x2d, 0x82, 0xe6, 0x3a, 0x7c, 0x09, 0xe2, 0x2c, 0xa5,
        0xa4, 0xa5, 0xa7, 0x5a, 0x5a, 0x39, 0xdc, 0x68, 0xb4, 0x12, 0x5d, 0xe7, 0xce, 0x2b, 0x6b, 0x6e, 0xfe, 0xf5,
        0x8b, 0xd9, 0xcc, 0x42, 0x12, 0xdd, 0x74, 0x4e, 0x81, 0xfd, 0x18, 0xb9, 0x58, 0xf0, 0x62, 0x5d, 0x38, 0xef,
        0xcc, 0x1b, 0x6f, 0xdb, 0x0d, 0xa3, 0x36, 0xf7, 0xe5, 0xee, 0x6b, 0xdb, 0xe8, 0xea, 0x5c, 0xda, 0x40, 0xc7,
        0x53, 0x44, 0xd0, 0xd5, 0xbf, 0xc1, 0xd5, 0x07, 0xe0, 0x2c, 0xf5, 0x12, 0x08, 0x71, 0x1b, 0xea, 0x88, 0x82,
        0xcf, 0xd6, 0xcc, 0xf7, 0x1d, 0x06, 0x62, 0xc7, 0x5e, 0xf1, 0x98, 0x5d, 0xf2, 0xc6, 0xd5, 0x6d, 0x3d, 0x2e,
        0x35, 0xda, 0xd6, 0x85, 0x3a, 0xc1, 0x76, 0xb7, 0x4d, 0xb7, 0xe0, 0x26, 0x51, 0x2d, 0xce, 0x34, 0x8b, 0xa6,
        0x03, 0xf1, 0x0e, 0xa2, 0x7a, 0x7f, 0xcb, 0x03, 0x8c, 0x71, 0xe2, 0xc7, 0x05, 0x7d, 0x8f, 0xef, 0x24, 0x94,
        0x51, 0x97, 0xa6, 0xdd, 0x60, 0x80, 0x98, 0xf9, 0xf4, 0xcc, 0x27, 0x5d, 0xd1, 0x97, 0x51, 0xad, 0x0f, 0x4b,
        0xf6, 0x18, 0x96, 0xc9, 0xc2, 0x84, 0x2e, 0x34, 0x60, 0x9e, 0x29, 0x16, 0x38, 0x4e, 0x71, 0x9f, 0x7f, 0x05,
        0x6c, 0x2a, 0x70, 0xf4, 0xb8, 0x59, 0x2c, 0x02, 0xd1, 0xd6, 0xf0, 0x91, 0x06, 0x5d, 0xac, 0x7e, 0xc8, 0xa7,
        0x5e, 0x28, 0x25, 0xfd, 0x08, 0x1e, 0x0d, 0xac, 0xbf, 0x1a, 0x32, 0xc2, 0x2e, 0x82, 0x39, 0x60, 0x6c, 0x41,
        0xf1, 0xb1, 0x3c, 0xd6, 0xb5, 0x9e, 0x04, 0xc4, 0x5a, 0xfb, 0xfe, 0xb3, 0x67, 0x00, 0xa9, 0xef, 0x25, 0x1c,
        0xf5, 0x72, 0xe1, 0xd7, 0x40, 0x85, 0xdb, 0xcc, 0x02, 0x79, 0x49, 0x1d, 0x77, 0x54, 0x96, 0x21, 0x85, 0x68,
        0x7a, 0xe8, 0x41, 0x02, 0xb2, 0x37, 0x02, 0x18, 0x98, 0x33, 0x5f, 0x44, 0x5d, 0x67, 0x3d, 0xcc, 0x82, 0xd0,
        0x3f, 0x78, 0x94, 0xdc, 0xc2, 0x87, 0x27, 0x39, 0xe4, 0x85, 0x3c, 0xb0, 0xc3, 0x3b, 0xa0, 0x33, 0x29, 0xf3,
        0x46, 0x8b, 0x93, 0xa5, 0x2b, 0x58, 0xb9, 0x42, 0x9a, 0x9b, 0xd1, 0x4b, 0xac, 0x37, 0x44, 0xdf, 0xee, 0x22,
        0x43, 0xd3, 0x0d, 0xe2, 0x11, 0xcf, 0x49, 0x0e, 0x56, 0xb5, 0x6c, 0x55, 0x40, 0xfc, 0x80, 0xf7, 0x68, 0xfa,
        0x47, 0x25, 0xe7, 0x5a, 0x6d, 0x3e, 0x8f, 0xe7, 0x74, 0xc1, 0x6a, 0x42, 0x8c, 0x42, 0x92, 0x79, 0xb0, 0x3f,
        0xad, 0x49, 0x17, 0x0f, 0xb3, 0x2a, 0xa8, 0x29, 0x00, 0x09, 0x64, 0xf1, 0xb1, 0xcb, 0xf3, 0x49, 0x22, 0x61,
        0xf0, 0xe7, 0x20, 0xdb, 0x11, 0x8f, 0x05, 0x3d, 0x50, 0xe6, 0x90, 0x4a, 0xc0, 0x76, 0x76, 0x62, 0x61, 0x43,
        0xfa, 0xaf, 0xe0, 0xbd, 0x4e, 0x24, 0x68, 0xf9, 0xae, 0x75, 0x1b, 0x58, 0x93, 0x81, 0x4b, 0x87, 0x3c, 0xdc,
        0x26, 0x3b, 0xfa, 0xa4, 0xca, 0xe7, 0x68, 0x0b, 0xf0, 0x37, 0x0c, 0x78, 0xd4, 0xd0, 0xcc, 0xaf, 0x54, 0xfd,
        0x93, 0x99, 0xba, 0x47, 0x3f, 0x88, 0x41, 0x7e, 0x61, 0xa6, 0xea, 0x72, 0xa7, 0xee, 0x89, 0xea, 0xd2, 0x4e,
        0x55, 0x99, 0x33, 0xcd, 0xef, 0x29, 0x3a, 0x89, 0xcf, 0xca, 0x6b, 0x9d, 0x7a, 0x5e, 0x72, 0x7e, 0x34, 0xb5,
        0xf7, 0xc8, 0x3f, 0xad, 0x44, 0xec, 0x25, 0xb7, 0x6b, 0xd7, 0x0e, 0x53, 0x06, 0xe0, 0x9d, 0x0d, 0x9b, 0x44,
        0xc1, 0xd5, 0xc1, 0x4f, 0x9d, 0xcb, 0x8b, 0xbf, 0xaf, 0x7e, 0x0f, 0x6f, 0xfa, 0xe0, 0x8c, 0x9a, 0x33, 0x4a,
        0x25, 0x37, 0x19, 0x11, 0x0d, 0xb5, 0x9d, 0x15, 0x09, 0x00, 0xe4, 0xaa, 0xef, 0x3d, 0x1a, 0x85, 0x3a, 0xc3,
        0xb0, 0x54, 0x03, 0xa7, 0x50, 0xec, 0x93, 0x8f,
    ];

    const SEED_PI_EXPECTED: [u8; 512] = [
        0xfa, 0x62, 0xa9, 0x26, 0xdc, 0x1f, 0xbf, 0x00, 0xf1, 0x3c, 0xe8, 0x68, 0x45, 0x9b, 0x6f, 0x74, 0x4b, 0xbf,
        0x2b, 0x57, 0x50, 0x5e, 0xd8, 0x16, 0x0e, 0x4e, 0xd9, 0x2a, 0x2e, 0xf6, 0x96, 0x5c, 0x01, 0xb5, 0xc9, 0xe7,
        0x9d, 0x84, 0xd8, 0xd9, 0x5f, 0x0d, 0xb7, 0x4a, 0x47, 0xf4, 0xac, 0xc8, 0x25, 0xcc, 0x0b, 0x2e, 0x3b, 0x90,
        0x03, 0x0a, 0x1d, 0x44, 0x3c, 0xd8, 0x27, 0xa8, 0x42, 0xe0, 0x6e, 0x8f, 0xa0, 0xc1, 0xb2, 0x8e, 0x18, 0x3d,
        0xe3, 0x93, 0x06, 0x79, 0x11, 0xdc, 0x92, 0x93, 0x0d, 0x85, 0xac, 0xde, 0xdb, 0xb3, 0x23, 0x04, 0xd0, 0xbe,
        0xfe, 0x74, 0xef, 0xbb, 0xbf, 0x19, 0xc1, 0x15, 0x0a, 0x34, 0x78, 0x45, 0xa2, 0x27, 0x93, 0xb7, 0xb2, 0x4d,
        0x4b, 0x4f, 0x6e, 0xb6, 0xc0, 0xdc, 0x42, 0x54, 0x6a, 0x9b, 0xcd, 0x50, 0x73, 0xfa, 0xa1, 0x9c, 0xb4, 0xd1,
        0xd2, 0x87, 0xf1, 0xd6, 0x97, 0x89, 0x88, 0xa7, 0x7d, 0xcd, 0x12, 0xe8, 0xfa, 0xa2, 0x78, 0x99, 0xc9, 0x2f,
        0x8f, 0xd5, 0x9e, 0x33, 0x7c, 0x42, 0xc6, 0xe9, 0x8b, 0x73, 0x48, 0x73, 0xfe, 0xfc, 0xef, 0x3a, 0xc5, 0x41,
        0x8b, 0x87, 0x3c, 0xfd, 0xc7, 0x3b, 0xff, 0xd8, 0x83, 0xb3, 0x38, 0x34, 0x8f, 0x4e, 0x3c, 0x10, 0x93, 0xcb,
        0x48, 0xab, 0xa8, 0x23, 0xd2, 0x3d, 0xa1, 0xec, 0x21, 0x69, 0xc9, 0x18, 0xe5, 0x61, 0x96, 0x93, 0x42, 0xbe,
        0x30, 0xe7, 0x8b, 0x48, 0x59, 0xed, 0xe4, 0x7c, 0x26, 0xb6, 0xc4, 0xdd, 0xbf, 0x36, 0x57, 0xea, 0x9d, 0x5f,
        0x1b, 0x05, 0xa5, 0xc2, 0x6c, 0x5e, 0x57, 0xec, 0xb1, 0x84, 0x2e, 0x16, 0x61, 0x11, 0x67, 0xa3, 0x89, 0xa8,
        0xda, 0xb6, 0x7a, 0x35, 0x51, 0xcb, 0x3a, 0x26, 0x4b, 0xe5, 0x39, 0xd3, 0x9d, 0x8d, 0xd8, 0x70, 0x73, 0x9f,
        0x9b, 0xab, 0x13, 0xe2, 0x7a, 0x49, 0x18, 0x32, 0x28, 0xc2, 0xac, 0xcd, 0xfa, 0x10, 0x73, 0x55, 0x28, 0xf8,
        0x18, 0x6c, 0x4e, 0x52, 0xdf, 0x54, 0xc8, 0x2c, 0xca, 0xd0, 0x48, 0x31, 0x10, 0x64, 0x68, 0xa4, 0x52, 0x7f,
        0xde, 0x74, 0x93, 0xc7, 0x73, 0x2d, 0xe8, 0x45, 0x74, 0x78, 0x4b, 0xeb, 0x3f, 0x5e, 0x0a, 0xe0, 0xa0, 0x2e,
        0x14, 0xcc, 0xaf, 0xa2, 0x98, 0xde, 0x47, 0x28, 0x26, 0x8d, 0x20, 0xf6, 0xe3, 0x8c, 0xe8, 0x02, 0x0d, 0xd3,
        0xaf, 0x39, 0x9c, 0x2e, 0xbf, 0x47, 0x81, 0x8d, 0x23, 0x75, 0x34, 0x7f, 0xa4, 0x5e, 0x3e, 0xb8, 0xd4, 0xa5,
        0xcd, 0x97, 0x0b, 0x0f, 0xa6, 0x41, 0x1d, 0x1f, 0x5d, 0x4f, 0xf6, 0xf2, 0x44, 0xaa, 0x2b, 0x66, 0x00, 0x65,
        0xbc, 0xa0, 0x71, 0xc8, 0xa9, 0x0b, 0x5e, 0x1f, 0xfb, 0x8e, 0x66, 0xf3, 0xa1, 0x16, 0x71, 0xa9, 0x92, 0x19,
        0x43, 0x0d, 0xd6, 0xa2, 0x38, 0xfd, 0xd1, 0xe5, 0x67, 0x29, 0xe8, 0x58, 0x8d, 0x20, 0x19, 0xa1, 0xca, 0x13,
        0x93, 0x01, 0xff, 0x72, 0x97, 0x23, 0x66, 0xae, 0x85, 0x80, 0x35, 0xd0, 0x74, 0x4e, 0x8f, 0xba, 0x30, 0x7c,
        0x61, 0xe6, 0xb0, 0xb4, 0x11, 0x6a, 0x29, 0x05, 0xc5, 0x0a, 0x27, 0x4e, 0x0b, 0xce, 0x96, 0xad, 0xfa, 0x41,
        0x5a, 0x14, 0x4f, 0xac, 0x24, 0x96, 0x32, 0xae, 0x94, 0x3f, 0x26, 0x61, 0x57, 0x61, 0xf9, 0xfd, 0x6d, 0x71,
        0x23, 0x33, 0x74, 0x17, 0xaa, 0x2f, 0xa9, 0xbd, 0x2e, 0x07, 0x01, 0xa8, 0x13, 0xed, 0x51, 0x48, 0x11, 0x37,
        0xc7, 0x51, 0x00, 0x7c, 0x9b, 0x76, 0x26, 0x67, 0x06, 0x57, 0x12, 0x94, 0xf8, 0xd7, 0x92, 0x0d, 0x4f, 0x7a,
        0x08, 0xb7, 0xbf, 0x54, 0x6e, 0x09, 0x29, 0x39, 0xf2, 0x53, 0xaa, 0x49, 0x81, 0xb2, 0x14, 0xee, 0xd2, 0x52,
        0x68, 0x4b, 0xe3, 0xc0, 0x4e, 0x1b, 0x75, 0xed,
    ];
}
