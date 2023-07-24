use std::simd::{f64x4, f64x8, u64x4, u64x8};

pub trait SimdRandX4 {
    fn next_u64x4(&mut self) -> u64x4;

    #[inline(always)]
    fn next_f64x4(&mut self) -> f64x4 {
        let v = self.next_u64x4();

        f64x4::from_array([
            (v[0] >> 11) as f64 * (1.0 / (1u64 << 53) as f64),
            (v[1] >> 11) as f64 * (1.0 / (1u64 << 53) as f64),
            (v[2] >> 11) as f64 * (1.0 / (1u64 << 53) as f64),
            (v[3] >> 11) as f64 * (1.0 / (1u64 << 53) as f64),
        ])
    }
}

pub trait SimdRandX8 {
    fn next_u64x8(&mut self) -> u64x8;

    #[inline(always)]
    fn next_f64x8(&mut self) -> f64x8 {
        let v = self.next_u64x8();

        f64x8::from_array([
            (v[0] >> 11) as f64 * (1.0 / (1u64 << 53) as f64),
            (v[1] >> 11) as f64 * (1.0 / (1u64 << 53) as f64),
            (v[2] >> 11) as f64 * (1.0 / (1u64 << 53) as f64),
            (v[3] >> 11) as f64 * (1.0 / (1u64 << 53) as f64),
            (v[4] >> 11) as f64 * (1.0 / (1u64 << 53) as f64),
            (v[5] >> 11) as f64 * (1.0 / (1u64 << 53) as f64),
            (v[6] >> 11) as f64 * (1.0 / (1u64 << 53) as f64),
            (v[7] >> 11) as f64 * (1.0 / (1u64 << 53) as f64),
        ])
    }
}
