use core::{arch::x86_64::*, mem::transmute};

use super::vecs::*;

pub trait SimdRand {
    fn next_m512i(&mut self) -> __m512i;

    #[allow(clippy::items_after_statements)]
    #[inline(always)]
    fn next_m512d(&mut self) -> __m512d {
        unsafe {
            let v = self.next_m512i();

            let lhs = _mm512_cvtepu64_pd(_mm512_srli_epi64::<11>(v));

            // PERF: This is precomputed based on the constants from the formula above
            // I found no other efficient (and succint) constant way of representing the RHS.
            // setzero and constants used in shifts like 11 above are automatically constant folded.
            // Writing out the actual formula ended up not being constant folded by the compiler.
            const RHS_FACTOR: [f64; 8] = [1.1102230246251565E-16; 8];
            const RHS: __m512d = unsafe { transmute::<[f64; 8], __m512d>(RHS_FACTOR) };

            _mm512_mul_pd(lhs, RHS)
        }
    }

    #[inline(always)]
    fn next_u64x8(&mut self) -> U64x8 {
        unsafe {
            let v = self.next_m512i();
            let mut vector = U64x8::default();
            _mm512_store_epi64(core::ptr::from_mut(&mut vector).cast::<i64>(), v);
            vector
        }
    }

    #[inline(always)]
    fn next_f64x8(&mut self) -> F64x8 {
        unsafe {
            let v = self.next_m512d();
            let mut vector = F64x8::default();
            _mm512_store_pd(vector.as_mut_ptr(), v);
            vector
        }
    }
}
