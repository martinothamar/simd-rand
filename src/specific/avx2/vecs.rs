use std::ops::{Deref, DerefMut};

#[derive(Default, Debug)]
#[repr(align(32))]
pub struct U64x4([u64; 4]);

#[derive(Default, Debug)]
#[repr(align(32))]
pub struct F64x4([f64; 4]);

impl U64x4 {
    #[inline(always)]
    pub fn new(values: [u64; 4]) -> Self {
        Self(values)
    }
}

impl F64x4 {
    #[inline(always)]
    pub fn new(values: [f64; 4]) -> Self {
        Self(values)
    }
}

impl Deref for U64x4 {
    type Target = [u64; 4];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for U64x4 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Into<U64x4> for [u64; 4] {
    fn into(self) -> U64x4 {
        U64x4::new(self)
    }
}

impl Deref for F64x4 {
    type Target = [f64; 4];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for F64x4 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Into<F64x4> for [f64; 4] {
    fn into(self) -> F64x4 {
        F64x4::new(self)
    }
}

#[cfg(test)]
mod tests {
    use std::{arch::x86_64::*, mem::size_of};

    use serial_test::parallel;

    use super::*;

    #[test]
    #[parallel]
    fn size() {
        assert_eq!(size_of::<__m256i>(), size_of::<U64x4>());
        assert_eq!(size_of::<__m256d>(), size_of::<F64x4>());
    }
}
