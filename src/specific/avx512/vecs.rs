use std::ops::{Deref, DerefMut};

#[derive(Default, Debug, PartialEq)]
#[repr(align(64))]
pub struct U64x8([u64; 8]);

#[derive(Default, Debug, PartialEq)]
#[repr(align(64))]
pub struct F64x8([f64; 8]);

impl U64x8 {
    #[inline(always)]
    pub fn new(values: [u64; 8]) -> Self {
        Self(values)
    }
}

impl F64x8 {
    #[inline(always)]
    pub fn new(values: [f64; 8]) -> Self {
        Self(values)
    }
}

impl Deref for U64x8 {
    type Target = [u64; 8];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for U64x8 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Into<U64x8> for [u64; 8] {
    fn into(self) -> U64x8 {
        U64x8::new(self)
    }
}

impl Deref for F64x8 {
    type Target = [f64; 8];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for F64x8 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Into<F64x8> for [f64; 8] {
    fn into(self) -> F64x8 {
        F64x8::new(self)
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
        assert_eq!(size_of::<__m512i>(), size_of::<U64x8>());
        assert_eq!(size_of::<__m512d>(), size_of::<F64x8>());
    }
}
