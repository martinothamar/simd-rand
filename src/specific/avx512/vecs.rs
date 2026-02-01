use core::ops::{Deref, DerefMut};

#[derive(Default, Debug, PartialEq, Eq)]
#[repr(align(64))]
pub struct U64x8([u64; 8]);

#[derive(Default, Debug, PartialEq)]
#[repr(align(64))]
pub struct F64x8([f64; 8]);

impl U64x8 {
    #[inline(always)]
    #[must_use]
    pub const fn new(values: [u64; 8]) -> Self {
        Self(values)
    }
}

impl F64x8 {
    #[inline(always)]
    #[must_use]
    pub const fn new(values: [f64; 8]) -> Self {
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

impl From<[u64; 8]> for U64x8 {
    fn from(val: [u64; 8]) -> Self {
        Self::new(val)
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

impl From<[f64; 8]> for F64x8 {
    fn from(val: [f64; 8]) -> Self {
        Self::new(val)
    }
}

#[cfg(test)]
mod tests {
    use core::{arch::x86_64::*, mem::size_of};

    use super::*;

    #[test]
    fn size() {
        assert_eq!(size_of::<__m512i>(), size_of::<U64x8>());
        assert_eq!(size_of::<__m512d>(), size_of::<F64x8>());
    }
}
