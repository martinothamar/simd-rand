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
    use core::{
        arch::x86_64::*,
        mem::{align_of, size_of},
    };

    use super::*;

    #[test]
    fn size() {
        assert_eq!(size_of::<__m512i>(), size_of::<U64x8>());
        assert_eq!(size_of::<__m512d>(), size_of::<F64x8>());
    }

    #[test]
    fn alignment() {
        assert!(align_of::<U64x8>() >= align_of::<__m512i>());
        assert!(align_of::<F64x8>() >= align_of::<__m512d>());
    }

    #[test]
    fn constructors_and_mutation_preserve_values() {
        let ints = [1, 2, 3, 4, 5, 6, 7, 8];
        let floats = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5];

        let mut u64s = U64x8::new(ints);
        let mut f64s = F64x8::from(floats);

        assert_eq!(&*u64s, &ints);
        assert!(
            f64s.iter()
                .zip(floats)
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
        );

        u64s[0] = 9;
        f64s[7] = 9.5;

        assert_eq!(&*u64s, &[9, 2, 3, 4, 5, 6, 7, 8]);
        assert!(
            f64s.iter()
                .zip([1.5_f64, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 9.5])
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
        );
    }
}
