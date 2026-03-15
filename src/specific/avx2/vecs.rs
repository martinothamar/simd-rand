use core::ops::{Deref, DerefMut};

#[derive(Default, Debug)]
#[repr(align(32))]
pub struct U64x4([u64; 4]);

#[derive(Default, Debug)]
#[repr(align(32))]
pub struct F64x4([f64; 4]);

impl U64x4 {
    #[inline(always)]
    #[must_use]
    pub const fn new(values: [u64; 4]) -> Self {
        Self(values)
    }
}

impl F64x4 {
    #[inline(always)]
    #[must_use]
    pub const fn new(values: [f64; 4]) -> Self {
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

impl From<[u64; 4]> for U64x4 {
    fn from(val: [u64; 4]) -> Self {
        Self::new(val)
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

impl From<[f64; 4]> for F64x4 {
    fn from(val: [f64; 4]) -> Self {
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
        assert_eq!(size_of::<__m256i>(), size_of::<U64x4>());
        assert_eq!(size_of::<__m256d>(), size_of::<F64x4>());
    }

    #[test]
    fn alignment() {
        assert!(align_of::<U64x4>() >= align_of::<__m256i>());
        assert!(align_of::<F64x4>() >= align_of::<__m256d>());
    }

    #[test]
    fn constructors_and_mutation_preserve_values() {
        let ints = [1, 2, 3, 4];
        let floats = [1.5, 2.5, 3.5, 4.5];

        let mut u64s = U64x4::new(ints);
        let mut f64s = F64x4::from(floats);

        assert_eq!(&*u64s, &ints);
        assert!(
            f64s.iter()
                .zip(floats)
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
        );

        u64s[0] = 9;
        f64s[3] = 9.5;

        assert_eq!(&*u64s, &[9, 2, 3, 4]);
        assert!(
            f64s.iter()
                .zip([1.5_f64, 2.5, 3.5, 9.5])
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
        );
    }
}
