use rand_core::RngCore;

pub struct FixedBytesRng<const N: usize> {
    bytes: [u8; N],
    offset: usize,
}

impl<const N: usize> FixedBytesRng<N> {
    #[must_use]
    pub const fn new(bytes: [u8; N]) -> Self {
        Self { bytes, offset: 0 }
    }
}

impl<const N: usize> RngCore for FixedBytesRng<N> {
    fn next_u32(&mut self) -> u32 {
        rand_core::impls::next_u32_via_fill(self)
    }

    fn next_u64(&mut self) -> u64 {
        rand_core::impls::next_u64_via_fill(self)
    }

    fn fill_bytes(&mut self, dst: &mut [u8]) {
        let end = self.offset + dst.len();
        assert!(end <= self.bytes.len());
        dst.copy_from_slice(&self.bytes[self.offset..end]);
        self.offset = end;
    }
}
