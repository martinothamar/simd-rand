#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub mod avx2;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512dq",
    target_feature = "avx512vl"
))]
pub mod avx512;
