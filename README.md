# Shishua for Rust

A Rust implementation of Shishua, 
[the worlds fastest PRNG](https://espadrine.github.io/blog/posts/shishua-the-fastest-prng-in-the-world.html) ([GitHub](https://github.com/espadrine/shishua)).

## Limitations

* Only AVX2 path
* Only x86_64

## Debuggin/inspection

```sh
cargo install cargo-binutils
rustup component add llvm-tools-preview
```
