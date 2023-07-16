# SIMD PRNG in Rust

SIMD implementations of common PRNGs in Rust.
Currently

* Shishua (buffered, AVX2)
* Xoshiro256++ (x4, AVX2)

Plan is to start of implementing PRNGs targeting specific instruction sets (such as AVX__N__) in the `simd_prng::specific` module,
and then maybe investigate portable implementations of these if possible.
The PRNGs implement the `SeedableRng` trait from `rand_core`.

## Performance

### Benchmarks

#### Shishua

Each iterations extracts 64 u64 values in a loop. This equates to 512 bytes per iteration.

```
shishua/Shishua u64x4 - Time
                        time:   [19.756 ns 19.873 ns 19.984 ns]
                        thrpt:  [5.9654 GiB/s 5.9987 GiB/s 6.0340 GiB/s]
slope  [19.756 ns 19.984 ns] R^2            [0.9368117 0.9371568]
mean   [19.665 ns 19.845 ns] std. dev.      [413.14 ps 502.91 ps]
median [19.719 ns 19.986 ns] med. abs. dev. [358.84 ps 645.57 ps]


shishua/SmallRng u64x4 - Time
                        time:   [62.098 ns 62.213 ns 62.361 ns]
                        thrpt:  [1.9116 GiB/s 1.9161 GiB/s 1.9197 GiB/s]
Found 8 outliers among 100 measurements (8.00%)
  4 (4.00%) high mild
  4 (4.00%) high severe
slope  [62.098 ns 62.361 ns] R^2            [0.9721204 0.9718331]
mean   [62.257 ns 62.836 ns] std. dev.      [533.84 ps 2.4491 ns]
median [62.058 ns 62.208 ns] med. abs. dev. [254.02 ps 481.92 ps]


shishua/Shishua f64x4 - Time
                        time:   [20.712 ns 20.836 ns 20.974 ns]
                        thrpt:  [5.6836 GiB/s 5.7213 GiB/s 5.7557 GiB/s]
Found 11 outliers among 100 measurements (11.00%)
  3 (3.00%) low mild
  4 (4.00%) high mild
  4 (4.00%) high severe
slope  [20.712 ns 20.974 ns] R^2            [0.8846125 0.8836745]
mean   [20.902 ns 21.483 ns] std. dev.      [543.61 ps 2.3143 ns]
median [20.928 ns 21.031 ns] med. abs. dev. [200.13 ps 611.67 ps]


shishua/SmallRng f64x4 - Time
                        time:   [88.399 ns 88.661 ns 88.952 ns]
                        thrpt:  [1.3402 GiB/s 1.3445 GiB/s 1.3485 GiB/s]
Found 3 outliers among 100 measurements (3.00%)
  3 (3.00%) high mild
slope  [88.399 ns 88.952 ns] R^2            [0.9783462 0.9780763]
mean   [88.817 ns 89.289 ns] std. dev.      [992.72 ps 1.4375 ns]
median [88.522 ns 89.344 ns] med. abs. dev. [978.24 ps 1.5708 ns]
```

#### Xoshiro256++

```
xoshiro/Xoshiro256PlusPlus u64x4 - Time
                        time:   [63.423 ns 63.660 ns 63.935 ns]
                        thrpt:  [7.4581 GiB/s 7.4904 GiB/s 7.5183 GiB/s]
Found 3 outliers among 100 measurements (3.00%)
  1 (1.00%) high mild
  2 (2.00%) high severe
slope  [63.423 ns 63.935 ns] R^2            [0.9612042 0.9605413]
mean   [63.690 ns 64.210 ns] std. dev.      [987.47 ps 1.6754 ns]
median [63.437 ns 64.032 ns] med. abs. dev. [890.68 ps 1.3204 ns]


xoshiro/Xoshiro256PlusPlusX4 u64x4 - Time
                        time:   [17.376 ns 17.458 ns 17.534 ns]
                        thrpt:  [27.195 GiB/s 27.313 GiB/s 27.442 GiB/s]
Found 5 outliers among 100 measurements (5.00%)
  5 (5.00%) low mild
slope  [17.376 ns 17.534 ns] R^2            [0.9595574 0.9599166]
mean   [17.420 ns 17.530 ns] std. dev.      [235.50 ps 324.30 ps]
median [17.547 ns 17.612 ns] med. abs. dev. [118.20 ps 222.58 ps]


xoshiro/Xoshiro256PlusPlus f64x4 - Time
                        time:   [89.209 ns 89.434 ns 89.675 ns]
                        thrpt:  [5.3174 GiB/s 5.3317 GiB/s 5.3452 GiB/s]
Found 7 outliers among 100 measurements (7.00%)
  6 (6.00%) high mild
  1 (1.00%) high severe
slope  [89.209 ns 89.675 ns] R^2            [0.9818900 0.9817640]
mean   [89.526 ns 90.119 ns] std. dev.      [1.1901 ns 1.8055 ns]
median [89.361 ns 90.081 ns] med. abs. dev. [821.16 ps 1.4784 ns]


xoshiro/Xoshiro256PlusPlusX4 f64x4 - Time
                        time:   [22.621 ns 22.729 ns 22.831 ns]
                        thrpt:  [20.885 GiB/s 20.979 GiB/s 21.079 GiB/s]
Found 1 outliers among 100 measurements (1.00%)
  1 (1.00%) low mild
slope  [22.621 ns 22.831 ns] R^2            [0.9544029 0.9546700]
mean   [22.770 ns 22.930 ns] std. dev.      [345.84 ps 468.65 ps]
median [22.847 ns 23.033 ns] med. abs. dev. [255.56 ps 473.33 ps]
```

#### Comparison

```
comparison/Shishua u64x4 - Time
                        time:   [19.315 ns 19.425 ns 19.561 ns]
                        thrpt:  [24.377 GiB/s 24.547 GiB/s 24.687 GiB/s]
Found 4 outliers among 100 measurements (4.00%)
  1 (1.00%) high mild
  3 (3.00%) high severe
slope  [19.315 ns 19.561 ns] R^2            [0.8895538 0.8875578]
mean   [19.516 ns 19.869 ns] std. dev.      [402.88 ps 1.2834 ns]
median [19.536 ns 19.732 ns] med. abs. dev. [235.65 ps 508.25 ps]


comparison/Xoshiro256PlusPlusX4 u64x4 - Time
                        time:   [16.755 ns 16.796 ns 16.837 ns]
                        thrpt:  [28.321 GiB/s 28.390 GiB/s 28.459 GiB/s]
slope  [16.755 ns 16.837 ns] R^2            [0.9878917 0.9878922]
mean   [16.766 ns 16.839 ns] std. dev.      [159.32 ps 210.83 ps]
median [16.721 ns 16.840 ns] med. abs. dev. [161.83 ps 257.43 ps]
```

### Notes

Prereqs for disassembly:

```sh
cargo install cargo-binutils
rustup component add llvm-tools-preview
```

Then you can run `make dasm`
