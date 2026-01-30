use std::{fmt::Debug, fmt::Display, ops::Range};

use num_traits::{Num, NumCast};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use rust_decimal_macros::dec;

pub(crate) const DOUBLE_RANGE: Range<f64> = 0.0..1.0;
pub(crate) const FLOAT_RANGE: Range<f32> = 0.0f32..1.0f32;

pub(crate) fn test_uniform_distribution<const SAMPLES: usize, T>(mut f: impl FnMut() -> T, range: Range<T>)
where
    T: Num + NumCast + Display + TryInto<Decimal>,
    <T as TryInto<Decimal>>::Error: Debug,
{
    // Even though T represents a generic floating point,
    // we still use f64 to make sure we dont lose too much precision

    let mut dist: Vec<Decimal> = Vec::with_capacity(SAMPLES);

    let range_start: Decimal = range.start.try_into().unwrap();
    let range_end: Decimal = range.end.try_into().unwrap();

    let range = range_start..range_end;

    let mut sum = Decimal::ZERO;
    for _ in 0..SAMPLES {
        let value: Decimal = f().try_into().unwrap();
        assert!(value >= range.start && value < range.end);
        sum += value;
        dist.push(value);
    }

    let samples_divisor: Decimal = SAMPLES.into();

    let mean: Decimal = sum / samples_divisor;

    let mut squared_diffs = Decimal::ZERO;
    for n in dist {
        let diff = (n - mean).powi(2);
        squared_diffs += diff;
    }

    // Statistical tests below
    // We expect all the metrics to deviate no more than DIFF_LIMIT
    // TODO: improve this testing..
    const DIFF_LIMIT: Decimal = dec!(0.001);

    // In uniform distribution, where the interval is a to b

    // the mean should be: μ = (a + b) / 2
    let expected_mean = (range.start + range.end - DIFF_LIMIT) / dec!(2.0);
    let mean_difference = (mean - expected_mean).abs();

    let variance = squared_diffs / samples_divisor;
    // the variance should be: σ2 = (b – a)2 / 12
    let expected_variance = (range.end - range.start).powi(2) / dec!(12.0);
    let variance_difference = (variance - expected_variance).abs();

    let stddev = variance.sqrt().unwrap();
    // The standard deviation should be: σ = √σ2
    let expected_stddev = expected_variance.sqrt().unwrap();
    let stddev_difference = (stddev - expected_stddev).abs();

    // If any of these metrics deviate by DIFF_LIMIT or more,
    // we should fail the test
    assert!(
        mean_difference <= DIFF_LIMIT,
        "Mean difference was more than {DIFF_LIMIT:.5}: {mean_difference:.5}. Expected mean: {expected_mean:.6}, actual mean: {mean:.6}"
    );
    assert!(
        variance_difference <= DIFF_LIMIT,
        "Variance difference was more than {DIFF_LIMIT:.5}: {variance_difference:.5}. Expected variance: {expected_variance:.6}, actual mean: {variance:.6}"
    );
    assert!(
        stddev_difference <= DIFF_LIMIT,
        "Std deviation difference was more than {DIFF_LIMIT:.5}: {stddev_difference:.5}. Expected std deviation: {expected_stddev:.6}, actual mean: {stddev:.6}"
    );
}

#[rustfmt::skip]
pub(crate) const REF_SEED_256: [u8; 128] = [
    1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0,
    2, 0, 0, 0, 0, 0, 0, 0, 
    2, 0, 0, 0, 0, 0, 0, 0, 
    2, 0, 0, 0, 0, 0, 0, 0, 
    2, 0, 0, 0, 0, 0, 0, 0, 
    3, 0, 0, 0, 0, 0, 0, 0,
    3, 0, 0, 0, 0, 0, 0, 0,
    3, 0, 0, 0, 0, 0, 0, 0,
    3, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0,
];

#[rustfmt::skip]
pub(crate) const REF_SEED_512: [u8; 256] = [
    1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0,
    2, 0, 0, 0, 0, 0, 0, 0, 
    2, 0, 0, 0, 0, 0, 0, 0, 
    2, 0, 0, 0, 0, 0, 0, 0, 
    2, 0, 0, 0, 0, 0, 0, 0, 
    2, 0, 0, 0, 0, 0, 0, 0, 
    2, 0, 0, 0, 0, 0, 0, 0, 
    2, 0, 0, 0, 0, 0, 0, 0, 
    2, 0, 0, 0, 0, 0, 0, 0, 
    3, 0, 0, 0, 0, 0, 0, 0,
    3, 0, 0, 0, 0, 0, 0, 0,
    3, 0, 0, 0, 0, 0, 0, 0,
    3, 0, 0, 0, 0, 0, 0, 0,
    3, 0, 0, 0, 0, 0, 0, 0,
    3, 0, 0, 0, 0, 0, 0, 0,
    3, 0, 0, 0, 0, 0, 0, 0,
    3, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0,
];
