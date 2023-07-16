use std::{fmt::Display, ops::Range};

use num_traits::{Float, ToPrimitive};

pub(crate) const DOUBLE_RANGE: Range<f64> = 0.0..1.0;
pub(crate) const FLOAT_RANGE: Range<f32> = 0.0f32..1.0f32;

pub(crate) fn test_uniform_distribution<const SAMPLES: usize, T: Float + Display>(
    // f: fn(&mut Shishua) -> T,
    mut f: impl FnMut() -> T,
    range: Range<f64>,
) {
    // Even though T represents a generic floating point,
    // we still use f64 to make sure we dont lose too much precision

    let mut dist = Vec::with_capacity(SAMPLES);

    let mut sum = 0.0;
    for _ in 0..SAMPLES {
        let value = f().to_f64().unwrap();
        assert!(value >= range.start && value < range.end);
        sum = sum + value;
        dist.push(value);
    }

    let samples_divisor = SAMPLES.to_f64().unwrap();

    let mean = sum / samples_divisor;

    let mut squared_diffs = 0.0;
    for n in dist {
        let diff = (n - mean).powi(2);
        squared_diffs += diff;
    }

    // Statistical tests below
    // We expect all the metrics to deviate no more than DIFF_LIMIT
    const DIFF_LIMIT: f64 = 0.00005;

    // In uniform distribution, where the interval is a to b

    // the mean should be: μ = (a + b) / 2
    let expected_mean = (range.start + range.end) / 2.0;
    let mean_difference = (mean - expected_mean).abs();

    let variance = squared_diffs / samples_divisor;
    // the variance should be: σ2 = (b – a)2 / 12
    let expected_variance = (range.end - range.start).powi(2) / 12.0;
    let variance_difference = (variance - expected_variance).abs();

    let stddev = variance.sqrt();
    // The standard deviation should be: σ = √σ2
    let expected_stddev = expected_variance.sqrt();
    let stddev_difference = (stddev - expected_stddev).abs();

    // If any of these metrics deviate by DIFF_LIMIT or more,
    // we should fail the test
    assert!(
        mean_difference <= DIFF_LIMIT,
        "Mean difference was more than {DIFF_LIMIT:.5}: {mean_difference:.5}. Expected mean: {expected_mean:.2}"
    );
    assert!(variance_difference <= DIFF_LIMIT, "Variance difference was more than {DIFF_LIMIT:.5}: {variance_difference:.5}. Expected variance: {expected_variance:.2}");
    assert!(stddev_difference <= DIFF_LIMIT, "Std deviation difference was more than {DIFF_LIMIT:.5}: {stddev_difference:.5}. Expected std deviation: {expected_stddev:.2}");
}
