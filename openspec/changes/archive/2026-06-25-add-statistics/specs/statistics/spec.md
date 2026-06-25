# statistics Specification

## ADDED Requirements

### Requirement: Cumulative and difference routines
NumPP SHALL provide cumsum, cumprod, nancumsum, nancumprod, diff, ediff1d,
gradient and ptp matching numpy (axis handling, dtype rules, NaN treatment).

#### Scenario: cumsum matches numpy
- WHEN cumsum is applied along an axis or to the flattened array
- THEN the result equals numpy.cumsum including dtype (bool→int64, int/float preserved)

#### Scenario: diff and gradient match numpy
- WHEN diff(n, axis) or gradient is applied
- THEN the result equals numpy.diff / numpy.gradient

### Requirement: Order statistics with linear interpolation
NumPP SHALL provide median, percentile, quantile, nanmedian and nanpercentile
using numpy's default linear interpolation, along an axis or flattened.

#### Scenario: percentile matches numpy
- WHEN percentile(q) is computed along an axis
- THEN the result equals numpy.percentile with linear interpolation
- AND the nan* variants ignore NaN values

### Requirement: Weighted averages, correlation and binning
NumPP SHALL provide average (optionally weighted), cov, corrcoef, digitize,
nanargmin and nanargmax matching numpy defaults (cov rowvar=True/ddof=1).

#### Scenario: cov and corrcoef match numpy
- WHEN cov/corrcoef are applied to a set of variables
- THEN the result equals numpy.cov / numpy.corrcoef

#### Scenario: digitize matches numpy
- WHEN digitize is applied with ascending bins and a right flag
- THEN the bin indices equal numpy.digitize
