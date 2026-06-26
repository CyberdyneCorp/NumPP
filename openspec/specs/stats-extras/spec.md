# stats-extras Specification

## Purpose
TBD - created by archiving change tier-a-completion. Update Purpose after archive.
## Requirements
### Requirement: Multi-dimensional histograms and weighted statistics
NumPP SHALL provide histogram2d, histogramdd, nanquantile and cov_weighted
matching numpy.

#### Scenario: stats extras match numpy
- WHEN histogram2d/histogramdd bin samples, nanquantile computes a nan-aware
  quantile, or cov_weighted computes a weighted covariance
- THEN the result equals numpy.histogram2d / numpy.histogramdd /
  numpy.nanquantile / numpy.cov(aweights=...)

