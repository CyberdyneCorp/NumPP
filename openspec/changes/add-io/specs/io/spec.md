# io Specification

## ADDED Requirements

### Requirement: NPY save and load

NumPP SHALL provide `save(path, array)` and `load(path)` reading and writing the
NumPy NPY format (header magic, version, descr/fortran_order/shape dict, 64-byte
aligned header), for all supported dtypes and both C and Fortran order, such that
files round-trip with `numpy.save`/`numpy.load` in both directions.

#### Scenario: Round trip through numpy
- WHEN an array is saved with `numpp::save` and loaded with `numpy.load`
- THEN the loaded array equals the original

#### Scenario: Load a numpy file
- WHEN a file written by `numpy.save` (C or Fortran order) is read with `numpp::load`
- THEN the result equals the original array with correct dtype and shape

### Requirement: NPZ archives

NumPP SHALL provide `savez`/`savez_compressed` writing a named set of arrays as a
ZIP of NPY members, and `load` returning the name->array map, interoperating with
`numpy.savez`/`numpy.load`.

#### Scenario: NPZ round trip
- WHEN multiple named arrays are saved with `savez` and loaded back
- THEN each array equals its original, keyed by name

### Requirement: Array printing

NumPP SHALL provide `array_str` and `array_repr` reproducing numpy's formatting:
per-dtype element formatting, default precision 8, sign handling, fixed vs
scientific selection, summarization for large arrays (threshold/edgeitems), nested
bracket layout, and the `array([...])` repr with a `dtype=` suffix for non-default
dtypes.

#### Scenario: repr matches numpy
- WHEN `array_repr` is computed for a representative array
- THEN it equals `repr(numpy_array)`

#### Scenario: Summarized large array
- WHEN `array_str` is computed for an array larger than the threshold
- THEN it shows edge items with an ellipsis, matching numpy
