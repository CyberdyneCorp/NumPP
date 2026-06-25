# array-manipulation Specification

## ADDED Requirements

### Requirement: Joining and splitting
NumPP SHALL provide concatenate, stack, hstack, vstack, dstack, column_stack and
array_split/split/hsplit/vsplit matching numpy (dtype promotion, axis handling,
even/uneven splits).

#### Scenario: concatenate matches numpy
- WHEN concatenate is applied to arrays along an axis
- THEN the result equals numpy.concatenate including dtype promotion

### Requirement: Tiling, rearranging, editing
NumPP SHALL provide tile, repeat, flip/fliplr/flipud, roll, rot90, moveaxis,
atleast_{1,2,3}d, append, insert, delete_, resize, and pad (constant/edge),
matching numpy.

#### Scenario: flip and roll match numpy
- WHEN flip/roll are applied with a given axis
- THEN the result equals numpy.flip / numpy.roll
