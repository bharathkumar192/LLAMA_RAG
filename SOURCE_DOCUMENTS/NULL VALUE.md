  4. Null Value

## Null Value

Purpose

The Null Value masking format replaces column data with `NULL`. The column
being masked must be allowed to contain null values.

Inputs

  * No inputs are required.

Supported Data Types

  * Character
  * Numeric
  * Date
  * Large Object (LOB)

Characteristics

  * Supports Double-Byte Characters: Does not apply
  * Combinable: No
  * Deterministic: Yes
  * Reversible: No
  * Uniqueness: No

Example

Suppose you have a column named `SALARY` that contains salary information and
you want to replace those numbers with `NULL`. You can apply the Null Value
masking format to the `SALARY` column.
