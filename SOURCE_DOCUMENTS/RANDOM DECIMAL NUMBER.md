  4. Random Decimal Number

## Random Decimal Number

Purpose

The Random Decimal Number masking format generates random decimal numbers
within a value range to replace the original column values.

Inputs

  * Start Number: The lower bound of the range within which decimal numbers should be generated 
  * End Number: The upper bound of the range within which decimal numbers should be generated 

The inputs can be any decimal numbers, including negative numbers. Start
Number must be less than or equal to End Number. They should be valid for the
column size.

Supported Data Types

  * Character
  * Numeric

Characteristics

  * Supports Double-Byte Characters: Does not apply
  * Combinable: Yes
  * Deterministic: No
  * Reversible: No
  * Uniqueness: Yes. The total number of distinct values in the specified range must be greater than or equal to the number of values in the column.

Example

Suppose you have a `HEIGHT` column and you want to generate random heights
from 0.5 through 2.2 meters. You can use the Random Decimals Number masking
format to generate decimal numbers from 0.5 through 2.2, including those
values.
