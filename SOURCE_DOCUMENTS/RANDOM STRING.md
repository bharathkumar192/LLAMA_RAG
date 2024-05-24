  4. Random String

## Random String

Purpose

The Random String masking format replaces column data with random strings of
length within the specified range. The generated strings consist of lowercase
letters only.

Inputs

  * Start Length: The minimum number of characters that the generated strings should have. 
  * End Length: The maximum number of characters that the generated strings should have. 

The inputs can be any integers greater than zero. Start Length must be less
than or equal to End Length. The inputs should be valid for the column size.

Supported Data Types

  * Character

Characteristics

  * Supports Double-Byte Characters: No
  * Combinable: Yes
  * Deterministic: No
  * Reversible: No
  * Uniqueness: Yes. The number of distinct values in the specified range must be greater than or equal to the number of values in the column.

Example

Suppose you have a `FIRST_NAME` column and you want to mask it with random
names of length from 5 through 15. You can use the Random String masking
format to generate strings of desired length by entering these two values as
input parameters.
