  4. Random Digits

## Random Digits

Purpose

The Random Digits masking format generates random digits of length within a
range. It pads to the appropriate length in a string, but does not pad when
used for a number column. This format is a complementary type of Random
Number, which is not padded.

Inputs

  * Start Length: The minimum number of digits each masked value should have 
  * End Length: The maximum number of digits each masked value should have 

Supported Data Types

  * Character
  * Numeric

Characteristics

  * Supports Double-Byte Characters: Does not apply
  * Combinable: Yes
  * Deterministic: No
  * Reversible: No
  * Uniqueness: Yes, however, if you do not specify a sufficient length range, you can run out of unique values within the range.

Example

For a random digit with a length of [5,5], an integer from zero through 99999
is randomly generated and left padded with zeros to satisfy the length and
uniqueness requirement.
