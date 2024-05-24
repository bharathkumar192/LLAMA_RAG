  4. Substring

## Substring

Purpose

The Substring masking format extracts a portion of the original column value,
and uses that to replace the original value. This masking format is similar to
the `SUBSTR` database function.

Inputs

  * Start Position: The starting position in the original string from where the substring should be extracted. The start position can be either a positive or a negative integer. If the start position is negative, the counting starts from the end of the string. 
  * Length: The number of characters that you want in the substring. It should be an integer and greater than zero. 

Supported Data Types

  * Character

Characteristics

  * Supports Double-Byte Characters: Yes
  * Combinable: Yes
  * Deterministic: Yes
  * Reversible: No
  * Uniqueness: No

Example

Suppose an original column value is `abcd`. A substring with a start position
of 2 and length of 3 generates a masked string of `bcd`. A substring with
start position of -2 and length of 3 generates a masked string of `cd`.
