  4. Random Number

## Random Number

Purpose

The Random Number masking format generates random integers within a specified
range to replace column data.

Inputs

  * Start Number: The lower bound of the range within which the integers should be generated. 
  * End Number: The upper bound of the range within which the integers should be generated. 

The inputs can be any integers, including negative integers. Start Number must
be less than or equal to End Number. They should be valid for the column size.

Supported Data Types

  * Character
  * Numeric

Characteristics

  * Supports Double-Byte Characters: Does not apply
  * Combinable: Yes
  * Deterministic: No
  * Reversible: No
  * Uniqueness: Yes. The number of distinct values in the specified range must be greater than or equal to the number of values in the column.

Example

Suppose you have an `EMPLOYEE_AGE` column and you want to generate random ages
from 21 through 65. You can use the Random Number masking format to generate
random integers from 21 through 65, including those values.

The following table shows the original `EMPLOYEE_AGE` column and the
`MASKED_EMPLOYEE_AGE` column.

EMPLOYEE_AGE | MASKED_EMPLOYEE_AGE  
---|---  
21 | 59  
35 | 22  
51 | 43  
28 | 38  
64 | 61  
75 | 21
