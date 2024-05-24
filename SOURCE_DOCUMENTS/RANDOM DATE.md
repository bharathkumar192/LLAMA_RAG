  4. Random Date

## Random Date

Purpose

The Random Date masking format generates random dates within a date range to
replace the original column values.

Inputs

  * Start Date: The lower bound of the range within which random dates should be generated 
  * End Date: The upper bound of the range within which random dates should be generated 

The inputs should be in format `YYYY-MM-DD`. Start Date should be less than or
equal to End Date.

Supported Data Types

  * Date

Characteristics

  * Supports Double-Byte Characters: Does not apply
  * Combinable: Yes
  * Deterministic: No
  * Reversible: No
  * Uniqueness: Yes. The total number of distinct values in the specified range must be greater than or equal to the number of values in the column.

Example

To generate random dates between January 1, 2016 and December 31, 2019 for the
column `BIRTH_DATE`, you can use the Random Date masking format with the dates
entered as the two parameters.

The following table shows the original `BIRTH_DATE` column and the
`MASKED_BIRTH_DATE` column.

BIRTH_DATE | MASKED_BIRTH_DATE  
---|---  
01/01/2010 | 02/09/2016  
05/02/2018 | 01/02/2018  
09/11/2009 | 08/10/2019
