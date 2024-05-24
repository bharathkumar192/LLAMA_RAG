  4. Random List

## Random List

Purpose

The Random List masking format randomly selects values from a list of values
to replace the original column values.

Inputs

  * List of Values: A comma-separated list of values that should be used to replace column values. The data type of each value in the list must be compatible with the data type of the column. If using a list of dates, the dates should be in format `YYYY-MM-DD`. The number of entries in the list cannot be more than 999. 

Supported Data Types

  * Character
  * Numeric
  * Date

Characteristics

  * Supports Double-Byte Characters: Yes
  * Combinable: Yes
  * Deterministic: No
  * Reversible: No
  * Uniqueness: Yes. The input list must have unique values, and the number of values in the list must be greater than or equal to the number of values in the column to be masked.

Example 1

Suppose you have a column with values 10, 20, 30, 40, 50. You can replace
these values with random values from an input list (99, 100, 101, 102, 103) by
using the Random List masking format. The following table compares the values
in the original column (`ORIGINAL`) to the values after the first masking job
(`MASK1`) and second masking job (`MASK2`). Notice that the masked values
change each time the masking job runs.

ORIGINAL | MASK1 | MASK2  
---|---|---  
10 | 101 | 100  
20 | 103 | 99  
30 | 100 | 101  
40 | 99 | 102  
50 | 102 | 103  
  
Example 2

The following table shows you how a `MARITAL_STATUS` column, consisting of
five distinct values, gets masked with the Random List masking format. The
list of values for the masking format is Single, Married, and Divorced.

MARITAL_STATUS | MASKED_MARITAL_STATUS  
---|---  
Single | Divorced  
Married | Single  
Windowed | Divorced  
Single | Married  
Divorced | Married  
Separated | Single
