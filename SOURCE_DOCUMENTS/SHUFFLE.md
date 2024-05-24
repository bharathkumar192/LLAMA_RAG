  4. Shuffle

## Shuffle

Purpose

The Shuffle masking format randomly shuffles values within a column.

Shuffle preserves data distribution. Suppose a column has 100 values, and all
values are either 21 or 10, and the value 21 appears 60 times and the value 10
appears 40 times, after shuffling this column, this count remains same.

Inputs

  * No input values are required.

Supported Data Types

  * Character
  * Numeric
  * Date

Characteristics

  * Supports Double-Byte Characters: Yes
  * Combinable: No
  * Deterministic: No
  * Reversible: No
  * Uniqueness: Yes, provided the column values are all unique

Example

In the following table, the values in the `SALARY` column are shuffled in the
`SHUFFLED_SALARY` column.

EMPLOYEE | JOB_CATEGORY | SALARY | SHUFFLED_SALARY  
---|---|---|---  
Alice | M | 90 | 70  
Bill | M | 88 | 57  
Carol | W | 72 | 88  
Denise | W | 57 | 45  
Eddie | W | 70 | 90  
Frank | W | 45 | 72
