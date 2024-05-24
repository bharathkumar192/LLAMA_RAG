  4. Preserve Original Data

## Preserve Original Data

Purpose

The Preserve Original Data masking format retains the original values in a
column. It is useful in conditional masking when you want to preserve a subset
of values in a column and mask the remaining values using some other masking
formats.

See Also:

[Example 1: Protecting Sensitive Identifiers Across Diverse Geographic
Regions](conditional-masking.html#GUID-3C52EE0C-9FBE-498D-AC8F-B59C259A93C0
"Learn how conditional masking can be used to mask unique personal identifiers
based on country.")

Inputs

  * No inputs are required.

Characteristics

  * Supports Double-Byte Characters: Yes
  * Combinable: No
  * Deterministic: Yes
  * Reversible: Does not apply
  * Uniqueness: If the original values are unique, they will remain unique after masking.

Example

Assume that a table has a `SALARY` column that you want to mask by using the
`EMPLOYEE ID` column in a condition. If the `EMPLOYEE ID` values are less than
100, you want to keep them. If they are from 100 to 199, you want to use the
fixed number 100000. Any `EMPLOYEE ID` greater than or equal to 200, you want
to use a random number between 30000 and 500000.

The masking logic for the `SALARY` column might look like the following:

    
    
    EMPLOYEE_ID < 100 
          PRESERVE ORIGINAL DATA
    EMPLOYEE_ID < 200
          FIXED NUMBER 100000
    EMPLOYEE_ID >= 200
          RANDOM NUMBER [Start Value: 30000 End Value: 500000]
