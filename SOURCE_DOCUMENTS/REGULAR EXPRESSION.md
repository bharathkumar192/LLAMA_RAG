  4. Regular Expression

## Regular Expression

Purpose

The Regular Expression masking format gives you the flexibility to use regular
expressions to search for sensitive data in a column of Large Object data type
(LOBs include BLOBs, CLOBs, NCLOBs), and replace the data with a fixed string,
fixed number, null value, or SQL expression. You can also use this masking
format for columns of `VARCHAR2` type to mask parts of strings.

Inputs

  * Regular Expression: The pattern that should be used to search for sensitive data 
  * Replace With: The value that should be used to replace the data matching the regular expression 

Supported Data Types

  * Character
  * Numeric
  * Large Object (LOB)

Characteristics

  * Supports Double-Byte Characters: Yes
  * Combinable: Yes
  * Deterministic: No
  * Reversible: No
  * Uniqueness: No

Examples

  * Use the regular expression `@abc\.com` to search for email addresses containing `@abc.com` and replace `@abc.com` with `@example.com`
  * Use the regular expression `[A-Z]+@[A-Z]+\.[A-Z]{2,4}` to mask email addresses by replacing with `john.doe@abcd.com`
  * Use the regular expression `[0-9]{3}[ -][0-9]{2}[ -][0-9]{4}` to match Social Security numbers and replace with `***-**-****`
  * Use the regular expression `<SALARY>[0-9]{2,6}</SALARY>` to zero out salary information by replacing with `<SALARY>0</SALARY>`
