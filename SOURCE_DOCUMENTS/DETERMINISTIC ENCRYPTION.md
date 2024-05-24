  4. Deterministic Encryption

## Deterministic Encryption

Purpose

The Deterministic Encryption masking format encrypts column data using a
cryptographic key and Advanced Encryption Standard (AES 128). The format of
the column data after encryption is similar to that of the original values.
For example, if you mask nine-digit numbers, the encrypted values also have
nine digits.

Deterministic Encryption is a deterministic and reversible masking format. It
is helpful when businesses need to mask and send their data to a third party
for analysis, reporting, or any other business processing purpose. After the
processed data is received from the third party, the original data can be
recovered (decrypted) using the same seed value that was used to encrypt the
data.

Note:

Deterministic Encryption is not supported for Oracle Database 11.2.0.4.

Inputs

  * Regular Expression: Provide a regular expression to mask a character or numeric column. 

For data with characters in the ASCII character set, providing a regular
expression is optional. However, you need to provide a regular expression if
the data contains multi-byte characters. If not provided, an error is returned
when a multi-byte character is found.

In the case of ASCII characters, if a regular expression is not provided,
Deterministic Encryption can encrypt variable-length column values while
preserving their original format.

If a regular expression is provided, the column values in all the rows must
match the regular expression. Deterministic Encryption supports a subset of
the regular expression language. It supports encryption of fixed-length
strings, and does not support * or + syntax of regular expressions. The
encrypted values also match the regular expression, which helps to ensure that
the original format is preserved. If an original value does not match the
regular expression, Deterministic Encryption might not produce a one-to-one
mapping. All non-confirming values are mapped to a single encrypted value,
thereby producing a many-to-one mapping.

Deterministic Encryption can encrypt column values with up to 27 characters.
This limit excludes special characters. Also, the limit can be lower for
multi-byte characters.

WARNING:

If you choose to encrypt without using a regular expression, the column values
exceeding the length restriction still get masked, but you might not be able
to decrypt them back properly. If a regular expression is provided, size
estimation is done using the regular expression and an error is returned if
the length restriction is exceeded.

  * Seed Value: Deterministic Encryption uses a seed value to generate a cryptographic key for encryption and decryption. Provide the seed value at the time of submitting the data masking job. It can be any string containing alphanumeric characters. 
  * Decrypt Option: If your masking policy has a sensitive column using the Deterministic Encryption masking format, you are shown the decrypt option while submitting the data masking job. Choosing this option, you can decrypt the encrypted column values. 
  * For Date types: To mask a date type column, provide a start and end date. You can use the calendar widget to select the dates. The start date must be less than or equal to the end date. 

The column values in all the rows must be within the specified date range. The
encrypted values are also within the specified range. Therefore, to ensure
uniqueness, the total number of dates in the range must be greater than or
equal to the number of distinct original values in the column. If an original
value is not in the specified date range, Deterministic Encryption might not
produce a one-to-one mapping. All non-confirming values are mapped to a single
encrypted value, thereby producing a many-to-one mapping.

Supported Data Types

  * Character
  * Numeric
  * Date

Characteristics

  * Supports Double-Byte Characters: Yes
  * Combinable: No
  * Deterministic: Yes
  * Reversible: Yes
  * Uniqueness: Yes. Refer to the Inputs section to see specific conditions.

Example

Suppose you want to encrypt US Social Security numbers, such as 333-93-4245.
You can simply choose Deterministic Encryption without providing a regular
expression. It automatically encrypts the numbers while preserving the format.

If you want to restrict the characters in encrypted values, you can provide a
regular expression. For example, you can use the regular expression
`[1-8][0-9]{2}-[0-9]{2}-[0-9]{4}` if the first digit in your numbers is
between 1 and 8, and you want to ensure the same in the encrypted values.

See Also:

[Regular Expressions](regular-
expressions.html#GUID-37A467A1-983E-412F-9C06-5E96A2D17991 "You can use
regular expressions to describe a set of strings based on common
characteristics shared by each string in the set.") for help on writing
regular expressions.
