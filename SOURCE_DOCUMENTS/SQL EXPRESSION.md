  4. SQL Expression

## SQL Expression

Purpose

The SQL Expression masking format lets you use a SQL expression to mask column
data. Data Masking uses the specified SQL expression to generate values which
are used to replace the original data.

Inputs

  * SQL Expression: The SQL expression generates the masked values. It can consist of one or more values, operators, and SQL functions that evaluate to a value. It can also contain substitution columns (columns from the same table as the column to be masked). Specify the substitution columns within percent (%) symbols. Use SQL expressions with `dbms_lob` and other user-defined functions to mask columns of Large Object data type (LOBs include BLOB, CLOB, and NCLOB). 

Supported Data Types

  * Character
  * Numeric
  * Date
  * Large Object (LOB)

Characteristics

  * Supports Double-Byte Characters: Yes, the SQL expression provided should generate multi-byte characters
  * Combinable: No
  * Deterministic: Yes, depending on the SQL expression defined
  * Reversible: No
  * Uniqueness: Yes, but the uniqueness is not guaranteed and depends on the SQL expression defined. However, because `ORA_HASH` uses a 32-bit algorithm, and considering the birthday paradox or pigeonhole principle, there is a 0.5 probability of collision after 232-1 unique values. 

Examples

  * Generate random email addresses.
    
        dbms_random.string('u', 8) || '@example.com'

  * Generate email addresses using values from substitution columns, for example, `FIRST_NAME` and `LAST_NAME`.
    
        %FIRST_NAME% || '.' || %LAST_NAME% || '@example.com'

  * Empty a CLOB.
    
        dbms_lob.empty_clob()

  * Apply a custom masking function to a CLOB column, for example, `CLOB_COL`.
    
        custom_mask_clob(%CLOB_COL%)

  * Perform conditional masking. For example, the following expression masks `PERSON_FULL_NAME` with the first and last name if the party type is `PERSON`. Otherwise, it uses a random string to mask the data.
    
        (case when %PARTY_TYPE%='PERSON' then %PERSON_FIRST_NAME%|| ' ' ||%PERSON_LAST_NAME% else (select dbms_random.string('U', 10) from dual) end)

  * Perform substitution masking. For example, the following expression selects 1000 rows in the substitution table, `DATA_MASK.DATA_MASK_ADDR`. It masks `%ZIPCODE%` with the `MASK_ZIPCODE` column in the substitution table. The row selected depends on `ora_hash` and is deterministic in this case. Selection is random if `dbms_random` procedures are used.
    
        select MASK_ZIPCODE from DATA_MASK.DATA_MASK_ADDR where ADDR_SEQ = ora_hash( %ZIPCODE% , 1000, 1234)
