  4. Random Substitution

## Random Substitution

Purpose

The Random Substitution masking format enables you to mask values in a column
using data from a substitution column. The values in the user-specified column
are randomly ordered before mapping them to the original column values.

Inputs

  * Schema Name: The name of the schema containing the substitution column 
  * Table Name: The name of the table containing the substitution column 
  * Column Name: The name of the substitution column containing the data that should be used for masking. The data types of the specified substitution column and column to be matched must be the same. 

Supported Data Types

  * Character
  * Numeric
  * Date

Characteristics

  * Supports Double-Byte Characters: Yes
  * Combinable: No
  * Deterministic: No, because the Random Substitution masking format randomly orders the mask values in the substitution column before replacing the sensitive data (unlike the Deterministic Substitution masking format)
  * Reversible: No
  * Uniqueness: Yes. The number of distinct values in the substitution column must be greater than or equal to the number of values in the column to be masked.

Example

Suppose you discover a sensitive column named `EMP_ID` that contains employee
IDs. Let's assume that you have fake employee ID values stored in another
column named `SUB_EMP_ID`, which resides in the` SUB_EMPLOYEES` table in the
`SUB_HR` schema (as shown in the following table).

SUB_EMP_ID  
---  
101  
102  
103  
104  
105  
106  
107  
  
When configuring the masking policy in the Data Masking wizard, you can choose
the Random Substitution masking format for the `EMP_ID` column. Provide the
following inputs: `SUB_HR`, `SUB_EMPLOYEES`, and `SUB_EMP_ID`. When the job
runs, Data Masking randomly orders the fake values in the `SUB_EMP_ID` column
and uses them to replace the values in the `EMP_ID` column.

The following table compares the values in the original column (`EMP_ID`) to
the values after the first masking job (`MASK1`) and second masking job
(`MASK2`). Notice that the masked values change each time the masking job
runs.

EMP_ID | MASK1 | MASK2  
---|---|---  
412 | 101 | 104  
185 | 107 | 105  
102 | 105 | 102  
322 | 102 | 101  
692 | 103 | 106
