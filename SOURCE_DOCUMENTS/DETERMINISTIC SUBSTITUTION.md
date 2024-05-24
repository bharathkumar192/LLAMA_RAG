  4. Deterministic Substitution

## Deterministic Substitution

Purpose

The Deterministic Substitution masking format uses the specified substitution
column as the source of masked values. It performs hash-based substitution to
replace the original data in a column with values from the substitution
column.

Inputs

  * Schema Name: The name of the schema containing the substitution column 
  * Table Name: The name of the table containing the substitution column 
  * Column name: The name of the substitution column containing the data that should be used for masking. The data types of the specified substitution column and column being masked must be the same. The substitution column must be present and accessible on the target database before masking. You can also use a pre-masking script to create this column. 
  * Seed value: Deterministic Substitution uses a seed value to perform hash-based substitution. Provide the seed value at the time of submitting a data masking job. It can be any string containing alphanumeric characters. To perform deterministic masking, you need to use the same seed value across multiple masking runs. 

Supported Data Types

  * Character
  * Numeric
  * Date

Characteristics

  * Supports Double-Byte Characters: Yes
  * Combinable: No
  * Deterministic: Yes, as long as the values in the substitution column do not change and you provide the same seed value
  * Reversible: No
  * Uniqueness: No

Example

Suppose you discover a sensitive column named `EMP_ID` that contains employee
IDs. Let's assume that you have fake employee ID values stored in another
column named `SUB_EMP_ID`, which resides in the `SUB_EMPLOYEES` table in the
`SUB_HR` schema. When configuring the masking policy in the Data Masking
wizard, you choose the Deterministic Substitution masking format for the
`EMP_ID` column and provide the inputs: `SUB_HR`,` SUB_EMPLOYEES`, and
`SUB_EMP_ID`.

You also specify a seed value at job submission time. When the job runs, Data
Masking replaces the values in the `EMP_ID` column with the fake values from
the `SUB_EMP_ID` column. In the future, you can mask this column (or other
similar columns) using the same substitution column and seed value to ensure
that the employee IDs are masked the same way.
