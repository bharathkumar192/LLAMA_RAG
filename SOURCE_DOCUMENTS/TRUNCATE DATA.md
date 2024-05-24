  4. Truncate Data

## Truncate Data

Purpose

The Truncate Data masking format drops all the rows in a table. If one of the
columns in a table is masked using Truncate Data, the entire table is
truncated, so no other masking format can be used for any of the other columns
in that table. If a table is being truncated, it cannot be referred to by a
foreign key constraint or a dependent column.

Inputs

  * No inputs are required.

Supported Data Types

  * Character
  * Numeric
  * Date

Characteristics

  * Supports Double-Byte Characters: Yes
  * Combinable: No
  * Deterministic: Does not apply
  * Reversible: Does not apply
  * Uniqueness: Does not apply

Example

Suppose that you want to mask ten tables in a database schema. In one of the
tables, all the columns contain highly sensitive data, and therefore, you do
not want to share this table. You can use the Truncate Data masking format to
drop all the rows in this table.
