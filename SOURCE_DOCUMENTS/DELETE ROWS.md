  4. Delete Rows

## Delete Rows

Purpose

The Delete Rows masking format deletes the rows that meet a user-specified
condition. It is useful in conditional masking when you want to delete a
subset of values in a column and mask the remaining values using some other
masking formats. You should be careful while using this masking format. If no
condition is specified, all rows in a table are deleted. If a column is being
masked using Delete Rows, there must not be a foreign key constraint or
dependent column referring to the table.

See Also:

[Example 1: Protecting Sensitive Identifiers Across Diverse Geographic
Regions](conditional-masking.html#GUID-3C52EE0C-9FBE-498D-AC8F-B59C259A93C0
"Learn how conditional masking can be used to mask unique personal identifiers
based on country.")

Inputs

  * No inputs are required.

Supported Data Types

  * Character
  * Numeric
  * Date

Characteristics

  * Supports Double-Byte Characters: Does not apply
  * Combinable: No
  * Deterministic: Does not apply
  * Reversible: No
  * Uniqueness: Does not apply

Example

Assume that a table has `EMPLOYEE_ID` and `SALARY` columns, and you want to
delete the salary data for a subset of employee IDs. You can specify a
condition on the `SALARY` column using `EMPLOYEE_ID` to delete rows matching
the condition. You can use some other masking formats to mask the remaining
salary values.

The logic to mask `SALARY` might look like the following:

    
    
    EMPLOYEE_ID < 100   
          DELETE ROWS
    EMPLOYEE_ID < 200
          RANDOM NUMBER [Start Value:30000 End Value:500000]
    DEFAULT
          PRESERVE ORIGINAL DATA
