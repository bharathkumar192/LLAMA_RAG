  4. Group Shuffle

## Group Shuffle

Purpose

The Group Shuffle masking format enables you to randomly reorder (shuffle)
column data within discrete units, or groups, where there is a relationship
among the members of each group.

Inputs

  * Grouping Columns (Optional): One or more reference columns that should be used to group the values in the column to be masked. The grouping columns and the column to be masked must belong to the same table. 

Supported Data Types

  * Character
  * Numeric
  * Date

Characteristics

  * Supports Double-Byte Characters: Yes
  * Combinable: No
  * Deterministic: No
  * Reversible: No
  * Uniqueness: Yes, this masking format ensures uniqueness for columns that have unique constraints

Example

Suppose you have two groups of employees: managers (M) and workers (W). You
want to shuffle all the salaries, but you do not want the salaries of the
managers getting mixed into the salaries of the workers. You can use the Group
Shuffle masking format to shuffle the `SALARY` column within each group, which
is derived from the unique values in the `JOB_CATEGORY` column.

The following table illustrates a group shuffle on the `SALARY` column, where
the `JOB_CATEGORY` column is the grouping column. The rows with `JOB_CATEGORY`
= M belong to one group and the `SALARY` values belonging to this group are
shuffled within the group. Similarly, the rows with `JOB_CATEGORY` = W belong
to another group and the `SALARY` values belonging to this group are shuffled
within the group.

EMPLOYEE | JOB_CATEGORY | SALARY | SHUFFLED_SALARY  
---|---|---|---  
Alice | M | 90 | 88  
Bill | M | 88 | 90  
Carol | W | 72 | 70  
Denise | W | 57 | 45  
Eddie | W | 70 | 57  
Frank | W | 45 | 72
