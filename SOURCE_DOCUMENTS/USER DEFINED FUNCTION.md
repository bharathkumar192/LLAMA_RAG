  4. User Defined Function

## User Defined Function

Purpose

The User Defined Function masking format lets you define your own logic to
mask column data. The return value of the user-defined function is used to
replace the original values. The user-defined function is a PL/SQL function
that can be invoked in a `SELECT` statement.

Inputs

  * Package Name: The name of the database package 
  * Function Name: The name of the database function 

The database function has a fixed signature:

    
    
    function udf_func (rowid varchar2, column_name varchar2, original_value varchar2) return varchar2;

where:

  * `rowid` is the row identifier of the row containing the value to be masked. 
  * `column_name` is the name of the column to be masked. 
  * `original_value` is the column value to be masked. 

Supported Data Types

  * Character
  * Numeric
  * Date

Characteristics

  * Supports Double-Byte Characters: Yes
  * Combinable: Yes 
  * Deterministic: Yes, depending on the function defined
  * Reversible: No
  * Uniqueness: Yes, depending on the function defined

Example

Suppose you create a user-defined function to mask string values.

To create the user-defined function, you might use the following code to
randomize the string values. This example is simple, however you can write
more complex code to suit your business use case.

    
    
    CREATE OR REPLACE FUNCTION 
    change_value (rowid varchar2, column_name varchar2, mask_value varchar2) 
    RETURN varchar2
    IS
    BEGIN
        RETURN DBMS_RANDOM.STRING('A',8);
    END;
