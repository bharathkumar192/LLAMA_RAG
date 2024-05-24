  4. Post Processing Function

## Post Processing Function

Purpose

The Post Processing Function masking format is a special masking option that
enables you to use a custom function to further transform column values after
they have been masked using some other masking formats. It takes the
intermediate masked values as input and returns the final masked values. For
example, you can use it for adding checksums or special encodings to the
masked values. This masking option requires some level of coding skills.

Inputs

  * Package Name (Optional): The name of the database package 
  * Function Name: The name of the database function 

The database function has a fixed signature:

    
    
    function post_proc_func (rowid varchar2, column_name varchar2, mask_value varchar2) 
    return varchar2;

where:

  * `rowid` is the row identifier of the row containing the value to be masked. 
  * `column_name` is the name of the column to be masked. 
  * `mask_value` is the value to be masked. 

Supported Data Types

  * Character
  * Numeric
  * Date

Characteristics

  * Supports Double-Byte Characters: Yes
  * Combinable: Yes
  * Deterministic: Does not apply
  * Reversible: Does not apply
  * Uniqueness: Does not apply

Example

You can use Post Processing Function to add a comma or dollar sign to a value.
Suppose that you mask a `SALARY` column by using the Random Number masking
format. You can then apply the Post Processing Function masking format to the
masked values to add a currency symbol, such as `$`.

    
    
    RANDOM NUMBER [START:25000 END: 100000]
    POST PROCESSING FUNCTION salary_post_processing 	    
    

To create the `salary_post_processing` function, your code might look like the
following:

    
    
    CREATE OR REPLACE FUNCTION 
    salary_post_processing (rowid varchar2, column_name varchar2, mask_value varchar2) 
    RETURN varchar2
    IS
    BEGIN
        RETURN ('$' || mask_value);
    END;
