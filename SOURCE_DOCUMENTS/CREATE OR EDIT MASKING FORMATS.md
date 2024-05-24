
## Create or Edit Masking Formats

You can create your own masking formats by using the basic masking formats in
Oracle Data Safe as your starting point.

### About Creating User-Defined Masking Formats

When creating user-defined masking formats, you select basic masking formats
and then customize them to meet your specific requirements. The results of the
masking formats are concatenated together. You can also choose to start with
an existing masking format, edit it, and then save it under a different name.

You cannot apply conditions when creating a masking format. You can create
conditions when you modify a masking format from within a masking policy.

Suppose you want to mask a national identifier that has ten digits. You can
create a new masking format, say My National Identifier, using the Random
Number masking format. Random Number takes two inputs: Start Number and End
Number. You can provide 1000000000 as the Start Number and 9999999999 as the
End Number, and then save your masking format. In the future, to mask a column
containing that national identifier, you can simply choose the My National
Identifier masking format. Input is not required. If you have a sensitive type
to discover your national identifier, you can also set My National Identifier
as the default masking format for that sensitive type. This way, whenever you
discover columns using this sensitive type, Data Masking selects the mapped
masking format by default.

### Supported Data Types for User-Defined Masking Formats

Before creating a masking format for a sensitive column, first determine the
column's data type. The data type dictates which basic masking formats you can
use.

Character Data Types

The following character types can use Delete Rows, Deterministic Encryption,
Deterministic Substitution, Fixed Number, Fixed String, Group Shuffle, Null
Value, Post Processing Function, Preserve Original Data, Random Decimal
Number, Random Digits, Random List, Random Number, Random String, Random
Substitution, Regular Expression, Shuffle, SQL Expression, Substring, Truncate
Data, and User Defined Function masking formats:

  * `CHAR`
  * `NCHAR`
  * `VARCHAR2`
  * `NVARCHAR2`

Numeric Data Types

The following numeric types can use Delete Rows, Deterministic Encryption,
Deterministic Substitution, Fixed Number, Group Shuffle, Null Value, Post
Processing Function, Preserve Original Data, Random Decimal Number, Random
Digits, Random List, Random Number, Random Substitution, Regular Expression,
Shuffle, SQL Expression, Truncate Data, and User Defined Function masking
formats:

  * `NUMBER`
  * `FLOAT`
  * `RAW`
  * `BINARY_FLOAT`
  * `BINARY_DOUBLE`

Date Data Types

The following date types can use Delete Rows, Deterministic Encryption,
Deterministic Substitution, Group Shuffle, Null Value, Post Processing
Function, Preserve Original Data, Random Date, Random List, Random
Substitution, Shuffle, SQL Expression, Truncate Data, and User Defined
Function masking formats:

  * `DATE`
  * `TIMESTAMP`

Large Object (LOB) Data Types

The following LOB data types can use Fixed Number, Fixed String, Null Value,
Regular Expression, and SQL Expression masking formats:

  * `BLOB`
  * `CLOB`
  * `NCLOB`

Unsupported Objects

Oracle Data Safe does not support masking for the following:

  * External tables
  * Clustered tables
  * Queue tables
  * Long columns
  * XML-type columns
  * Virtual columns
  * `ROWID` columns 
  * JSON columns
  * Graph tables

### Create a User-Defined Masking Format

  1. Under Security Center, click Data Masking.
  2. Under Related Resources, click Masking Formats.

The Masking Formats page is displayed.

  3. Click Create Masking Format. 

The Create Masking Format window is displayed.

  4. Enter a name for your masking format.
  5. Select a compartment in which to store your masking format.
  6. (Optional) Enter a description for your masking format.
  7. (Optional) If you want to create a masking format based on an existing masking format, do the following:
    1. (Optional) If needed, click Change Compartment, and browse to and select the correct compartment. Oracle predefined masking formats are listed in all compartments. User-defined masking formats may reside in other compartments.
    2. From the Create Like Masking Format drop-down list. Select an Oracle predefined masking format or a user-defined masking format. The masking format fields are automatically populated.
  8. Configure the masking format. 
    1. From the Masking Format Entry drop-down list, select a basic masking format and configure its parameters. Or, if you previously selected a masking format to copy, edit the existing parameters as needed.
    2. To add another masking format, click Add Format Entry and configure its parameters. If you enter more than one masking format, the masking formats will be concatenated.
    3. To delete a masking format, click the X button next to it.
  9. Click Create Masking Format.

Your new masking format is now displayed on the Masking Formats page. You can
select your masking format whenever you create a data masking job.

### Edit a User-Defined Masking Format

  1. Under Security Center, click Data Masking.
  2. Under Related Resources, click Masking Formats.

The Masking Formats page is displayed.

  3. Search for and click the name of your masking format. 

The Masking Format Details page is displayed.

  4. Click Edit. 

The Edit Masking Format window is displayed.

  5. (Optional) Modify the name and/or description of the masking format.
  6. (Optional) Modify existing masking format entries.
  7. (Optional) To add another masking format entry, click \+ Another Format Entry, select a basic masking format, and then configure its values.
  8. (Optional) To delete a masking format, click the X button next to the right of the masking format entry. 
  9. Click Save.

The masking format is immediately updated.
