
## Edit Masking Policies

After you generate the initial masking policy for a target database, you most
likely will need to edit it. For example, you might need to address sensitive
columns that do not have an associated masking format, change masking formats,
apply conditions to some masking formats, mask related columns together as a
group (group masking), or add or remove columns from the masking policy.

### Fix Columns that Need Attention

If you have one or more columns in your masking policy that are not
automatically associated with a masking format, you need to address these
columns. This may happen in the following scenarios:

  * The sensitive column was discovered by a user-defined sensitive type, but the sensitive type does not have a default masking format assigned to it.
  * Data Safe tried to associate a masking format, but it was not possible. This could've happened in the following scenarios: 
    * The column contains a value that is incompatible with the column format
    * The assigned masking format generates data that exceeds the column size
    * The masking format does not guarantee sufficient number of distinct values which could lead to loss of data integrity

You can quickly find the list of columns needing your attention on the Masking
Columns Needing Attention page.

  1. Under Security Center, click Data Masking.
  2. Under Related Resources, click Masking Policies.
  3. Click the name of your masking policy to view its details.
  4. Under Resources, click Masking Columns Needing Attention.
  5. Locate the rows that have an exclamation mark next to the masking policy. Hover your mouse over the exclamation mark to learn about the issue.
  6. Select a different masking format for the rows or edit the existing masking formats to resolve the issues. When a masking format is successfully updated, a message states Masking Format Updated Successfully.

### Change or Edit the Masking Format for a Sensitive Column

By default, Oracle Data Safe associates a masking format with each sensitive
column in a masking policy. If needed, you can select a different masking
format or edit the default masking format.

  1. Under Security Center, click Data Masking.
  2. Under Related Resources, click Masking Policies.
  3. Click the name of your masking policy to view its details.
  4. Scroll down to the Masking Columns section where all the columns are listed with their associated masking formats.
  5. Locate the row for the column for which you want to change the masking format.
  6. Perform one of the following actions to change the masking format: 
     * From the Masking Format drop-down list, select a different predefined masking format. The Edit Masking Format page is displayed with the new masking format configuration. Edit the values as needed, and then click Continue. 
     * Click the pencil button next to the masking format to open the Edit Masking Format page. Select a different masking format, configure the parameters, and then click Continue. 
  7. (Optional) Repeat step 6 to change the masking formats of other columns.
  8. Verify that the highlighted rows are the ones that contain the masking format updates that you want. Note that your updates are not yet saved. If you navigate away from this page without saving, your changes will be lost.
  9. To save all masking format updates at one time, click Save Masking Formats.

### Add Conditions to a Masking Format

See the steps and examples below to implement conditional masking formats in
your masking policies.

  1. Under Security center, click Data masking. 
  2. Under Related Resources, click Masking Policies. 
  3. Click the name of a masking policy.
  4. Locate the row for the column for which you want to add conditional masking in the Masking Columns section.
  5. Click the pencil icon to edit the masking format.
  6. Enter the desired condition in the Condition field, removing the default condition 1=1. 
  7. Select the Masking Format Entry. 
  8. Fill out any additional fields related to the selected masking format.
  9. To add another condition, click Another Masking Format and repeat steps six through eight. 
  10. Click Continue once you have set all your conditional masking formats. 

Example 7-1 Personal Identifiers Based on Country

In this example the goal is to create a masking format where unique personal
identifiers are masked differently based on the country that the unique
personal identifiers apply to. American (USA) identifiers can be masked using
the Social Security Number masking format, and British (UK) identifiers can be
masked using the National Insurance Number masking format. The below
screenshot shows the conditions that could be set to implement a similar
conditional masking format.

  

![The following conditions have been applied to this sensitive data:
Condition: COUNTRY = USA, Masking Format Entry: US Social Security Number,
Condition: COUNTRY = UK, Masking Format Entry: UK National Insurance Number
\(Space Separated\).](img/conditional_masking_country.png)

  

Example 7-2 Fixed Salary Based on Job Category

In this example the goal is to create a masking format where salaries are
masked by being set to different values based on the employees job category.
The below screenshot shows the conditions that could be set to implement a
similar conditional masking format.

  

![The column to be masked is the SALARY column in the EMPLOYEES table in the
TEST schema.There was no original masking format, but it has now been edited
according to the following conditions: Condition: JOB_CATEGORY = MANAGER,
Masking Format Entry: Random Number, Start Value: 100000, End Value: 150000.
Condition: JOB_CATEGORY = WORKER, Masking Format Entry: Fixed Number, Fixed
Number: 25000. Condition: 1 = 1, Masking Format Entry: Preserve Original
Data.](img/conditional_masking_employee.png)

  

Example 7-3 Fixed Salary Based on Salary Amount

In this example the goal is to create a masking format where salaries are
masked by being set to fixed values based on the salary amount. The below
screenshot shows the conditions that could be set to implement a similar
conditional masking format.

  

![The column to be masked is the SALARY column in the EMPLOYEES table in the
HCM1 schema. The original masking format was Income, but it has now been
edited according to the following conditions: Condition: SALARY < 3000,
Masking Format Entry: Fixed Number, Fixed Number: 3000. Condition: SALARY
between 3000 and 10000, Masking Format Entry: Fixed Number, Fixed Number:
10000. Condition: SALARY > 10000, Masking Format Entry: Fixed Number, Fixed
Number: 50000.](img/conditional_masking_fixed_income.png)

  

**Related Topics**

  * [Conditional Masking](conditional-masking.html#GUID-9C1036BE-FA0C-440D-B2AC-1EE29B2745E1 "Conditional masking allows you to set multiple logical conditions that alter the masking format of a masking column. Conditional masking can only be done when editing an existing masking format of a masking column in a masking policy.")

### Mask Related Columns Together as a Group (Group Masking)

You can mask related columns together as a group, ensuring that the masked
data across the related columns retain the same relationship.

  1. Open a masking policy and scroll down to the Masking Columns section.
  2. Select Group Masking from the drop-down list for one of the columns that is part of the group.

The Edit Masking Format page displayed. By default, the Group Name field,
Masking Format Entry drop-down list, and the column are displayed. You can add
and remove columns from the group.

  3. In the Group Name field, enter a new group name if this is the beginning of a group masking configuration. Or, select an existing group name if you want to add the column to an existing group masking configuration.
  4. From the Masking Format Entry drop-down list, select the masking format that you want to apply to the columns in the group. You can select Shuffle, Deterministic Substitution, Random Substitution, or User Defined Function.
  5. If you selected Shuffle as the masking format in step 4, you can optionally enter "group by" column names in the Group Columns box.
  6. If you selected Deterministic Substitution as the masking format in step 4, enter the name of the substitution schema and table. Also, for each column listed, enter the name of the substitution column. 
  7. If you selected Random Substitution as the masking format in step 4, enter the name of the substitution schema and table. Also, for each column listed, enter the name of the substitution column. 
  8. If you selected User Defined Function as the masking format in step 4, enter the name of the schema and function for each column listed. Optionally, you can also enter a package name.
  9. To add another column to the group, click Add Column. 

You can repeat this step until all columns in the table are listed, after
which point the Add Column button becomes unavailable. Make sure that the
column you initially selected to configure in step 2 is listed.

  10. To remove a column from the group, select the check box for the column, and then click Remove Column.
  11. Click Save.

### Add Columns to a Masking Policy

  1. Under Security Center, click Data Masking.
  2. Under Related Resources, click Masking Policies.
  3. Click the name of your masking policy to view its details.
  4. Scroll down to the Masking Columns section and click Add Columns.

The Add Columns window is displayed.

  5. (Optional) If the schemas on the target database have been updated since the stated time and date, click Refresh Database Schemas.
  6. Select the sensitive type that best describes the columns that you want to add to your masking policy.
  7. Find columns by entering or selecting one or more of the following items, and then click Search.
     * Schema name
     * Table name
     * Column name

A list of columns that match your selection criteria are displayed.

  8. (Optional) Change the sensitive type of a column by selecting a new sensitive type from the Sensitive Type column.
  9. Select the columns that you want to add to your masking policy, and then click Add Columns. To select all the columns, select the check box next to the Schema column heading.

The columns are added to the masking policy.

### Add Previously Removed Columns to a Masking Policy

You can view the list of columns that were removed from a masking policy in
the past and add them back to the masking policy if needed.

  1. Under Security Center, click Data Masking. 
  2. Under Related Resources, click Masking Policies. 

A list of masking policies to which you have access is displayed.

  3. Click the name of the masking policy for which you want to view or add previously removed columns.
  4. Scroll down to the Masking Columns list and click View/Add Previously Removed Columns. 

The Add Previously Removed Columns panel shows the schema, table, column, and
data type for each previously removed column.

  5. To add all previously removed columns back to the masking policy, select All columns.
  6. To add specific columns back to the masking policy, select Select specific columns, and then select individual columns from the list.
  7. Click Add Columns to Masking Policy.

### Remove Columns from a Masking Policy

You can remove columns from your masking policy that you don't want to mask on
the target database. Note that the underlying sensitive data model is not
affected.

  1. Under Security Center, click Data Masking.
  2. Under Related Resources, click Masking Policies.
  3. Click the name of your masking policy to view its details.
  4. Scroll down to the Masking Columns section where all the columns are listed with their associated masking formats.
  5. To remove a singular column, click the â® symbol to the right of Masking Column to be removed in the Masking Columns list.
    1. Click the Remove option.
    2. Click Remove Column in the dialog box to confirm the removal of the column.
  6. To remove multiple columns, click Remove Columns above the Masking Columns list. The Remove Columns window is displayed.
    1. (Optional) Select a sensitive type that best describes the columns that you want to remove.
    2. Enter or select one or more of the following items, and then click Search.

       * Schema name
       * Table name
       * Column name

A list of sensitive columns that match your selection criteria are displayed.

    3. Select the columns that you want to remove from your masking policy, and then click Remove Columns.

To select all the columns, select the check box next to the Schema column
heading. The columns are removed from the masking policy and the masking
policy is automatically saved.

### Update Tags, Masking Scripts, and Masking Options for a Masking Policy

  1. Under Security Center, click Data Masking. 
  2. Under Related Resources, click Masking Policies
  3. Click the name of your masking policy to view its details 
  4. From the More Actions menu select either Add Tags, Update Pre/Post Masking Scripts, or Update Masking Options. 
  5. (Optional) If you would like to add or update tags for your masking policy, configure them in the pop-up after selecting Add Tags. Select the Tag Namespace, Tag Key, and Tag Value from the drop-down lists. 
  6. (Optional) To upload pre-masking and post-masking scripts, do the following after selecting Update Pre/Post Masking Scripts: 
    1. In the Upload Pre-Masking Script area, drop your SQL file. Or, click the select one link, browse to and select your SQL file, and click Open. 
    2. In the Upload Post-Masking Script area, drop your SQL file. Or, click the select one link, browse to and select your SQL file, and click Open. 
  7. (Optional) To customize the execution of the Masking Policy, do the following after selecting Update Masking Options: 
    1. Disable or enable redo log generation during masking. This is disabled by default. Redo log generation allows you to use a flashback database to retrieve the original unmasked data after it has been masked. 
    2. Specify the value for parallel execution: 
       * NONE \- No parallelism is used when data masking process is running. 
       * DEFAULT \- The default value is the optimum number of CPUs to be used in parallel. This is calculated by the Oracle Database. 
       * DEGREE OF PARALLELISM \- Allows you to input an integer to set the number of CPUs to be used in parallel. Refer to the [Oracle Database parallel execution framework](/pls/topic/lookup?ctx=en/database/oracle/oracle-database/21&id=VLDBG-GUID-3E2AE088-2505-465E-A8B2-AC38813EA355) when choosing an integer value. 

Note:

The degree of parallelism is limited by the number of CPUs you have available.
If the integer entered in DEGREE OF PARALLELISM exceeds the number of
available CPUs, it will default to the maximum CPUs available when processing.

    3. Specify how you would like invalid objects to recompile after data masking: 
       * NONE \- Invalid objects do not recompile. 
       * SERIAL\- Invalid objects recompile serially, only when the previous objects has finished compiling. 
       * PARALLEL \- Invalid objects recompile using the same value for parallelism as specified above. 

Note:

If a value for parallelism was not specified, the value used will be the
optimized value calculated by the Oracle Database.

    4. Enable or disable dropping temporary tables created during data masking after masking is completed. This is enabled by default. Data Masking creates temporary tables that map the original sensitive data values to the mask values. Preserve these table to track how masking changed your data. 

Note:

Disabling dropping the temporary tables compromises security. These tables
must be dropped before the database is available for unprivileged users.

    5. Enable or disable refreshing the statistics gathered on masked database tables after masking. This is enabled by default.

### Compare a Masking Policy to a Sensitive Data Model

When a sensitive data model is modified, a comparison to an associated masking
policy can be initiated. The comparison identifies any differences between the
sensitive data model and masking policy and allows you to select changes that
will sync with the masking policy.

To run a comparison between a masking policy and it's associated sensitive
data model:

  1. Under Security Center, click Data Masking. 
  2. Under Related Resources, click Masking Policies. 
  3. Select a masking policy from the list.
  4. Under Resources, click Compare with Sensitive Data Model. 

This is only available if the masking policy is associated with a sensitive
data model.

  5. Click the Compare with Sensitive Data Model button. 
  6. Click Submit. 
  7. Once the comparison is complete, review any changes and select any changes that you like to sync under Planned Actions. 
  8. If you did not select all the changes in the previous step, click the Save Changes for Planned Actions button. 
    1. Click Save. 
  9. Click the Synchronize Masking Policy button. 
  10. Click the Synchronize Masking Policy button in the confirmation dialog. 

Once complete the masking policy will be updated with all changes that were
selected.
