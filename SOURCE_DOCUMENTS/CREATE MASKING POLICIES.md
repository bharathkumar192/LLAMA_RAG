
## Create Masking Policies

You can create a masking policy from a sensitive data model or create an empty
masking policy for a target database and add columns later.

### About Creating Masking Policies

You can create masking policies from the Masking Policies page in Oracle Data
Safe. You have a two options when creating a masking policy:

  * Create a masking policy starting with a sensitive data model. To use this option, you need to have access to a pre-built sensitive data model. Oracle Data Safe lists all the sensitive columns from the sensitive data model and automatically associates them with default masking format. You can then modify and edit the selections as needed. 
  * Create an empty masking policy and associate it with a target database. Later, you add columns to the masking policy and associate masking formats with them.

### Create a Masking Policy Starting From a Sensitive Data Model

  1. Under Security Center, click Data Masking. 
  2. Under Related Resources, click Masking Policies. The Masking Policies page is displayed. 
  3. Click Create Masking Policy. The Create Masking Policy window is displayed. 
  4. Enter a name for your masking policy. 
  5. Select a compartment in which to store your masking policy. 
  6. (Optional) Enter a brief description of your masking policy. 
  7. Leave the Using a sensitive data model tile selected. 
  8. Select a sensitive data model. If needed, click Change Compartment, and browse to and select a different compartment. 
  9. (Optional) To upload pre-masking and post-masking scripts, do the following: 
    1. Expand Upload Scripts. 
    2. In the Upload Pre-Masking Script area, drop your SQL file. Or, click the select one link, browse to and select your SQL file, and click Open. 
    3. In the Upload Post-Masking Script area, drop your SQL file. Or, click the select one link, browse to and select your SQL file, and click Open. 
  10. (Optional) To customize the processing of the masking job, do the following: 
    1. Expand Masking Options
    2. Disable or enable redo log generation during masking. This is disabled by default. Redo log generation allows you to use a flashback database to retrieve the original unmasked data after it has been masked. 
    3. Specify the value for parallel execution: 
       * NONE \- No parallelism is used when data masking process is running. 
       * DEFAULT \- The default value is the optimum number of CPUs to be used in parallel. This is calculated by the Oracle Database. 
       * DEGREE OF PARALLELISM \- Allows you to input an integer to set the number of CPUs to be used in parallel. Refer to the [Oracle Database parallel execution framework](/pls/topic/lookup?ctx=en/database/oracle/oracle-database/21&id=VLDBG-GUID-3E2AE088-2505-465E-A8B2-AC38813EA355) when choosing an integer value. 

Note:

The degree of parallelism is limited by the number of CPUs you have available.
If the integer entered in DEGREE OF PARALLELISM exceeds the number of
available CPUs, it will default to the maximum CPUs available when processing.

    4. Specify how you would like invalid objects to recompile after data masking: 
       * NONE \- Invalid objects do not recompile. 
       * SERIAL\- Invalid objects recompile serially, only when the previous objects has finished compiling. 
       * PARALLEL \- Invalid objects recompile using the same value for parallelism as specified above. 

Note:

If a value for parallelism was not specified, the value used will be the
optimized value calculated by the Oracle Database.

    5. Enable or disable dropping temporary tables created during data masking after masking is completed. This is enabled by default. Data Masking creates temporary tables that map the original sensitive data values to the mask values. Preserve these table to track how masking changed your data. 

Note:

Disabling dropping the temporary tables compromises security. These tables
must be dropped before the database is available for unprivileged users.

    6. Enable or disable refreshing the statistics gathered on masked database tables after masking. This is enabled by default.
  11. (Optional) To create tags, click Show Advanced Options and configure tags for your masking policy. 
  12. Click Create Masking Policy. 

Note:

It's important that you wait for the masking policy to be created before
closing the window so that all sensitive columns from the sensitive data model
are successfully added to the masking policy. When the masking policy is fully
created, the Masking Policy Details page is displayed and the status is set to
ACTIVE.

  13. Review your masking policy. 
     * The Masking Policy Information tab shows you the name and OCID of your masking policy, the work request information, the compartment in which the masking policy is stored, the target database with which the masking policy is associated, the name of the sensitive data model, and when the masking policy was created and last updated. 
     * The Masking Columns section shows you the list of sensitive columns, their associated default masking formats, and if they have associated child columns. 

### Create an Empty Masking Policy and Associate it With a Target Database

  1. Under Security Center, click Data Masking. 
  2. Under Related Resources, click Masking Policies. The Masking Policies page is displayed. 
  3. Click Create Masking Policy. The Create Masking Policy window is displayed. 
  4. Enter a name for your masking policy. 
  5. Select a compartment in which to store your masking policy. 
  6. (Optional) Enter a brief description of your masking policy. 
  7. Select the Using a target database tile. 
  8. Select a target database. If needed, click Change Compartment, and browse to and select a different compartment. 
  9. (Optional) To upload pre-masking and post-masking scripts, do the following: 
    1. Expand Upload Scripts. 
    2. In the Upload Pre-Masking Script area, drop your SQL file. Or, click the select one link, browse to and select your SQL file, and click Open. 
    3. In the Upload Post-Masking Script area, drop your SQL file. Or, click the select one link, browse to and select your SQL file, and click Open. 
  10. (Optional) To customize the processing of the masking job, do the following: 
    1. Expand Masking Options
    2. Disable or enable redo log generation during masking. This is disabled by default. Redo log generation allows you to use a flashback database to retrieve the original unmasked data after it has been masked. 
    3. Specify the value for parallel execution: 
       * NONE \- No parallelism is used when data masking process is running. 
       * DEFAULT \- The default value is the optimum number of CPUs to be used in parallel. This is calculated by the Oracle Database. 
       * DEGREE OF PARALLELISM \- Allows you to input an integer to set the number of CPUs to be used in parallel. Refer to the [Oracle Database parallel execution framework](/pls/topic/lookup?ctx=en/database/oracle/oracle-database/21&id=VLDBG-GUID-3E2AE088-2505-465E-A8B2-AC38813EA355) when choosing an integer value. 

Note:

The degree of parallelism is limited by the number of CPUs you have available.
If the integer entered in DEGREE OF PARALLELISM exceeds the number of
available CPUs, it will default to the maximum CPUs available when processing.

    4. Specify how you would like invalid objects to recompile after data masking: 
       * NONE \- Invalid objects do not recompile. 
       * SERIAL\- Invalid objects recompile serially, only when the previous objects has finished compiling. 
       * PARALLEL \- Invalid objects recompile using the same value for parallelism as specified above. 

Note:

If a value for parallelism was not specified, the value used will be the
optimized value calculated by the Oracle Database.

    5. Enable or disable dropping temporary tables created during data masking after masking is completed. This is enabled by default. Data Masking creates temporary tables that map the original sensitive data values to the mask values. Preserve these table to track how masking changed your data. 

Note:

Disabling dropping the temporary tables compromises security. These tables
must be dropped before the database is available for unprivileged users.

    6. Enable or disable refreshing the statistics gathered on masked database tables after masking. This is enabled by default.
  11. (Optional) To create tags, click Show Advanced Options and configure tags for your masking policy. 
  12. Click Create Masking Policy. The Masking Policy Details page is displayed. When the masking policy is successfully created, the status is set to ACTIVE. 
  13. Review your empty masking policy. 
     * The Masking Policy Information tab shows you the name and OCID of your masking policy, the work request information, the compartment in which the masking policy is stored, the target database with which the masking policy is associated, and when the masking policy was created and last updated. 
     * The Masking Columns section is empty. You can add and remove columns as needed. 
