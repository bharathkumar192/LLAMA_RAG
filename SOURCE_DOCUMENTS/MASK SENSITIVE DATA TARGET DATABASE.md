
## Mask Sensitive Data on a Target Database

You can run a Data Masking job from the Data Masking page or the Masking
Policy Details page.

### Mask Sensitive Data from the Data Masking Page

Be sure that you are not trying to mask sensitive data on your production
database.

  1. Under Security Center, click Data Masking. 

The Data Masking page is displayed.

  2. Click Mask Sensitive Data. 

The Mask Sensitive Data window is displayed.

  3. Select a target database. If needed, click Change Compartment, and browse to and select a different compartment. 
  4. Select a masking policy for the selected target database. If needed, click Change Compartment, and browse to and select a different compartment. 
  5. (Optional) Select the tablespace.
  6. (Optional) To customize the processing of the masking job for the first time or to override the existing options associated with the selected masking policy, do the following: 
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
  7. Click Mask Data. 

A warning message states that you should not mask data on a production
database.

The work request page is displayed so that you can see the progress of the
masking job.

You can run only one data masking job at a time on a target database.

### Mask Sensitive Data from the Masking Policies Details Page

  1. Under Security Center, click Data Masking. 
  2. Under Related Resources, click Masking Policies. 
  3. Click the name of a masking policy to view its details. 

The Masking Policies Details page is displayed.

  4. Click Mask Target. 

The Mask Sensitive Data page is displayed.

  5. Select the target database that you want to mask. If needed, click Change Compartment, and browse to and select a different compartment. 
  6. (Optional) To customize the processing of the masking job for the first time or to override the existing options associated with the selected masking policy, do the following: 
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
  7. Click Mask Data. 

A warning message states that you should not mask data on a production
database.

The work request page is displayed so that you can see the progress of the
masking job.

You can run only one data masking job at a time on a target database.
