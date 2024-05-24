
## Pre-Masking Check

Prior to initiating a masking job, a pre-masking check should be run. The pre-
masking check performs a number of checks on the selected target database to
determine if a masking run can be successfully performed.

Tip:

It is strongly recommended to run the pre-masking check.

The pre-masking check performs the following checks:

  * Has the Data Safe service account been granted the required data masking privileges on the database? 

For more information see, [Grant Roles to the Oracle Data Safe Service Account
on Your Target Database](/pls/topic/lookup?ctx=en/cloud/paas/data-
safe&id=ADMDS-GUID-6E55095B-4963-4568-A7B6-74508B30A25E).

  * Is there sufficient free space in the default tablespace or the one that was specified in the pre-masking check? The default tablespace is the one that was specified when the Data Safe service account was created in the database. 

For more information see, [Create an Oracle Data Safe Service Account on a
Target Database](/pls/topic/lookup?ctx=en/cloud/paas/data-safe&id=ADMDS-
GUID-E93C203A-6F57-43FC-BEF9-98581254AD48)

  * Are the statistics for the tables in the masking policy up-to-date? 

For more information see, [GATHER_TABLE_STATS
Procedure](/pls/topic/lookup?ctx=dblatest&Id=ARPLS-GUID-
CA6A56B9-0540-45E9-B1D7-D78769B7714C) in the Oracle Database PL/SQL Packages
and Types Reference guide.

  * Are there zero invalid objects? 

If there are dependent objects then masking might encounter errors when
recreating the objects.

  * Is there at least one masking column in the masking policy?
  * Do all masking columns in the masking policy exist in the target database?
  * Are there zero database/system level triggers?
  * Are there zero Oracle Label Security (OLS) policies in the masking policy?
  * Are there zero Virtual Private Database (VPD) policies in the masking policy?
  * Are there no active masking jobs currently being performed on the database?

If any of the above checks fail, it is recommended to perform the remediation
actions listed in the pre-masking report. Once the issues have been
remediated, perform the pre-masking check again to ensure the masking job will
complete successfully. Once all of the checks have passed, you can perform a
masking job.

### Perform a Pre-masking Check

Prior to initiating a masking job, you should perform a pre-masking check to
ensure the masking job will be successful. If the pre-masking check produces
any failures then you should perform the remediation recommendations.

Tip:

It is strongly recommended to run the pre-masking check.

  1. Under Security center, click Data masking. 
  2. Click Pre-masking check. 

The Pre-masking check window is displayed.

  3. Select a target database. If needed, click Change compartment and browse to and select a different compartment. 
  4. Select a masking policy. If needed, click Change compartment and browse to and select a different compartment. 
  5. (Optional) Enter the tablespace that you want to use for masking if is different than the default tablespace of the Data Safe service account.
  6. Click Submit. 
  7. Wait for the pre-masking check to finish. Perform any remediation actions and ensure all checks pass prior to initiating a masking job. This may require running an additional pre-masking check. 

Note:

Though it is not recommended, a masking job can be performed even if there are
invalid objects.

### View a Pre-masking Check Report

After performing a pre-masking check, you will need to view the report to
determine if checks were failed or passed. If the pre-masking check produces
any failures then you should perform the remediation recommendations.

  1. Under Security center, click Data masking. 
  2. Under Related resources, click Pre-masking reports. 
  3. (Optional) Under List scope, select the compartment that contains your target database. Optionally select Include child compartments to include target database in the list from child compartments. 
  4. (Optional) Under Filters, narrow down the scope of reports by selecting a Policy name, Target database, or entering a Report name. 
  5. From the list of reports, select the one you want to view.
