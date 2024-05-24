  3. View and Analyze Data Masking Reports

## View and Analyze Data Masking Reports

After you run a data masking job, Oracle Data Safe saves all the details about
the data masking job in Security Center as an online report and tracks
statistics across the top five masking policies for a target database.

### Statistics About a Masked Target Database

The following information is available in Data Masking for a masked target
database:

  * The number of masking policies created for a target database and their names
  * Statistics across all policies for a target database: Includes the number of masked sensitive types, schemas, tables, columns, and values
  * Charts comparing the percentage of masked values and percentage of masked columns for the top five masking policies.
  * Statistics per masking policy: Includes the number of reports available, masked sensitive types, schemas, tables, columns, and values
  * Details and statistics for each data masking job: Includes the target database name, masking policy name, report Oracle Cloud Identifier (OCID), the start and finish date/time of the data masking job, and the number of masked sensitive types, schemas, tables, columns, and values. For each column, the schema name, table name, masking format used, sensitive type, parent column, and total number of masked values is provided. Log files are available for the data masking job.

### View and Analyze Masked Data for a Target Database

  1. Under Security Center, click Data Masking. 
  2. On the Masked Target Databases tab, click the name of the target database for which you want to view the Data Masking report.

Notice that the Masking Policies column tells you how many masking policies
exist for the target database.

  3. On the Masking Summary tab, view statistics across all masking policies for the target database. 

You can view the number of masked sensitive types, schemas, tables, columns,
and values. There are two charts included. The first chart compares the
percentage of masked values for the top five masking policies. The second
chart compares the percentage of masked columns for the top five masking
policies.

  4. In the Masking Polices section, view the list of masking policies for the target database. 

Notice that the Masking Reports column tells you how many reports exist for
each masking policy.

  5. In the Masking Reports column, click the numerical link for a masking policy report.
  6. On the Masking Summary tab, view statistics for the latest data masking job. 

You can view the target database name, the masking policy name, and the number
of masked sensitive types, schemas, tables, columns, and values.

  7. In the Masking Reports section, you can view totals for each masking job, including the number of masked sensitive types, schemas, tables, columns, and values. 

The Report Time column provides a link to each Data Masking report.

Notice that each report is named according to its date and time.

  8. In the Report Time column, click the link to a Data Masking report.

The Masking Report page is displayed.

  9. On the Masking Report Information tab, view details and statistics for the data masking job. 

Details include the target database name, masking policy name, report OCID
(Oracle Cloud Identifier), and the start and finish date/time of the data
masking job. Statistics included are the number of masked sensitive types,
schemas, tables, columns, and values.

  10. In the Masked Columns section, view the details about each masked column. For each column, the schema name, table name, masking format used, sensitive type, parent column, and total number of masked values is provided.
  11. To view the data masking job logs, under Resources, click Masking Logs.

The Masking Logs section lists the log messages and when they occurred.
