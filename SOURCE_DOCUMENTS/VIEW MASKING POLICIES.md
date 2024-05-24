
## View Masking Policies

You can view masking policies from the Masking Policies page. Masking policies
are either created from this page or uploaded to this page.

### Search for a Masking Policy

  1. Under Security Center, click Data Masking.
  2. Under Related Resources, click Masking Policies.
  3. Under List Scope, select the compartment that contains your masking policy. Optionally select Include child compartments to include masking policies in the list from child compartments.
  4. To filter the list of masking policies, under Filters, do the following:
    1. (Optional) To search by state: From the State drop-down list, select a state (Any state, Creating, Updating, Active, Deleting, Deleted, or Failed).
    2. (Optional) To search by name: In the Masking Policy Name box, enter the full and exact masking policy name. The search is case-sensitive.
    3. Click Apply Filters.
  5. (Optional) If there are multiple pages of masking policies, click the left and right navigation arrow buttons at the bottom of the page to navigate between pages.

### View Details for a Masking Policy

  1. Under Security Center, click Data Masking.
  2. Under Related Resources, click Masking Policies.
  3. Search for and click the name of the masking policy that you want to view.

The Masking Policies Details page is displayed.

  4. View the masking policy details.

     * The Masking Policy Information tab shows you the name and Oracle Cloud Identifier (OCID) of your masking policy, the work request information, the compartment in which the masking policy is stored, the target database with which the masking policy is associated, the name of the sensitive data model, and when the masking policy was created and last updated. 
     * The Masking Columns section shows you the list of sensitive columns, their associated masking formats, and if a column has child columns. 
       * If a column has child columns, click on View Details to view the name and location of the child column(s). 

Note:

Child column(s) will have the same masking format applied as their parent
columns.

  5. To view the work requests related to the masking policy, you can do the following:
    1. To view the latest work request, on the Masking Policy Information tab, click the View Details link next to Work Request. The Work Request page is displayed. Here you can view the work request information, log messages, and error messages (select Error Messages under Resources).
    2. To view all the work requests for the past seven days (work requests are stored for only 7 days in Oracle Cloud Infrastructure), under Resources, click Work Requests. From here, you can view the status (for example, SUCCEEDED or FAILED), percent completed, date started, and date finished details for each work request. Click a particular work request to view its log messages and error messages.
    3. (Optional) If there was a work request failure, notice the error message displayed at the top of the page, for example, "There is at least one work request associated with this policy that has failed."
  6. To explore the list of masking columns, do the following:
    1. Select one or more schemas from the Schema Name list. Click Load more if there are more than 1000 schemas and your desired schema is not already listed. 
    2. Select one or more tables from the Table Name list. Click Load more if there are more than 1000 tables and your desired table is not already listed. 
    3. Select one or more columns from the Column Name list. A list of columns will only be available once either a schema or table is selected. Click Load more if there are more than 1000 columns and your desired column is not already listed. 
    4. Click Show More Options to filter by sensitive type.
    5. Select a sensitive types from the Sensitive Type list. Click Load more if there are more than 1000 sensitive types and your sensitive type is not already listed.
    6. When all of your filters are created, click Apply.
    7. To remove a filter, click the X button next to the selected item.
