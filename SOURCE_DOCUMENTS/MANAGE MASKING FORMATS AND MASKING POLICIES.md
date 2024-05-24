  3. Manage Masking Formats and Masking Policies

## Manage Masking Formats and Masking Policies

You can move user-defined masking formats and masking policies to different
compartments and delete them as needed. You cannot delete Oracle predefined
masking formats.

### Change Target Database of a Masking Policy

Learn how to edit the target database of an existing masking policy.

  1. Under Security Center, click Data Masking.
  2. Under Related Resources, click Masking Policies.
  3. Click the name of your masking policy to view its details.

The Masking Policies Details page is displayed. The Masking Policy Information
tab shows you the name and Oracle Cloud Identifier (OCID) of your masking
policy, the work request information, the compartment in which the masking
policy is stored, the target database with which the masking policy is
associated, the name of the sensitive data model, and when the masking policy
was created and last updated.

  4. Change the target database associated with the masking policy by clicking the pencil icon next to the Target Database field.

The Change Target Database dialog will appear.

  5. (Optional) change the compartment by selecting Change Compartment. Select the new compartment from the list.
  6. Select the new target database from the target database drop-down list.
  7. Click Submit.

### Move a Masking Format or Masking Policy to a Different Compartment

  1. Under Security Center, click Data Masking. 
  2. Under Related Resources, click Masking Formats or Masking Policies.
  3. Search for and click the name of the masking format or masking policy that you want to move.

The Masking Format Details page or the Masking Policy Details page is
displayed.

  4. Click Move Resource.

The Move Resource to a Different Compartment dialog box is displayed.

  5. Select a different compartment, and then click Move Resource.

The masking format or masking policy is immediately moved to the selected
compartment.

### Delete a User-Defined Masking Format or Masking Policy

Deleting a masking format is permanent and cannot be undone.

  1. Under Security Center, click Data Masking. 
  2. Under Related Resources, click Masking Formats or Masking Policies.
  3. Search for and click the name of the masking format or masking policy that you want to delete.

The Masking Format Details page or the Masking Policy Details page is
displayed.

  4. If you want to delete a masking format, click Delete. If you want to delete a masking policy, from the More Actions menu, select Delete.

A Confirm dialog box is displayed, asking you to confirm the deletion.

  5. Click Delete.

The user-defined masking format or masking policy is immediately deleted.
