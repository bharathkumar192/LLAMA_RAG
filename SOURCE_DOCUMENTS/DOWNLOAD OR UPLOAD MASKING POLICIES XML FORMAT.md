  3. Download or Upload Masking Policies in XML Format

## Download or Upload Masking Policies in XML Format

You can download an XML version of a masking policy and upload it into Oracle
Data Safe, replacing an existing masking policy or creating a new one. Before
downloading a masking policy, you first need to generate it as an XML file.

### About Downloading and Uploading Masking Policies

There are several use cases for downloading and uploading masking policies.
For example:

  * You have multiple test databases in different regions in Oracle Cloud Infrastructure, all with the same schemas, and you want to mask them all the same way.
  * Your test database has moved to another region in Oracle Cloud Infrastructure and you want to move the masking policy with it.
  * Your masking policy is large and complex so you prefer to manually edit it in a text editor instead of going through the Data Masking interface.

### Generate a Masking Policy in XML Format

  1. Under Security Center, click Data Masking. 
  2. Under Related Resources, click Masking Policies.

The Masking Policies page is displayed.

  3. Search for and click the masking policy that you want to download.

The Masking Policy Details page is displayed.

  4. Click Generate Policy.

The Generate Masking Policy for Download is displayed.

  5. Click Generate Policy and wait for the XML file to be generated.

A message states the XML file generation is complete. You can download it
using the Download Policy button.

  6. Click Close.

### Download a Masking Policy in XML Format

You need to first generate the masking policy before you can download it.

  1. Under Security Center, click Data Masking. 
  2. Under Related Resources, click Masking Policies.

The Masking Policies page is displayed.

  3. Search for and click the masking policy that you want to download.

The Masking Policy Details page is displayed.

  4. Click Download Policy.

The Download Masking Policy dialog box is displayed.

  5. Click Download Policy.

The Opening Policy-download.xml dialog box is displayed.

  6. Either open the XML file with a selected application or leave Save File selected, and click OK. If you choose to save the file, browse to a location on your local computer, enter a file name, and click Save.

### Upload a Masking Policy in XML Format

  1. Under Security Center, click Data Masking. 
  2. Under Related Resources, click Masking Policies.

The Masking Policies page is displayed.

  3. Click Upload Masking Policy.

The Upload Masking Policy page is displayed.

  4. To replace an existing masking policy, do the following:
    1. Leave the Update an existing masking policy tile selected.
    2. Select the masking policy that you want to replace. If needed, click Change Compartment and select a different compartment.
    3. Add your masking policy. There are two ways to do this. The first way is to drag your masking policy file (XML file) onto the Upload Masking Policy File area. The second way is to click select one, browse to and select your XML file in the File Upload dialog box, and then click Open.
    4. Click Upload Masking Policy.
  5. To create a new masking policy using the XML file, do the following:
    1. Select the Create a new masking policy tile.
    2. Enter a name for your new masking policy.
    3. Select the compartment where you want to store your masking policy.
    4. (Optional) Enter a description for your masking policy.
    5. Choose how you want to create the masking policy, either associating it with a Sensitive Data Model or with a Target Database. When using a sensitive data model, select the sensitive data model from the drop down menu. When using a target database, select the target database from the drop down menu.
    6. Add your masking policy. There are two ways to do this. The first way is to drag your masking policy file (XML file) onto the Upload Masking Policy File area. The second way is to click select one, browse to and select your XML file in the File Upload dialog box, and then click Open.
    7. (Optional) To add tags, click Show Advanced Options, and create tags.
    8. Click Upload Masking Policy.

It's important to leave the window open during the upload process.
