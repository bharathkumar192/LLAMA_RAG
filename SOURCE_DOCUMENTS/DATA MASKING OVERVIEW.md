
## Data Masking Overview

Data masking, also known as static data masking, is the process of permanently
replacing sensitive data with fictitious yet realistic looking data. It helps
you generate realistic and fully functional data with similar characteristics
as the original data to replace sensitive or confidential information.



Tablespace requirements for data masking:
Data Masking requires additional space, which today is around 5x where x is the size of the largest table being masked.
The breakdown of this 5x is:
TEMP tablespace should be roughly 2x and
The tablespace where masking is running should be roughly 3x.


### The Challenge

The amount of data that organizations collect and manage, including sensitive
and personal data, is growing every day. The growing security threats have
made it necessary to limit exposure of sensitive data. At the same time,
different data privacy laws and standards such as EU GDPR, PCI-DSS, and HIPPA
mandate you to protect personal data. Live production database environments
contain valuable and sensitive data, and to meet security and compliance
requirements, you need to protect this data. Usually, organizations implement
multiple security controls in their production environments to ensure that
access to sensitive data is tightly controlled.

You collect data probably to improve your products and services, provide
better user experience, and support and grow your business. To best utilize
the collected data, you need to share it with different teams, both internal
and external, for various use-cases such as development, testing, training,
and data analytics. Copying production data for non-production purposes
proliferates sensitive data, expands the security and compliance boundary, and
increases the likelihood of data breaches. If left unprotected, contractors or
offshore workers might access the data and possibly move it across locations.
Data privacy standards such as PCI-DSS and EU GDPR also emphasize on
protecting sensitive information in non-production environments because these
environments are typically not as protected or monitored as production
systems.

The challenge is to reduce the unnecessary spread and exposure of sensitive
data while maintaining its usability for non-production purposes.

### The Solution

Even in non-production environments, you need to protect your sensitive data
and stay compliant with data privacy regulations. The recommended solution is
to mask your sensitive data before using it in non-production environments.
This way, you minimize the sensitive data you have, and thus, reduce the risk
and compliance boundary.

Data masking, also known as static data masking, is the process of permanently
replacing sensitive data with fictitious yet realistic looking data. It helps
you generate realistic and fully functional data with similar characteristics
as the original data to replace sensitive or confidential information. Data
masking limits sensitive data proliferation by anonymizing sensitive data
while enabling you to use production-like data. It ensures that malicious
actors cannot benefit from the fictitious data even if they gain access to it.

Data masking is ideal for virtually any situation when confidential or
regulated data needs to be shared with non-production users. These users may
include internal users, such as application developers or external business
partners, such as offshore testing companies, suppliers, and customers. Data
masking contrasts with encryption, which simply hides data, and the original
data can be retrieved with the appropriate access or key. With data masking,
the original sensitive data cannot be retrieved or accessed.

One of the key aspects of data masking is to replace sensitive information
with fictitious data, without breaking the semantics and structure of the
data. The masked data must be realistic and pass specific checks, such as Luhn
validation. For example, a masked credit card number must not only be a valid
credit card number, but also a valid Visa, Mastercard, American Express, or
Discover card number. Failing to maintain this data integrity may break the
corresponding application. The predefined masking formats ensure that the
generated data passes common validation checks.

### Common Data Masking Requirements

Organizations typically mask data using custom scripts or solutions. While
these in-house solutions might work for a few columns, they do not work for
large applications with distributed databases and thousands of columns. An
enterprise data masking solution should be able to fulfill the following data
masking requirements:

  * Locate sensitive data in the midst of numerous applications, databases, and environments.
  * Correctly mask sensitive data having different shapes and forms such as names, Social Security numbers, email addresses, credit card numbers (Mastercard, Visa, and so on), and blood type.
  * Ensure that the masked data is irreversible, that is, one should not be able to retrieve the original data from the masked data.
  * Ensure that the masked data is realistic enough to be useful for non-production purposes such as development and analytics.
  * Ensure that the applications continue to work with the masked data.

### Data Masking in Oracle Data Safe

The Data Masking feature in Oracle Data Safe addresses the common data masking
requirements and more. It simplifies the process of masking data in your non-
production databases by providing an automated, flexible, and easy-to-use
solution. It enables you to:

  * Maximize the business value of your data without exposing the sensitive data
  * Minimize the compliance boundary by not proliferating the sensitive production data
  * Mask your Oracle databases
  * Use various masking techniques to meet your specific business requirements
  * Preserve data integrity ensuring that the masked data continues to work with applications

### Masking Policies and Masking Formats

By using Data Discovery and Data Masking in Oracle Data Safe, you can mask the
sensitive data in your target databases. Data Discovery lets you discover
sensitive data and referential (parent-child) relationships in a target
database, and generate a sensitive data model containing all the sensitive
columns and metadata. Data Masking lets you create a masking policy for the
sensitive data model and then apply that policy against a target database to
mask its sensitive data. Data masking ensures referential integrity by masking
related columns consistently.

When creating a masking policy, you select a masking format for each sensitive
column in the sensitive data model. A masking format defines the logic to mask
data in a database column. Oracle Data Safe automatically selects a default
predefined masking format for each sensitive column for you, however, you can
change the selections as needed. Oracle Data Safe provides a comprehensive set
of predefined masking formats for common sensitive and personal data, such as
names, national identifiers, credit card numbers, phone numbers, and religion.
For example, the Email Address masking format replaces values with random
email addresses. You can also create your own masking formats, use conditional
masking, and implement group masking.

One of the key aspects of data masking is to replace the sensitive information
with fictitious data, without breaking the semantics and structure of the
data. The masked data must be realistic and pass specific checks, such as Luhn
validation. For example, a masked credit card number must not only be a valid
credit card number, but also a valid Visa, Mastercard, American Express, or
Discover card number. Failing to maintain this data integrity may break the
corresponding application. The predefined masking formats ensure that the
generated data passes common validation checks.

Similar to sensitive data models, Oracle Data Safe stores your masking
policies and user-defined masking formats in compartments in Oracle Data Safe.
You can move them from compartment to compartment and delete them as needed.
To apply masking policies to target databases in different regions or to
simply edit masking policies manually in a text editor, you can download and
upload masking policies in XML format.

When configuring a data masking job, you have the option to upload scripts to
be run on the target database before and/or after the data masking job. For
example, you can upload a pre-masking script to create a column on the target
database that should be used for the Deterministic Substitution masking
format. And, you can upload a post-masking script to remove this column after
data masking completes.

### Characteristics of Masking Formats

Data masking formats have characteristics. Some common characteristics include
combinable, uniqueness, reversible, and deterministic. Oracle Data Safe has a
wide range of predefined and basic masking formats. Some masking formats
support double-byte characters, for example, Japanese, other Asian, and
Cyrillic characters.

#### Combinable

A masking format is considered combinable when it can be combined with other
basic masking formats or predefined masking formats though the use of
conditions.

For example, assume that you want to mask a column containing data in format
`999-999`, where `9` signifies a digit. You want to replace the first three
digits with a fixed three-digit number, preserve the hyphen, and replace the
last three digits with some random digits. To generate the expected data, you
could combine three basic masking formats: Fixed Number, Fixed String, and
Random Number, as shown in the following example. The outputs of these three
masking formats are concatenated to generate the masked values, for example,
`678-333`, `678-110`, `678-656`, and `678-999`.

    
    
    FIXED NUMBER 678
    FIXED STRING "-"
    RANDOM NUMBER [START:100 END: 999]

Another example uses a basic masking format with a predefined masking format.
Suppose you want to mask a social security number. The logic is: If a social
security number exists, replace it with a predefined social security number.
Otherwise, replace it with a random number.

#### Uniqueness

A masking format is characterized as having uniqueness if it ensures
uniqueness of the generated masked data. These types of masking formats are
useful for masking columns with uniqueness constraints. For example, you may
want to mask a column of EMPLOYEE IDs with unique ID masked values. No two
rows can have the same ID.

#### Reversible

A masking format that is characterized as reversible can retrieve original
column data from masked data. Data masking usually means permanently replacing
the data and ensuring that no one can retrieve the original data. But,
sometimes you might want to see the original data. Reversible masking is
helpful when businesses need to mask and send their data to a third party for
analysis, reporting, or any other business processing purpose. After the
processed data is received from the third party, the original data can be
recovered. The Deterministic Encryption masking format supports reversible
masking.

#### Deterministic

One of the key requirements for masking data in large databases or multiple
database environments is to mask some data consistently. That is, for a given
input, the output should always be the same. At the same time, the masked
output should not be predictable. A deterministic masking format generates
consistent output for a given input across databases and data masking jobs.
Deterministic masking helps to maintain data integrity across multiple
applications and preserve system integrity in a single sign-on environment.

For example, consider three applications: a human capital management
application, a customer relationship management application, and a sales data
warehouse. These three applications may have key common fields such as`
EMPLOYEE_ID` that must be masked consistently across these applications.
Deterministic masking techniques can be used here to ensure consistency.

Let's consider another example. Suppose that two values, Joe and Tom, are
masked to Henry and Peter by using a deterministic masking technique. When you
repeat the technique on another database, Bob and Tom (if they exist), might
be replaced with Louise and Peter. Notice that even though the two runs have
different data, Tom is always replaced with Peter.

The Deterministic Encryption, Deterministic Substitution, SQL Expression, and
User Defined Function masking formats support deterministic masking.

#### Characteristics of Each Data Masking Formats

Masking Format | Supports Double-Byte Characters* | Supports Large Object Columns | [Combinable](data-masking-overview.html#GUID-F3FB8BAE-9ADF-4E35-AD39-12BB5C4E247E) | [Deterministic](data-masking-overview.html#GUID-3A6CDF21-B546-4947-8C64-359B495CC10E) | [Reversible](data-masking-overview.html#GUID-741A8F8A-46DF-4347-A48D-F03B4C330CEF) | [Uniqueness](data-masking-overview.html#GUID-0235249A-6567-43F8-8DF7-1484A0DEEE67)  
---|---|---|---|---|---|---  
Age | Number - Not Applicable | No | Yes | No | No | No  
Bank Account Number | Number - Not Applicable | No | Yes | No | No | Yes, if the number of distinct values that can be generated by the LMF is greater than the number of distinct values in the column  
Bank Routing Number | Number - Not Applicable | No | Yes, if the generated masked values passes the post processing function validation | No | No | Yes, if the number of distinct values that can be generated by the LMF is greater than the number of distinct values in the column  
Blood Type | No | No | Yes | No | No | No  
Canada Postal Code | No | No | Yes | No | No | No  
Canada Social Insurance Number | Number - Not Applicable | No | Yes, if the generated masked values passes the post processing function validation | No | No | Yes, if the number of distinct values that can be generated by the LMF is greater than the number of distinct values in the column  
Canada Social Insurance Number (hyphenated) | Number - Not Applicable | No | Yes, if the generated masked values passes the post processing function validation | No | No | Yes, if the number of distinct values that can be generated by the LMF is greater than the number of distinct values in the column  
Credit Card Number | Number - Not Applicable | No | Yes, if the generated masked values passes the post processing function validation | No | No | Yes, if the number of distinct values that can be generated by the LMF is greater than the number of distinct values in the column  
Credit Card Number (Hyphenated) | Number - Not Applicable | No | Yes, if the generated masked values passes the post processing function validation | No | No | Yes, if the number of distinct values that can be generated by the LMF is greater than the number of distinct values in the column  
Credit Card Number-American Express | Number - Not Applicable | No | Yes, if the generated masked values passes the post processing function validation | No | No | Yes, if the number of distinct values that can be generated by the LMF is greater than the number of distinct values in the column  
Credit Card Number-Discover | Number - Not Applicable | No | Yes, if the generated masked values passes the post processing function validation | No | No | Yes, if the number of distinct values that can be generated by the LMF is greater than the number of distinct values in the column  
Credit Card Number-Mastercard | Number - Not Applicable | No | Yes, if the generated masked values passes the post processing function validation | No | No | Yes, if the number of distinct values that can be generated by the LMF is greater than the number of distinct values in the column  
Credit Card Number-Visa | Number - Not Applicable | No | Yes, if the generated masked values passes the post processing function validation | No | No | Yes, if the number of distinct values that can be generated by the LMF is greater than the number of distinct values in the column  
Date-Card Expiration | Number - Not Applicable | No | Yes | No | No | Yes, if the number of distinct values that can be generated by the LMF is greater than the number of distinct values in the column  
Date-Past | Number - Not Applicable | No | Yes | No | No | Yes, if the number of distinct values that can be generated by the LMF is greater than the number of distinct values in the column  
Delete Rows | Number - Not Applicable | No | No | Not Applicable | No | Not Applicable  
Deterministic Encryption | Yes | No | No | Yes | Yes | Yes. Refer to the Inputs section to see specific conditions.  
Deterministic Substitution | Yes | No | No | Yes, as long as the values in the substitution column do not change and you provide the same seed value | No | No  
Email Address | No | No | Yes | No | No | No  
Finland Personal Identity Code | Number - Not Applicable | No | Yes, if the generated masked values passes the post processing function validation | No | No | Yes, if the number of distinct values that can be generated by the LMF is greater than the number of distinct values in the column  
Fixed Number | Number - Not Applicable | Yes | Yes | Yes | No | No  
Fixed String | Yes | Yes | Yes | Yes | No | No  
Format Preserving Randomization | No | No | Yes | No | No | No  
Gender | No | No | Yes | No | No | No  
Group Shuffle | Yes | No | No | No | No | Yes, this masking format ensures uniqueness for columns that have unique constraints  
Height (Centimeter) | Number - Not Applicable | No | Yes | No | No | No  
Identification Number | Number - Not Applicable | No | Yes | No | No | Yes, if the number of distinct values that can be generated by the LMF is greater than the number of distinct values in the column  
IMEI Number | Number - Not Applicable | No | Yes, if the generated masked values passes the post processing function validation | No | No | Yes, if the number of distinct values that can be generated by the LMF is greater than the number of distinct values in the column  
Income | Number - Not Applicable | No | Yes | No | No | Yes, if the number of distinct values that can be generated by the LMF is greater than the number of distinct values in the column  
Marital Status | No | No | Yes | No | No | No  
Null Value | Number - Not Applicable | Yes | No | Yes | No | No  
Post Processing Function | Yes | No | Yes | Not Applicable | Not Applicable | Not Applicable  
Preserver Original Data | Yes | No | No | Yes | Not Applicable | If the original values are unique, they will remain unique after masking  
Race | No | No | Yes | No | No | No  
Random Date | Number - Not Applicable | No | Yes | No | No | Yes. The total number of distinct values in the specified range must be greater than or equal to the number of values in the column  
Random Decimal Number | Number - Not Applicable | No | Yes | No | No | Yes. The total number of distinct values in the specified range must be greater than or equal to the number of values in the column  
Random Digits | Number - Not Applicable | No | Yes | No | No | Yes, however, if you do not specify a sufficient length range, you can run out of unique values within the range  
Random List | Yes | No | Yes | No | No | Yes. The input list must have unique values, and the number of values in the list must be greater than or equal to the number of values in the column to be masked.  
Random Name | No | No | Yes | No | No | Yes, if the number of distinct values that can be generated by the LMF is greater than the number of distinct values in the column  
Random Number | Number - Not Applicable | No | Yes | No | No | Yes. The number of distinct values in the specified range must be greater than or equal to the number of values in the column  
Random String | No | No | Yes | No | No | Yes. The number of distinct values in the specified range must be greater than or equal to the number of values in the column  
Random Substitution | Yes | No | No | No | No | Yes. The number of distinct values in the substitution column must be greater than or equal to the number of values in the column to be masked  
Regular Expression | Yes | Yes | Yes | No | No | No  
Religion | No | No | Yes | No | No | No  
Sexual Orientation | No | No | Yes | No | No | No  
Shuffle | Yes | No | No | No | No | Yes, provided the column values are all unique  
SQL Expression | Y, the sql expression provided should generate multi byte chars | Yes | No | Yes, depending on the SQL expression defined | No | Yes, but the uniqueness is not guaranteed and depends on the SQL expression defined. However, because ORA_HASH uses a 32-bit algorithm, and considering the birthday paradox or pigeonhole principle, there is a 0.5 probability of collision after 232-1 unique values.  
Stock | Number - Not Applicable | No | Yes | No | No | Yes, if the number of distinct values that can be generated by the LMF is greater than the number of distinct values in the column  
Substring | Yes | No | Yes | Yes | No | No  
Truncate Data | Yes | No | No | Not Applicable | Not Applicable | Not Applicable  
UK National Insurance Number (Space-Separated) | Number - Not Applicable | No | Yes, if the generated masked values passes the post processing function validation | No | No | Yes, if the number of distinct values that can be generated by the LMF is greater than the number of distinct values in the column  
UK Postal Code (Space-Separated) | Number - Not Applicable | No | Yes | No | No | Yes, if the number of distinct values that can be generated by the LMF is greater than the number of distinct values in the column  
URL | No | No | Yes | No | No | Yes, if the number of distinct values that can be generated by the LMF is greater than the number of distinct values in the column  
US Phone Number | Number - Not Applicable | No | Yes, if the generated masked values passes the post processing function validation | No | No | Yes, if the number of distinct values that can be generated by the LMF is greater than the number of distinct values in the column  
US Phone Number (With Country Code) | Number - Not Applicable | No | Yes, if the generated masked values passes the post processing function validation | No | No | Yes, if the number of distinct values that can be generated by the LMF is greater than the number of distinct values in the column  
US Social Security Number | Number - Not Applicable | No | Yes, if the generated masked values passes the post processing function validation | No | No | Yes, if the number of distinct values that can be generated by the LMF is greater than the number of distinct values in the column  
US Social Security Number (Hyphenated) | Number - Not Applicable | No | Yes, if the generated masked values passes the post processing function validation | No | No | Yes, if the number of distinct values that can be generated by the LMF is greater than the number of distinct values in the column  
User Defined Function | Yes | No | Yes | Yes, depending on the function defined | No | Yes, depending on the function defined  
Weight (Pound) | Number - Not Applicable | No | Yes | No | No | Yes, if the number of distinct values that can be generated by the LMF is greater than the number of distinct values in the column  
  
*If a masking format supports double-byte characters then you can use it for characters from languages such as Japanese, Chinese, and Cyrillic.

### Data Masking Dashboard

The Data Masking dashboard, as shown below, provides a high-level view of your
masked target databases in your selected compartment(s). The two charts at the
top of the dashboard focus on your top five target databases. The first chart
helps you to identify which target databases have the most masked columns by
showing you a percentage breakdown of masked columns across the five targets.
The second chart helps you to identify which target databases contain the most
masked values by showing you a percentage breakdown of masked values across
the five targets.

The table below the charts shows statistics about the target databases in the
selected compartment(s). Each target database listed must have been masked at
least once. You can view the number of masking policies created for each
target database, the number of masked sensitive types, masked schemas, masked
tables, masked columns, and masked values on each target database.

You can explore key features and workflows with the guided tour option by
clicking the "Take the tour" button in the Data Masking dashboard.

![Description of data-masking-dashboard.png follows](img/data-masking-
dashboard.png)  
[Description of the illustration data-masking-dashboard.png](img_text/data-
masking-dashboard.html)

You can click a target database name to view a masking summary for just that
target database and details about its masking policies. The following
screenshot shows you a summary and details for a target database.

### Data Masking Workflow

Oracle recommends that you use the following approach and workflow to mask
sensitive data with Oracle Data Safe.

  1. Important: Create a backup of your production database. For example, you can use Recovery Manager (RMAN) and Oracle Cloud Storage service (or any other backup location) to create and store your production backups. You never want to mask the actual production database. 
  2. Clone the backup of your production database to create a stage database. Do not expose the stage database to users. Create the stage database on the Oracle Cloud with supported services.
  3. [Register your stage database withOracle Data Safe](/pls/topic/lookup?ctx=en/cloud/paas/data-safe&id=ADMDS-GUID-B5F255A7-07DD-4731-9FA5-668F7DD51AA6). 
  4. [Use Data Discovery](/pls/topic/lookup?ctx=en/cloud/paas/data-safe&id=UDSCS-GUID-B647CD31-3354-4757-81A8-5762247CFFA5) to discover sensitive data on the stage database and generate a sensitive data model. 
  5. [Use Data Masking to create new masking formats](/pls/topic/lookup?ctx=en/cloud/paas/data-safe&id=UDSCS-GUID-48F42F04-E3F1-4C26-9882-E12FFBB250DC) if you require masking formats other than the Oracle predefined ones. 
  6. [Use Data Masking to create a masking policy](/pls/topic/lookup?ctx=en/cloud/paas/data-safe&id=UDSCS-GUID-DD6B344E-3846-436D-A99E-A81F25860BF1) that associates default masking formats with the sensitive columns. You can change the formats as needed. 
  7. Run a [Pre-Masking Check](masking-check.html#GUID-D1A7E2B8-143E-497E-89D5-F069BF9D50B0 "Prior to initiating a masking job, a pre-masking check should be run. The pre-masking check performs a number of checks on the selected target database to determine if a masking run can be successfully performed.") on your stage database to check if it is ready for masking. Perform any remediation recommendations prior to initating a masking job. 
  8. [Run a data masking job against your stage database](/pls/topic/lookup?ctx=en/cloud/paas/data-safe&id=UDSCS-GUID-A4A8B496-68F0-4386-91F8-3FB6D56520DD) to mask the sensitive data using the masking policy. Oracle Data Safe also generates a data masking report that shows you the results of your data masking job. 
  9. [Verify the masked data](/pls/topic/lookup?ctx=en/cloud/paas/data-safe&id=UDSCS-GUID-5E7F3F1A-BC9C-4A07-80FD-E5C48C079E90) by reviewing the Data Masking report and by validating the masked data. 
  10. Clone the stage database to create a test database. Or, export the masked data from the stage database, create a test database, and then import the masked data into the test database. Oracle strongly recommends creating a test database instead of giving your test and developer users access to your stage database. 
  11. Grant your test and developer users access to your test database.
  12. [Set up event notifications](/pls/topic/lookup?ctx=en/cloud/paas/data-safe&id=UDSCS-GUID-CA74CEE3-C2D3-41BC-8C28-BC91FE137537). For example, you can subscribe to the `Masking Job Begin` event to be automatically informed if a masking job is initiated. 

### Prerequisites for Using Data Masking

These are the prerequisites for using Data Masking:

  * Register the target databases that you want to use with Data Masking.
  * Grant the Data Masking role on the target database. A Database Administrator can grant this role to the Oracle Data Safe Service Account on the target database. 
  * Obtain permission in Oracle Cloud Infrastructure Identity and Access Management (IAM) to use the Data Masking feature in Oracle Data Safe. An OCI administrator can grant these permissions. These resources require permissions: 
    * `data-safe-masking-policies `
    * `data-safe-library-masking-formats`
    * `data-safe-masking-reports`

In order to perform data masking a user will need `manage` permissions on
`data-safe-masking-reports` in the compartment of the target database.

    * `data-safe-masking-policy-health-report`
    * `data-safe-work-requests`

As an alternative to selectively granting permissions, you can grant
permissions on `data-safe-masking-family` in the relevant compartments, which
would include permissions on all of the resources above. See [data-safe-
masking-family Resource](/pls/topic/lookup?ctx=en/cloud/paas/data-
safe&id=ADMDS-GUID-56A59CC3-45E4-411C-B862-E825FDC23E1D) in the Administering
Oracle Data Safe guide for more information.

