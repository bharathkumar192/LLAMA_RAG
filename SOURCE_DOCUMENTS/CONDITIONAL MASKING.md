
## Conditional Masking

Conditional masking allows you to set multiple logical conditions that alter
the masking format of a masking column. Conditional masking can only be done
when editing an existing masking format of a masking column in a masking
policy.

### Example 1: Protecting Sensitive Identifiers Across Diverse Geographic
Regions

Learn how conditional masking can be used to mask unique personal identifiers
based on country.

Problem

A large organization manages a database containing personal identifiers such
as Social Security Numbers, National Insurance Numbers, and so on from
individuals living in various countries. They are required by regulations and
data protection laws to safeguard the sensitive information while having to
maintain usability for authorized purposed. However, sharing this data for
testing, development, or analysis poses significant privacy risks.

Solution

By utilizing Data Masking available in Data Safe, the organization is able to
assign the appropriate masking formats to the personal identifiers to meet the
privacy regulations. Using conditional masking allows for the masking format
to change based on the country of residence listed in the database.

Consider the database contains the following information:

Table 7-1 Employee Personal Identifiers, Pre-Masking

Employee | Country | Identifier  
---|---|---  
Alice | US | 987-65-4320  
Bill | UK | BH 123654G  
Carol | UK | AJ 763482K  
Denise | US | 798-66-4329  
  
Following the implementation of a conditional masking format conditional on
the country, the database may look something like the following:

Table 7-2 Employee Personal Identifiers, Post-Masking

Employee | Country | Identifier  
---|---|---  
Alice | US | 674-58-2371  
Bill | UK | PA 123456C  
Carol | UK | AB 987654B  
Denise | US | 543-23-5431  
  
Benefits

By implementing conditional masking formats, an organization:

  * Prevents unauthorized access to sensitive personal identifiers. 
  * Complies with diverse regional privacy laws and data protection requirements. 
  * Preserves data integrity and usefulness for authorized activities, such as testing and analysis. 
  * Reduces the risk of data breaches and potential harm to individuals.
  * Enables secure data sharing for collaboration and knowledge advancement.

**Related Topics**

  * [Add Conditions to a Masking Format](edit-masking-policies.html#GUID-F8CF6FF2-E09F-4892-BC3E-B1AD932977C5 "See the steps and examples below to implement conditional masking formats in your masking policies.")

### Example 2: Protecting Sensitive Salary Data Across Different Employee
Groups

Learn how conditional masking can be used to mask salary data based on an
employee's role.

Problem

A company needs to analyze salary data to identify potential pay gaps between
different employee groups, but is unable to share actual salary figures due to
internal privacy concerns or competitive reasons.

Solution

By utilizing Data Masking available in Data Safe, the company is able to
create a pseudonymized dataset suitable for salary disparity analysis. Using
conditional masking allows for the original salary data to be masked by a
random number in a specified range based on the employee group.

Consider the database contains the following information:

Table 7-3 Employee Salary Data, Pre-Masking

Employee | Job Category | Salary  
---|---|---  
Alice | Manager | 90,000  
Bill | Manager | 88,000  
Carol | Worker | 72,000  
Denise | Worker | 57,000  
Eddie | Worker | 70,000  
Frank | Worker | 45,000  
George | Assistant | 45,000  
  
Following the implementation of a conditional masking format conditional on
the following:

  * If job category is Manager, replace salary with a random number from 100000 through 150000.
  * If job category is Worker, set salary to a fixed number (75000).
  * Default is to preserve the existing value.

The database may look something like the following:

Table 7-4 Employee Salary Data, Post-Masking

Employee | Job Category | Salary  
---|---|---  
Alice | Manager | 100,200  
Bill | Manager | 132,000  
Carol | Worker | 75,000  
Denise | Worker | 75,000  
Eddie | Worker | 75,000  
Frank | Worker | 75,000  
George | Assistant | 45,000  
  
Benefits

By implementing conditional masking formats, the company:

  * Protects individual employee salary information while enabling analysis of potential pay gaps between different job categories. 
  * Maintains data utility by creating a masked dataset that retains the statistical properties necessary for identifying salary disparity trends. 
  * Supports internal fairness by enabling data-driven decisions to promote fair compensation practices within the organization.

**Related Topics**

  * [Add Conditions to a Masking Format](edit-masking-policies.html#GUID-F8CF6FF2-E09F-4892-BC3E-B1AD932977C5 "See the steps and examples below to implement conditional masking formats in your masking policies.")
