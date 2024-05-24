
## Group Masking

Group masking, also known as compound masking, enables you to mask related
columns together as a group, ensuring that the masked data across the related
columns retain the same relationship. You can use the group masking feature
when you create data masking jobs.

### About Group Masking

In a masking policy, the columns being masked as a group must belong to the
same table. You can use the Shuffle, User Defined Function, Deterministic
Substitution, and Random Substitution masking formats for group masking. The
Deterministic Substitution and Random Substitution masking formats use data
from another table to mask your sensitive data.

### Group Masking Example Using Shuffle

The following is an example of group masking using the Shuffle masking format.
Suppose that you have customers from across the world. You have their details
stored in a table, as shown below.

CUST_ID | CUST_NAME | CITY | STATE | COUNTRY  
---|---|---|---|---  
678123 | Michael Lee | Denpasar | Bali | Indonesia  
678124 | Sophia Lopes | Rio de Janeiro | Rio de Janeiro | Brazil  
678125 | Richard Williams | Santa Clara | California | United States  
678126 | Aaryan | Mumbai | Maharashtra | India  
  
You don't want your developers to know the location of your customers. So, you
want to mask the `CITY`, `STATE` and `COUNTRY` columns before sharing this
data with the development team. But you want to have realistic masked data.
For example, Richard lives in Santa Clara, California in the United States.
After masking, if the city and state are Atlanta and Georgia respectively,
India as the country is not valid. In this case, you want to ensure that the
country remains the United States.

You can group these columns and use the Shuffle masking format to shuffle them
together. After shuffling, your masked data might look like the data shown
below.

CUST_ID | CUST_NAME | CITY | STATE | COUNTRY  
---|---|---|---|---  
678123 | Michael Lee | Mumbai | Maharashtra | India  
678124 | Sophia Lopes | Denpasar | Bali | Indonesia  
678125 | Richard Williams | Rio de Janeiro | Rio de Janeiro | Brazil  
678126 | Aaryan | Santa Clara | California | United States  
  
### Group Masking Example Using Deterministic Substitution

This example shows you how to use the Deterministic Substitution masking
format with group masking to mask sensitive data with data from another table.
Suppose that you have customers from across the world. You have their details
stored in a table, as shown below.

CUST_ID | CUST_NAME | CITY | STATE | COUNTRY  
---|---|---|---|---  
678123 | Michael Lee | Denpasar | Bali | Indonesia  
678124 | Sophia Lopes | Rio de Janeiro | Rio de Janeiro | Brazil  
678125 | Richard Williams | Santa Clara | California | United States  
678126 | Aaryan | Mumbai | Maharashtra | India  
  
Let's assume that you want to use the data from the following table for group
masking:

SUB_CITY | SUB_STATE | SUB_COUNTRY  
---|---|---  
New York | New York | United States  
Noida | Uttar Pradesh | India  
Toronto | Ontario | Canada  
Cape Town | Western Cape | South Africa  
  
After masking these columns using the group masking option with the
Deterministic Substitution masking format, your masked data might look like
the data shown below.

CUST_ID | CUST_NAME | CITY | STATE | COUNTRY  
---|---|---|---|---  
678123 | Michael Lee | Cape Town | Western Cape | South Africa  
678124 | Sophia Lopes | Toronto | Ontario | Canada  
678125 | Richard Williams | New York | New York | United States  
678126 | Aaryan | Noida | Uttar Pradesh | India
