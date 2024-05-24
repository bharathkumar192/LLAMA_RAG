
## Predefined Masking Formats

To help you mask common sensitive and personal data, such as credit card
numbers, phone numbers, and national identifiers, Oracle Data Safe provides
predefined masking formats. You can use predefined masking formats as is
without providing any input. You cannot edit or delete predefined masking
formats.

The following table describes the predefined masking formats.

Masking Format | Description  
---|---  
Age |  Replaces values with random numbers from 0 through 110 Examples:

  * `18`
  * `75`
  * `102`

  
Bank Account Number |  Replaces values with random 9 to 16-digit numbers Examples:

  * `7411024398`
  * `392663014671`
  * `24914700572445`

  
Bank Routing Number |  Replaces values with random bank routing numbers. Ensures that the routing numbers pass the checksum test. Examples:

  * `121122676`
  * `322271627`
  * `061000052`

  
Blood Type |  Replaces column data with values picked randomly from the following list:

  * A+
  * A-
  * B+
  * B-
  * AB+
  * AB-
  * O+
  * O-

  
Canada Postal Code (Space-Separated) | Replaces values with random Canada postal codes, which are in `A9A A9A` format, where `A` signifies a letter and `9` signifies a digit Examples:

  * `T7S T3R`
  * `J0L G6L`
  * `E4B L0V`

Details: First character:

  * Randomly picks letters from A to Z except D, F, I, O, Q, U, W, and Z

Second character:

  * Randomly picks digits from 0 to 9

Third character:

  * Randomly picks letters from A to Z except D, F, I, O, Q, and U

Fourth character:

  * Space

Fifth character:

  * Randomly picks letters from A to Z except D, F, I, O, Q, and U

Sixth character:

  * Randomly picks digits from 0 to 9

Seventh character:

  * Randomly picks letters from A to Z except D, F, I, O, Q, and U

  
Canada Social Insurance Number |  Replaces values with random Canada Social Insurance Numbers. Ensures that the numbers pass the Luhn's validation. Examples:

  * `688637008`
  * `346612823`
  * `734411531`

  
Canada Social Insurance Number (Hyphenated) |  Replaces values with random Canada Social Insurance Numbers, which are in 999-999-999 format, where 9 signifies a digit. Ensures that the numbers pass the Luhn validation. Examples:

  * `688-637-008`
  * `346-612-823`
  * `734-411-531`

  
Credit Card Number |  Replaces values with random credit card numbers. Generates card numbers of types: American Express, Diners Club, Discover, enRoute, JCB, Mastercard, and Visa. Ensures that the numbers pass the Luhn validation. Examples:

  * `4485780314771620`
  * `6011867455059259`
  * `5253901798047025`

  
Credit Card Number (Hyphenated) |  Replaces values with random hyphenated credit card numbers. It generates card numbers of type: American Express, Diners Club, Discover, enRoute, JCB, Mastercard, and Visa. Ensures that the numbers pass the Luhn validation. Examples:

  * `4485-7803-1477-1620`
  * `6011-8674-5505-9259`
  * `5253-9017-9804-7025`

  
Credit Card Number-American Express |  Replaces values with random 15-digit American Express credit card numbers. Ensures that the numbers pass the Luhn validation. Examples:

  * `377428083214575`
  * `342545797384840`
  * `371449635398431`

  
Credit Card Number-Discover |  Replaces values with random 16-digit Discover credit card numbers. Ensures that the numbers pass the Luhn validation. Examples:

  * `6011174868103745`
  * `6011006830091113`
  * `6011326843007736`

  
Credit Card Number-Mastercard |  Replaces values with random 16-digit Mastercard credit card numbers. Ensures that the numbers pass the Luhn validation. Examples:

  * `5233316245315286`
  * `5171736663215508`
  * `5479143620815877`

  
Credit Card Number-Visa |  Replaces values with random 16-digit Visa credit card numbers. Ensures that the numbers pass the Luhn validation. Examples:

  * `4929680877575125`
  * `4716403468935369`
  * `4532622699903274`

  
Date-Card Expiration |  Replaces values with random dates between 2000 and present. Day is always the last day of the month. Examples:

  * `2008-02-29`
  * `2014-08-31`
  * `2018-04-30`

  
Date-Past  |  Replaces values with random dates from 1950 through to the present date Examples:

  * 1970-01-01
  * 2001-08-05
  * 2018-10-16

  
Email Address |  Replaces values with random email addresses while preserving the number of periods, hyphens, and underscores before the address sign (`@`). Possible top-level domains are: `.com`, `.org`, `.net`,` .edu`, `.gov`, `.int`,` .us`,` .uk`,` .eu`, `.cn`,` .in`, `.ru`, `.jp`, and `.au`.  Examples:

  * `samar@example.com` could become `svkrpw@dmsoen.org`
  * `mike.williams@gmail.com` could become` sbvtud.ramzonibt@terim.net`
  * `ross_amara@gatech.edu` could become` qcipp_pnjetya@nbreqgp.gov`

  
Finland Personal Identity Code |  Replaces values with random Finland Personal Identity Codes Examples:

  * `160811A0142`
  * `251017A561N`
  * `300399-888Y`

Details: Day of Birth:

  * Generates random 2-digit numbers between 01 and 30

Month of Birth:

  * Generates random 2-digit numbers between 01 and 12

Year of Birth:

  * Generates random 2-digit numbers between 00 and 99

Century Identification Sign:

  * Randomly picks characters from +, -, or A

Individual Number:

  * Generates random 3-digit numbers between 000 and 999

Checksum Character:

  * Randomly picks characters from 0 through 9 or from A through Z, except for G, I, O, Q, and Z

Sanity Check:

  * Uses Post Processing Function to ensure validity of the generated Personal Identity Codes

  
Format Preserving Randomization |  Randomizes values while preserving their length, the position of letters and digits, the case of letters, and the special characters Examples:

  * `AjHjK123#@` could become `SbVbU574#@`
  * `678-704-7862` could become `281-272-1795`
  * `!@#$` remains `!@#$`

  
Gender |  Replaces column data with values picked randomly from the following list:

  * Male
  * Female
  * Other

  
Height (Centimeter) |  Replaces values with random numbers from 45 cm through 200 cm. Examples:

  * `60`
  * `162`
  * `176`

  
Identification Number |  Replaces values with random numbers from 1 through 999,999 Examples:

  * `166050`
  * `9887`
  * `46803`

  
IMEI Number |  Replaces values with random 15-digit IMEI numbers. Ensures that the numbers pass the Luhn validation. Examples:

  * `490154203237518`
  * `357805023984942`
  * `352066060926230`

  
Income |  Replaces values with random numbers from 30,000 through 999,999 Examples:

  * `75001`
  * `155000`
  * `700999`

  
Marital Status |  Replaces column data with values picked randomly from the list: Single, Married, Separated, Divorced, Remarried, and Widowed  
Race |  Replaces column data with values picked randomly from the list: White, African American, Asian, American Indian, Alaska Native, Native Hawaiian, and Other Pacific Islander  
Random Name |  Replaces values with random letters of random length. Compatible with character type columns only. Examples:

  * `AjHjK123#@` could become `Sbvtud`
  * `Michael` could become `Ramzoni`
  * `Richard Williams` could become `Madpalvik`

  
Religion |  Replaces column data with values picked randomly from the list: Christianity, Islam, Nonreligious, Hinduism, Buddhism, Sikhism, Jainism, Judaism, and Other  
Sexual Orientation |  Replaces column data with values picked randomly from the list: Heterosexual, Homosexual, Bisexual, and Asexual  
Stock |  Replaces values with random numbers from 100 through 9,999 Examples:

  * `1300`
  * `5499`
  * `9990`

  
UK National Insurance Number (Space-Separated) |  Replaces values with random UK National Insurance numbers, which are in AA 99 99 99 A format, where A signifies a letter and 9 a digit Examples:

  * `AA 69 94 50 A`
  * `ZR 50 16 33 A`
  * `EE 25 37 53 D`

Details: First Prefix Letter

  * Randomly picks letters from A to Z except D, F, I, Q, U, and V

Second Prefix Letter

  * Randomly picks letters from A to Z except D, F, I, Q, U, and V

6 Digits

  * Generates random 6-digit numbers

Suffix Letter

  * Randomly picks letters from A to D

Sanity Check and Formatting

  * Uses Post Processing Function to format and ensure validity of the generated National Insurance numbers

  
UK Postal Code (Space-Separated) |  Replaces values with random UK postal codes, which are in AA9A 9AA format, where A signifies a letter and 9 a digit Examples:

  * `SE1P 4SA`
  * `EC1A 1BB`
  * `SW1A 0AA`

Details: First Character:

  * Randomly picks letters from A to Z except Q, V, and X

Second Character:

  * Randomly picks letters from A to Z except I, J, and Z

Third Character:

  * Randomly picks digits from 0 to 9

Fourth Character:

  * Randomly picks letters from A, B, E, H, M, N, P, R, V, W, X, and Y 

Fifth Character:

  * Space

Sixth Character:

  * Randomly picks digits from 0 to 9

Seventh Character:

  * Randomly picks letters from A to Z except C, I, K, M, O, and V

Eighth Character:

  * Randomly picks letters from A to Z except C, I, K, M, O, and V

  
URL |  Replaces values with random URLs starting with `http` or `https`. Possible top-level domains are: `.com`, `.org`, `.net`, `.edu`, `.gov`, `.int`, `.us`, `.uk`, `.eu`,` .cn`, `.in`,` .ru`,` .jp`, and `.au`.  Examples:

  * `https://www.hapiden.com`
  * `http://www.qazwsx937.gov`
  * `https://www.bhatag.in`

  
US Phone Number |  Replaces values with random 10-digit US phone numbers Examples:

  * `6787047862`
  * `2025550149`
  * `5206625256`

Details: Area Code:

  * Randomly picks 3-digit codes from 328 US area codes

Remaining 7 Digits:

  * Generates random 7-digit numbers

Sanity Check:

  * Uses Post Processing Function to ensure validity of the generated phone numbers

  
US Phone Number (With Country Code) |  Replaces values with random US phone numbers, which are in +1 (999) 999-9999 format, where 9 signifies a digit Examples:

  * `+1 (678) 704-7862`
  * `+1 (202) 555-0149`
  * `+1 (520) 662-5256`

Details: Country Code:

  * +1

Area Code:

  * Randomly picks 3-digit codes from 328 US area codes

Remaining 7 Digits:

  * Generates random 7-digit numbers

Sanity Check and Formatting:

  * Uses Post Processing Function to format and ensure validity of the generated phone numbers

  
US Social Security Number |  Replaces values with random US Social Security numbers Examples:

  * `148923857`
  * `771182740`
  * `562998392`

  
US Social Security Number (Hyphenated) |  Replaces values with random US Social Security numbers, which are in `999-99-9999` format, where `9` signifies a digit  Examples:

  * `148-92-3857`
  * `771-18-2740`
  * `562-99-8392`

  
Weight (Pound) |  Replaces values with random numbers from 5 through 250. The range covers weight in pounds.  Examples:

  * `45`
  * `176`
  * `210`
