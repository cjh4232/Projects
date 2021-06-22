## Overview
The company I work for had a significant amount of data stored in individual html files that required some form of consolidation based on the assembly serial number and formatted
in a useful way for analysis. The information for each file was broken down into four features:

- Serial Number
- Location
  - Bottom Left (BL)
  - Bottom Right (BR)
  - Center (C)
  - Top Left (TL)
  - Top Right (TR)
- Saggital Measurement
- Tangential Measurement

I created a class, instantiated with the root directory of the data, with built-in functions to obtain the data given a file name within the root directory.

The get_data function uses all the built-in functions to build and return a pandas DataFrame of all the data gathered.

Once the DataFrame has been returned the user can perform their analysis.

## Details
The file names were saved with a standard nomenclauture and that formatting was used to obtain some of the data.

- Example
  - 4_RMA_BL.html
- Serial Number = 4
- Location = BL

Since the documents were not consistent with each other I couldn't simply read the document and assign a specific index for the data. Instead I used the Natural Language
Tool Kit (NLTK) to find matches to the frequency of interest (in my case 200 lp/mm) and obtain the corresponding measurement values. I saved the data into an Excel
spreadsheet where I was able to create a pivot table and make determinations on which assemblies needed our attention and which could move through the rest of our
process.
