## Overview
There was a significant amount of data stored in individual html files that required some form of consolidation based on the assembly serial number and formatted
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

Created a class, instantiated with the root directory of the data, with built-in functions to pull the data out of a given file within the root directory.

The get_data function uses all the built-in functions to construct and return a pandas DataFrame of all the data gathered.

Once the DataFrame has been returned, the user can perform their analysis within the Python environment or output into a file for use in other software.

## Details
The file names were saved with a standard nomenclauture and that formatting was used to obtain some of the data.

- Example
  - 4_RMA_BL.html
- Serial Number = 4
- Location = BL

Since the documents were not consistent with each other, it wasn't possible to simply read the document and assign a specific index for the data. Instead, the Natural Language
Tool Kit (NLTK) was implemented to find matches to the frequency of interest (in this case 200 lp/mm) and obtain the corresponding measurement values. The data was output into an Excel spreadsheet where a pivot table was created to make determinations on which assemblies needed attention and which could move through the rest of the process.
