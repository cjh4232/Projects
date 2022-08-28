from html_doc_reader import HTMLDocReader

ROOT_DIRECTORY = "\\Documents\Data\\" # Root directory - all data stored here
FREQUENCY = "200" # Frequency of interest - 200 lp/mm

# Instantiate HTMLDocReader object
data = HTMLDocReader(ROOT_DIRECTORY)

# Get the pandas DataFrame of the data at the frequency of interest
df = data.get_data(FREQUENCY)

# Save to an Excel spreadsheet
df.to_excel("Consolidated_Data.xlsx")
print ("Script Completed")
