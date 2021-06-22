from html_doc_reader import HTML_Doc_Reader

root_dir = "\\Documents\Data\\" # Root directory - all data stored here
freq = "200" # Frequency of interest - 200 lp/mm

# Instantiate HTML_Doc_Reader object
data = HTML_Doc_Reader(root_dir)

# Get the pandas DataFrame of the data at the frequency of interest
df = data.get_data(freq)

# Save to an Excel spreadsheet
df.to_excel("Consolidated_Data.xlsx")
print("Script Completed")