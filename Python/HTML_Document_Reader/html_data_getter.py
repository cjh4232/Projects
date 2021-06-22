from html_doc_reader import HTML_Doc_Reader

root_dir = "\\Documents\Data\\"

data = HTML_Doc_Reader(root_dir)

df = data.get_data()

df.to_excel("Consolidated_Data.xlsx")
print("Script Completed")