from typing import List
import os
import pandas as pd
import nltk
#nltk.download()

class HTML_Doc_Reader():
    
    # This class was created to load data acquired from a Trioptics
    # ImageMaster and output into an html document
    # The data of interest was both the sagittal & tangential MTF values 
    # at 200 lp/mm
 
    # The class requires a root directory where the data is stored to be
    # instantiated
    def __init__(self, root_dir) -> None:
        self.root_dir = root_dir
    
    # The get_list function acquires a list of all the files in the root
    # directory with the file type of "html" and returns that list
    def get_list(self) -> List:
        
        # Begin with full list of files & directories in root
        full_list = os.listdir(self.root_dir)
        
        # Create empty list to store upcoming html file names
        html_list = []
        
        # Check the last 4 indexes for the "html" file type
        for file in full_list:
            if file[-4:] == "html":
                html_list.append(file) # Add them to the list
        
        return html_list

    # The get_SN function assumes the html file was saved with a specific naming
    # nomenclature and the SN is the first set of numbers in the filename
    def get_SN(self, fileName) -> str:
        if fileName[1] == "_":
            return fileName[0]
        else:
            return fileName[0:2]

    # The get_loc function assumes the html file was saved with a standards naming
    # nomenclature and the location is within the file name
    def get_loc(self, fileName) -> str:
        if fileName[-6] == "C":
            return fileName[-6]
        else:
            return fileName[-7:-5]
    
    # The read_html_NLTK function uses the Natural Language Tool Kit to parse the html document,
    # find the lp/mm value of interest (in this case, 200) and then retrieve the data points
    # associated with that value of interest by going into the nested lists of document information
    def read_html_NLTK(self, fileName, freq):
        
        # Opens the file using the root directory provided in the class instaiation & file name
        file = open(self.root_dir + fileName, 'r')
        
        # Reads file in variable
        read_file = file.read()
        
        # Creates a text object to store the document
        text = nltk.Text(nltk.word_tokenize(read_file))
        
        # Obtain a list of matches to the value of interest
        match = text.concordance_list(freq)
        
        # Create an empty list to store upcoming data
        data = []
        
        # Iterate through each matching search item and obtain the corresponding 
        # measurement value within the nested lists and convert to a float
        for line in match:
            data.append(float(line[2][14]))
        
        return data
    
    # The get_tan function uses the read_html_NLTK function to get the two datapoints
    # out of the document and returns the tangential value
    def get_tan(self, fileName, freq) -> float:
        data = self.read_html_NLTK(fileName, freq)
        return data[1]

    # The get_sag function uses the read_html_NLTK function to get the two datapoints
    # out of the document and returns the saggital value
    def get_sag(self, fileName, freq) -> float:
        data = self.read_html_NLTK(fileName, freq)
        return data[0]
    
    # The get_data function makes use of all of the built-in class functions to build
    # mulitple lists of data and returns them zipped together into a pandas DataFrame
    def get_data(self, freq):
        
        # Create empty lists to store upcoming data
        SN = []
        LOC = []
        SAG = []
        TAN = []
        
        # Create a list of column names - the order matters
        COL_NAMES = ["SN", "Location", "Sagittal", "Tangential"]
        
        # Creates list of files in root directory of html file type
        file_list = self.get_list()

        # Iterate through each file and add all of the information from the file to 
        # the corresponding list using the class functions
        for file in file_list:
            SN.append(self.get_SN(file))
            LOC.append(self.get_loc(file))
            SAG.append(self.get_sag(file, freq))
            TAN.append(self.get_tan(file, freq))
        
        # Create a pandas DataFrame from the zipped lists and column names
        df = pd.DataFrame(zip(SN, LOC, SAG, TAN), columns=COL_NAMES)
        
        return df