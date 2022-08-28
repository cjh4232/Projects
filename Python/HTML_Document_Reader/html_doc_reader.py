from typing import List
import os
import pandas as pd
import nltk
#nltk.download()

class HTMLDocReader():

    """
    This code creates a custom class that allows users to convert complicated output
    documents (html) from the Trioptics ImageMaster, which are frankly garbage, into
    a usable form for data analysis.

    The class is instantiated using the directory in which the data is being stored.

    The get_data() function is called on the new object and a Pandas DataFrame is returned

    In this version the data of interest was both the sagittal & tangential  MTF
    values @ 200 lp/mm
    """

    def __init__(self, root_dir) -> None:
        self.root_dir = root_dir

    def get_list(self) -> List:

        """
        This function acquires a list of all the files in the root directory with the file type
        "html" and returns that list
        """
        # Create a full list of files & directories in root directory
        full_list = os.listdir(self.root_dir)

        # Create empty list to store upcoming html file names
        html_list = []

        # Check the last 4 indexes for the "html" file type
        for file in full_list:
            if file[-4:] == "html":
                html_list.append(file) # Add them to the list

        return html_list

    def __get_sn(self, file_name) -> str:
        """
        This function assumes the html file was saved with a specific naming convention
        and the SN is the first set of numbers in the filename and returns that SN
        """
        if file_name[1] == "_":
            return file_name[0]

        return file_name[0:2]

    def __get_loc(self, file_name) -> str:
        """
        This function assumes the html file was saved with a specific naming convention
        and the location is within the filename
        """
        if file_name[-6] == "C":
            return file_name[-6]
        return file_name[-7:-5]

    def __read_html_nltk(self, file_name, freq) -> List:
        """
        This function uses the Natural Language Tool Kit to parse the html document, find the lp/mm
        value of interest (in this case, 200) and then retrieve the data points associated with that
        value of interest by going into the nested lists of document information and return a list
        """

        # Opens the file using the root directory provided in the class instaiation & file name
        file = open(self.root_dir + file_name, 'r', encoding='utf-8')

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

    def __get_tan(self, file_name, freq) -> float:
        """
        This function uses the __read_html_nltk fucntion to get the two datapoints out
        of the document and returns the tangential value
        """
        data = self.__read_html_nltk(file_name, freq)
        return data[1]

    def __get_sag(self, file_name, freq) -> float:
        """
        This function uses the __read_html_nltk fucntion to get the two datapoints out
        of the document and returns the saggital value
        """
        data = self.__read_html_nltk(file_name, freq)
        return data[0]

    def get_data(self, freq) -> pd.DataFrame:
        """
        This function uses all of the built-in class functions to build multiple lists of
        data and returns them zipped together into a Pandas DataFrame.

        This is the only function that should be called using the HTML_Document_Reader class
        and takes one input - the frequency of interest.
        """
        # Create empty lists to store upcoming data
        serial_numbers = []
        locations = []
        saggital_values = []
        tangential_values = []

        # Create a list of column names - the order matters
        column_names = ["SN", "Location", "Sagittal", "Tangential"]

        # Creates list of files in root directory of html file type
        file_list = self.get_list()

        # Iterate through each file and add all of the information from the file to
        # the corresponding list using the class functions
        for file in file_list:
            serial_numbers.append(self.__get_sn(file))
            locations.append(self.__get_loc(file))
            saggital_values.append(self.__get_sag(file, freq))
            tangential_values.append(self.__get_tan(file, freq))

        # Create a pandas DataFrame from the zipped lists and column names
        dataframe = pd.DataFrame(zip(serial_numbers, locations, saggital_values,
                                    tangential_values), columns=column_names)

        return dataframe
