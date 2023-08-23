import pandas as pd

from preprocessing.Imputer import Imputer


class DataCleaner:
    """
    Parent class that will deal with the first step of the pipeline. Standardization of columns, duplicate checking,
    data formatting, imputation, and outlier removal are performed in this step.

    There are two classes that inheritate from this class, Imputer and OutlierRemover that will be in charge of this
    particular tasks of the pipeline.
    """

    def __init__(self, dataframe, parameters):
        # We attempt to convert columns to its proper type
        self.dataframe = dataframe.infer_objects()
        self.parameters = parameters
        self.remove_duplicates()

        self.imputer = Imputer()


    """
    This is the main function of the class that will perform the data cleaning
    """
    def clean_data(self):
        if 'imputer' in self.parameters:
            self.dataframe = self.impute(self.parameters['imputer'])

        return self.dataframe

    def remove_duplicates(self):
        self.dataframe = self.dataframe.drop_duplicates()

    def get_dataframe(self):
        return self.dataframe

    def impute(self, imputation_technique):
        return self.imputer.impute(self.dataframe, imputation_technique)
