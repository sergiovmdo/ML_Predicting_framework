import pandas as pd
import DataCleaner

class PreprocessingPipeline:
    """
    This is the pipeline that will be in charge of performing all the preprocessing steps.
    """
    def __init__(self, dataframe, parameters):
        self.categorical_data = None
        self.numerical_data = None
        self.dataframe = dataframe
        self.parameters = parameters

        # We want to have two separate dataframes according to its data type so we can perform different operations on them
        self.split_by_data_type()


    def split_by_data_type(self):
        categorical_variables = self.dataframe.select_dtypes(include=['object', 'category']).columns.to_list()
        numerical_variables = list(filter(lambda x: x not in categorical_variables, self.dataframe.columns))

        self.numerical_data = self.dataframe[numerical_variables]
        self.categorical_data = self.dataframe[categorical_variables]


    def run(self):
        data_cleaner = DataCleaner(self.dataframe)