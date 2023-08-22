import pandas as pd
import DataCleaner
from Preprocessing.DataTransformer import DataTransformer
from Preprocessing.FeatureSelector import FeatureSelector


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
        # Data cleaning
        data_cleaner = DataCleaner(self.numerical_data, self.parameters)
        self.numerical_data = DataCleaner.clean_data()

        # Data transforming
        data_transformer = DataTransformer(self.numerical_data, self.categorical_data, self.parameters)
        self.numerical_data, self.categorical_data = DataTransformer.transform_data()

        ### FEATURE EXTRACTION ###

        #TODO

        ### FEATURE COMBINATION ###

        #TODO

        # Concatenate both dataframes
        self.dataframe = pd.concat([self.numerical_data, self.categorical_data], ignore_index=True)

        # Feature selection

        if 'feature_selector' in self.parameters:
            feature_selector = FeatureSelector()
            self.dataframe = feature_selector.select_features(self.dataframe, self.parameters['target'], self.parameters['feature_selector'],
                                                              self.parameters['num_features'])





print('hello')