import pandas as pd
import Imputer, OutlierRemover


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
        self.outlier_remover = OutlierRemover()

    def clean_data(self):
        if 'imputation_technique' in self.parameters:
            self.impute(self.parameters['imputation_technique'])

    def remove_duplicates(self):
        self.dataframe = self.dataframe.drop_duplicates()

    def get_dataframe(self):
        return self.dataframe

    def impute(self, imputation_technique):
        return self.imputer.impute(self.dataframe, imputation_technique)

    def remove_outliers(self, removal_technique):
        return self.outlier_remover.remove_outliers(self.dataframe, removal_technique)
