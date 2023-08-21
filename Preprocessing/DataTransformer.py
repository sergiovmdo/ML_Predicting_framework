from Preprocessing.Encoder import Encoder
from Preprocessing.Scaler import Scaler


class DataTransformer:
    """
    Transforming the data is the second step of the pipeline, this class will deal with all the processes related to it
    including scaling and encoding. Each of this processes has its respective subclass that will inheritate from
    this one.
    """
    def __init__(self, numerical_dataframe, categorical_dataframe, parameters):
        self.numerical_dataframe = numerical_dataframe
        self.categorical_dataframe = categorical_dataframe

        self.parameters = parameters

        self.scaler = Scaler()
        self.encoder = Encoder()


    """
    This is the main function of the class that will deal with all the transformation to be performed
    """
    def transform_data(self):
        if 'scaler' in self.parameters:
            self.numerical_dataframe = self.scale(self.parameters['scaler'])

        if 'encoder' in self.parameters:
            self.categorical_dataframe = self.encode(self.parameters['encoder'])

        return self.numerical_dataframe, self.categorical_dataframe

    def scale(self, scaling_technique):
        return self.scaler.scale(self.dataframe, scaling_technique)

    def encode(self, encoding_technique):
        return self.encoder.encode(self.dataframe, encoding_technique)