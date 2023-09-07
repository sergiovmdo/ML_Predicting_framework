from preprocessing.Encoder import Encoder
from preprocessing.Scaler import Scaler


class DataTransformer:
    """
    Transforming the data is the second step of the pipeline, this class will deal with all the processes related to it
    including scaling and encoding. Each of this processes has its respective subclass that will inheritate from
    this one.
    """

    def __init__(self, numerical_dataframe, categorical_dataframe, parameters):
        """
        Initialize a new instance of DataTransformer

        Args:
            numerical_dataframe (dataframe): part of the data that is only numerical.
            categorical_dataframe (dataframe): part of the data that is only categorical.
            parameters (dictionary): parameters dictionary that contain all the information related to the process.

        """
        self.numerical_dataframe = numerical_dataframe
        self.categorical_dataframe = categorical_dataframe

        self.parameters = parameters

        self.scaler = Scaler()
        self.encoder = Encoder()

    def transform_data(self):
        """
        This is the main function of the class that will deal with all the transformations.

        Returns:
            Both the numerical and categorical dataframes but already transformed.
        """

        if 'scaler' in self.parameters:
            self.numerical_dataframe = self.scale(self.parameters['scaler'])

        if 'encoder' in self.parameters:
            self.categorical_dataframe = self.encode(self.parameters['encoder'])

        return self.numerical_dataframe, self.categorical_dataframe

    def scale(self, scaling_technique):
        """
        Auxiliary method used for invoking the scaler.

        Returns:
            The scaled dataframe.
        """
        return self.scaler.scale(self.numerical_dataframe, scaling_technique)

    def encode(self, encoding_technique):
        """
        Auxiliary method used for invoking the encoder.

        Returns:
            The encoded dataframe.
        """
        return self.encoder.encode(self.categorical_dataframe, encoding_technique, self.parameters['target'])
