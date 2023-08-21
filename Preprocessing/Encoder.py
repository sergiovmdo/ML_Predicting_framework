import DataTransformer
import pandas as pd


class Encoder(DataTransformer):
    """
    Subclass of DataTransformer that is in charge of encoding categorical variables into a format that will be
    readable for the ML algorithm.
    """

    def __init__(self):
        pass

    def encode(self, dataframe, technique):
        if technique == 'one_hot':
            return self.one_hot_encoder()

        else:
            return dataframe

    def one_hot_encoder(self, dataframe):
        encoded_df = pd.DataFrame()
        for c in dataframe:
            encoded_df[c] = pd.get_dummies(dataframe[c])

        return encoded_df
