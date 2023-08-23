import pandas as pd


class Encoder():
    """
    Subclass of DataTransformer that is in charge of encoding categorical variables into a format that will be
    readable for the ML algorithm.
    """

    def __init__(self):
        pass

    def encode(self, dataframe, technique):
        if technique == 'one_hot':
            return self.one_hot_encoder(dataframe)

        else:
            return dataframe

    def one_hot_encoder(self, dataframe):
        encoded_df = pd.DataFrame()
        for c in dataframe.columns:
            encoded_cols = pd.get_dummies(dataframe[c], prefix=c, drop_first=True)
            encoded_cols = encoded_cols.astype(int)  # Convert boolean columns to integers
            encoded_df = pd.concat([encoded_df, encoded_cols], axis=1)

        return encoded_df
