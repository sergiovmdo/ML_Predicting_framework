import pandas as pd


class Encoder():
    """
    Class in charge of encoding categorical variables into a format that will be readable for the ML algorithm.
    """

    def encode(self, dataframe, technique):
        """
        Invokes the encoding technique that was set via parameters.

        Returns:
            The encoded dataframe.
        """
        if technique == 'one_hot':
            return self.one_hot_encoder(dataframe)

        else:
            return dataframe

    def one_hot_encoder(self, dataframe):
        """
        Implementation of one hot encoding.

        Returns:
            The encoded dataframe.
        """
        encoded_df = pd.DataFrame()
        for c in dataframe.columns:
            encoded_cols = pd.get_dummies(dataframe[c], prefix=c, drop_first=True)
            encoded_cols = encoded_cols.astype(int)  # Convert boolean columns to integers
            encoded_df = pd.concat([encoded_df, encoded_cols], axis=1)

        return encoded_df
