import pandas as pd
from category_encoders import TargetEncoder


class Encoder():
    """
    Class in charge of encoding categorical variables into a format that will be readable for the ML algorithm.
    """

    def __init__(self, categorical_df, parameters):
        self.categorical_df = categorical_df
        self.parameters = parameters


    def encode(self):
        """
        Invokes the encoding technique that was set via parameters.

        Returns:
            The encoded dataframe.
        """
        if self.parameters['encoder'] == 'one_hot':
            return self.one_hot_encoder()
        elif self.parameters['encoder'] == 'target_encoding':
            return self.target_encoder(self.categorical_df, self.parameters['target'])

        else:
            return self.categorical_df

    def one_hot_encoder(self):
        """
        Implementation of one hot encoding.

        Args:
            dataframe (dataframe): Data.

        Returns:
            The encoded dataframe.
        """
        encoded_df = pd.DataFrame()
        for c in self.categorical_df.columns:
            encoded_cols = pd.get_dummies(self.categorical_df[c], prefix=c, drop_first=True)
            encoded_cols = encoded_cols.astype(int)  # Convert boolean columns to integers
            if c in self.parameters['target']:
                encoded_cols = encoded_cols.rename(columns={encoded_cols.columns[0]: self.parameters['target']})
            encoded_df = pd.concat([encoded_df, encoded_cols], axis=1)

        return encoded_df

    def target_encoder(self, dataframe, target):
        """
        Implementation of target encoding.

        Args:
            dataframe (dataframe): Data.
            target (string): name of the target variable.

        Returns:
            The encoded dataframe.
        """
        dataframe[target] = pd.get_dummies(dataframe[target], prefix=target, drop_first=True).astype(int)

        encoder = TargetEncoder(handle_missing='return_nan')
        encoder = encoder.fit(dataframe, dataframe[target])

        dataframe = encoder.transform(dataframe)

        return dataframe

