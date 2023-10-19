import pandas as pd


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
            return self.target_encoder()

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

        # Initialize a dictionary to store the target encodings
        target_encodings = {}

        # Store the names of categorical variables
        categorical_columns = dataframe.columns
        categorical_columns.remove(target)

        # Loop through each categorical column and calculate target encodings
        for c in dataframe:
            target_encoding = dataframe.groupby(c)[target].mean().to_dict()
            dataframe[c + '_encoded'] = dataframe[c].map(target_encoding)
            target_encodings[c] = target_encoding

        # Drop the original categorical columns if needed
        dataframe.drop(categorical_columns, axis=1, inplace=True)

        return dataframe

