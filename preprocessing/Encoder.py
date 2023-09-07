import pandas as pd


class Encoder():
    """
    Class in charge of encoding categorical variables into a format that will be readable for the ML algorithm.
    """

    def encode(self, dataframe, technique, target):
        """
        Invokes the encoding technique that was set via parameters.

        Returns:
            The encoded dataframe.
        """
        if technique == 'one_hot':
            return self.one_hot_encoder(dataframe)
        elif technique == 'target_encoding':
            return self.target_encoder(dataframe, target)

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

    def target_encoder(self, dataframe, target):
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

