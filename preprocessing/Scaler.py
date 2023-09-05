import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Scaler:
    """
    Subclass of DataTransformer that is in charge of scaling the original data.
    """

    def __init__(self):
        """
        Initialize a new instance of

        Args:

        """
        pass

    def scale(self, dataframe, technique):
        if technique == 'min_max':
            return self.min_max_scaler(dataframe)

        else:
            return dataframe

    def min_max_scaler(self, dataframe):
        scaler = MinMaxScaler()
        scaled_df = scaler.fit_transform(dataframe)

        return pd.DataFrame(scaled_df, columns=dataframe.columns)
