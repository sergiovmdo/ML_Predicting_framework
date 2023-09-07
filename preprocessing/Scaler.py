import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Scaler:
    """
    Class that is in charge of scaling the original data.
    """
    def scale(self, dataframe, technique):
        """
        Scales the dataframe using a particular technique.

        Args:
            dataframe (dataframe): Data.
            technique (string): The technique used to scale the dataframe.

        Returns:
            The scaled dataframe.
        """
        if technique == 'min_max':
            return self.transform(MinMaxScaler(), dataframe)
        elif technique == 'z_score':
            return self.transform(StandardScaler(), dataframe)

        else:
            return dataframe

    def transform(self, scaler, dataframe):
        scaled_dataframe = scaler.fit_transform(dataframe)

        return pd.DataFrame(scaled_dataframe, columns=dataframe.columns)
