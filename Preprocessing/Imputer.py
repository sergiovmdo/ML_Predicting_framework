from sklearn.impute import SimpleImputer
import pandas as pd


class Imputer():
    """
    Subclass of DataCleaner that is in charge of imputing null values existing in our original data.
    """
    def __init__(self):
        pass

    def impute(self, dataframe, technique):
        if technique == 'simple_imputer':
            return self.simple_imputer()

        else:
            return dataframe

    def simple_imputer(self, dataframe):
        imputer = SimpleImputer(strategy='mean')
        imputed_df = imputer.fit_transform(dataframe)

        return pd.DataFrame(imputed_df, columns=dataframe.columns)
