from sklearn.impute import SimpleImputer
import pandas as pd


class Imputer():
    """
    Class that is in charge of imputing null values existing in our original data.
    """

    def impute(self, dataframe, technique):
        """
        Main method that is in charge of invoke the appropiate imputing technique.

        Args:
            dataframe (dataframe): Data.
            technique (string): Imputation technique to be used.

        Returns:
            The imputed dataframe.
        """
        if technique == 'simple_imputer':
            return self.simple_imputer(dataframe)

        else:
            return dataframe

    def simple_imputer(self, dataframe):
        """
        Implementation of mean imputation

        Args:
            dataframe (dataframe): Data.

        Returns:
            The imputed dataframe.
        """
        imputer = SimpleImputer(strategy='mean')
        imputed_df = imputer.fit_transform(dataframe)

        return pd.DataFrame(imputed_df, columns=dataframe.columns)
