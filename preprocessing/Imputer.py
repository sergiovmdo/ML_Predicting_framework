from sklearn.impute import SimpleImputer
import pandas as pd


class Imputer():
    """
    Class that is in charge of imputing null values existing in our original data.
    """

    def impute(self, dataframe, parameters):
        """
        Main method that is in charge of invoke the appropiate imputing technique.

        Args:
            dataframe (dataframe): Data.
            technique (string): Imputation technique to be used.

        Returns:
            The imputed dataframe.
        """
        if parameters['imputer'] == 'simple_imputer':
            return self.simple_imputer(dataframe, parameters['target'])

        else:
            return dataframe

    def simple_imputer(self, dataframe, target):
        """
        Implementation of mean imputation for non-target columns
        and mode (most frequent) imputation for the target column.

        Args:
            dataframe (dataframe): Data.
            target (str): The target column name.

        Returns:
            pd.DataFrame: The imputed dataframe.
        """
        # Separate target and non-target columns
        non_target_cols = dataframe.columns.difference([target])

        # Impute non-target columns with mean
        mean_imputer = SimpleImputer(strategy='mean')
        dataframe[non_target_cols] = mean_imputer.fit_transform(dataframe[non_target_cols])

        # Impute target column with the most frequent value
        mode_imputer = SimpleImputer(strategy='most_frequent')
        dataframe[[target]] = mode_imputer.fit_transform(dataframe[[target]])

        return dataframe
