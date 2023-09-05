import DataCleaner
from sklearn.ensemble import IsolationForest


class OutlierRemover:
    """
    Class that is in charge of detecting and removing the possible outliers that may exist in our
    original data.
    """

    def remove_outliers(self, dataframe, technique):
        """
        Method that is in charge of invoking the appropriate outlier removal technique.

        Args:
            dataframe (dataframe): Data.
            technique (string): outlier removal technique.

        Returns:
            the dataframe with all the outliers removed.
        """
        if technique == 'isolation_forest':
            return self.isolation_forest(dataframe)
        else:
            return dataframe

    def isolation_forest(self, dataframe):
        """
        Implementation of the isolation forest technique.

        Args:
            dataframe (dataframe): Data.

        Returns:
            the dataframe with all the outliers removed.
        """
        isolation_forest = IsolationForest(contamination='auto')
        # Fit the IsolationForest to your data
        isolation_forest.fit(dataframe)

        # Outliers will have a value of -1 in this column
        dataframe['Outlier'] = isolation_forest.predict(dataframe)

        return dataframe
