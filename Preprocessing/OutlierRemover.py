import DataCleaner
from sklearn.ensemble import IsolationForest


class OutlierRemover(DataCleaner):
    """
    Subclass of DataCleaner that is in charge of detecting and removing the possible outliers that may exist in our
    original data.
    """
    def __init__(self):
        pass

    def remove_outliers(self, dataframe, technique):
        if technique == 'isolation_forest':
            return self.isolation_forest(dataframe)


    def isolation_forest(self, dataframe):
        isolation_forest = IsolationForest(contamination='auto')
        # Fit the IsolationForest to your data
        isolation_forest.fit(dataframe)

        # Outliers will have a value of -1 in this column
        dataframe['Outlier'] = isolation_forest.predict(dataframe)

        return dataframe
