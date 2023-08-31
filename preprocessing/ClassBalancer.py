from imblearn.over_sampling import SMOTE


class ClassBalancer:
    """
    Balances the data according to the target variable that we are studying, different methods will be implemented
    including oversampling and undersampling techniques.
    """

    def __init__(self):
        pass

    def balance_classes(self, dataframe, target, technique):
        if technique == 'smote':
            return self.smote(dataframe, target)

        else:
            return dataframe

    def smote(self, dataframe, target):
        oversampler = SMOTE()

        # Apply SMOTE oversampling to the dataset
        X_resampled, y_resampled = oversampler.fit_resample(dataframe.drop(target, axis=1), dataframe[target])

        dataframe_resampled = X_resampled.copy()
        dataframe_resampled[target] = y_resampled

        return dataframe_resampled
