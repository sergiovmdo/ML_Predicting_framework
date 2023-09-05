from imblearn.over_sampling import SMOTE


class ClassBalancer:
    """
    Balances the data according to the target variable that we are studying, different methods will be implemented
    including oversampling and undersampling techniques.
    """

    def balance_classes(self, dataframe, target, technique):
        """
        Depending on the balancing technique calls the implementation of it.

        Args:
            dataframe (dataframe): Data
            target (string): target variable name.
            technique (string): technique to be used in the balancing procedure.

        Returns:
            Returns the balanced dataframe.
        """
        if technique == 'smote':
            return self.smote(dataframe, target)

        else:
            return dataframe

    def smote(self, dataframe, target):
        """
        Synthetic Minority Oversampling Technique implementation.

        Args:
            dataframe (dataframe): Data.
            target (string) target variable name.

        Returns:
            Returns the balanced dataframe.
        """
        oversampler = SMOTE()

        # Apply SMOTE oversampling to the dataset
        X_resampled, y_resampled = oversampler.fit_resample(dataframe.drop(target, axis=1), dataframe[target])

        dataframe_resampled = X_resampled.copy()
        dataframe_resampled[target] = y_resampled

        return dataframe_resampled
