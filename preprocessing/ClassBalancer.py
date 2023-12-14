from imblearn.over_sampling import SMOTE, RandomOverSampler


class ClassBalancer:
    """
    Balances the data according to the target variable that we are studying, different methods will be implemented
    including oversampling and undersampling techniques.
    """

    def __init__(self, dataframe, target, technique, seed):
        """
        Initialization of ClassBalancer class

        Args:
            dataframe (dataframe): Data
            target (string): target variable name.
            technique (string): technique to be used in the balancing procedure.
            seed (int): seed to be used when balancing classes.
        """
        self.dataframe = dataframe
        self.target = target
        self.technique = technique
        self.seed = seed

    def  balance_classes(self):
        """
        Depending on the balancing technique calls the implementation of it.

        Returns:
            Returns the balanced dataframe.
        """
        if self.technique == 'smote':
            return self.transform(SMOTE(random_state=self.seed))
        elif self.technique == 'random_oversampling':
            return self.transform(RandomOverSampler(random_state=self.seed))

        else:
            return self.dataframe

    def transform(self, balancer):
        X_resampled, y_resampled = balancer.fit_resample(self.dataframe, self.target)

        dataframe_resampled = X_resampled.copy()

        return dataframe_resampled, y_resampled
