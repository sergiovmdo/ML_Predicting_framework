class FeatureSelector:
    """
    After performing all the operations over variables and creating new ones, this module will sellect those variables
    that will be more useful to our predictive model.
    """

    def __init__(self, dataframe, target, technique, num_features=None):
        self.dataframe = dataframe
        self.target = target
        self.technique = technique
        self.num_features = num_features

    def select_features(self):
        """
        Invokes the appropriate feature selection method.

        Args:
            dataframe (dataframe): Data.
            target (string): Name of the target variable.
            technique (string): Technique to be used in the feature selection step.
            num_features (int): Number of features to be selected (if needed).

        Returns:
            The dataframe with only the selected variables.
        """
        if self.technique == 'correlation':
            return self.correlation()

        else:
            return self.dataframe

    def correlation(self):
        """
        Implementation of correlation feature selection technique

        Args:
            dataframe (dataframe): Data.
            target (string): Name of the target variable.
            num_features (int): Number of features to be selected.
        """
        correlations = self.dataframe.corr()[self.target].abs().sort_values(ascending=False)
        selected_features = correlations[1:self.num_features + 1].index.to_list()
        selected_features = selected_features + [self.target]

        return self.dataframe[selected_features]
