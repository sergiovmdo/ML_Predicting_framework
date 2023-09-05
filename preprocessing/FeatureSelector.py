class FeatureSelector:
    """
    After performing all the operations over variables and creating new ones, this module will sellect those variables
    that will be more useful to our predictive model.
    """

    def select_features(self, dataframe, target, technique, num_features):
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
        if technique == 'correlation':
            return self.correlation(dataframe, target, num_features)

        else:
            return dataframe

    def correlation(self, dataframe, target, num_features):
        """
        Implementation of correlation feature selection technique

        Args:
            dataframe (dataframe): Data.
            target (string): Name of the target variable.
            num_features (int): Number of features to be selected.
        """
        correlations = dataframe.corr()[target].abs().sort_values(ascending=False)
        selected_features = correlations[1:num_features + 1].index.to_list()
        selected_features = selected_features + [target]

        return dataframe[selected_features]
