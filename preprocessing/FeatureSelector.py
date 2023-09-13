from sklearn.feature_selection import SelectKBest, mutual_info_classif

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
        elif self.technique == 'mutual_information':
            return self.mutual_information()

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

    def mutual_information(self):
        """
        Select the top 'n' features using Mutual Information.

        Returns:
            dataframe (dataframe): The feature matrix with selected features.
        """
        # Initialize the SelectKBest selector with mutual information scoring
        selector = SelectKBest(score_func=mutual_info_classif, k=self.num_features)

        return selector.fit_transform(self.dataframe, self.target)
