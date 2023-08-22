class FeatureSelector:
    """
    After performing all the operations over variables and creating new ones, this module will sellect those variables
    that will be more useful to our predictive model.
    """

    def __init__(self):
        pass

    def select_features(self, dataframe, target, technique, num_features):
        if technique == 'correlation':
            return self.correlation(dataframe, target, num_features)

        else:
            return dataframe

    def correlation(self, dataframe, target, num_features):
        correlations = dataframe.corr()[target].abs().sort_values(ascending=False)
        selected_features = correlations[1:num_features + 1].index.to_list()

        return dataframe[selected_features]
