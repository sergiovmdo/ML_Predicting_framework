from model.models.Model import Model
from sklearn.ensemble import RandomForestClassifier


class RandomForest(Model):

    param_grid = {
        'n_estimators': [100],  # Number of trees in the forest
        'criterion': ['gini'],  # Splitting criterion (impurity measure)
        'max_depth': [None],  # Maximum depth of individual trees
        'min_samples_split': [2],  # Minimum number of samples required to split a node
        'min_samples_leaf': [2],  # Minimum number of samples required to be at a leaf node
        'max_features': ["sqrt"],  # Number of features to consider when splitting
        'bootstrap': [True],  # Whether to bootstrap samples when building trees
        'oob_score': [True],  # Whether to use out-of-bag samples to estimate generalization error
    }

    def __init__(self, X, y, seed):
        """
        Initialize a new instance of LogisticRegression which is a subclass of the Model class which is also
        instantiated inside this constructor.

        Args:
            X (dataframe): Dataframe containing the training information for the model.
            y (array): Array containing the training target variable.
            seed (int): Seed to be used in the LogisticRegression

        """
        Model.__init__(self, X, y, RandomForestClassifier(random_state=seed), self.param_grid)

    def train(self, enable_parameter_search=False):
        """
        Used for training the model, it just calls to the method in the superclass.
        """
        return super().train(enable_parameter_search)