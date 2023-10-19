from model.models.Model import Model
import xgboost as xgb


class XGBoost(Model):

    param_grid = {
        'n_estimators': [10, 50, 100],  # Number of boosting rounds (trees)
        'learning_rate': [0.15],  # Step size shrinkage used in each boosting round
        'max_depth': [4],  # Maximum depth of individual trees
        'min_child_weight': [5],  # Minimum sum of instance weight (hessian) needed in a child
        'subsample': [0.9],  # Fraction of samples used for fitting the trees
        'colsample_bytree': [0.9],  # Fraction of features used for building each tree
        'gamma': [0.1],  # Minimum loss reduction required to make a further partition on a leaf node
        'alpha': [1e-3],  # L1 regularization term on weights
        'lambda': [1e-3],  # L2 regularization term on weights
    }

    def __init__(self, X, y, seed):
        """
        Initialize a new instance of XGBoost which is a subclass of the Model class which is also
        instantiated inside this constructor.

        Args:
            X (dataframe): Dataframe containing the training information for the model.
            y (array): Array containing the training target variable.
            seed (int): Seed to be used in the LogisticRegression

        """

        Model.__init__(self, X, y, xgb.XGBClassifier(random_state=seed), self.param_grid)

    def train(self):
        """
        Used for training the model, it just calls to the method in the superclass.
        """
        return super().train()