from model.models.Model import Model
from sklearn.linear_model import SGDClassifier


class GradientDescent(Model):
    param_grid = {
        'loss': ['log_loss'],  # The loss function to be optimized
        'penalty': ['l2'],  # The regularization term to be applied ('l2', 'l1', or 'elasticnet')
        'alpha': [0.0001],  # Constant that multiplies the regularization term
        'l1_ratio': [0.15], # The mix ratio between 'l1' and 'l2' regularization for 'elasticnet' penalty
        'learning_rate': ['optimal'],  # The learning rate schedule
        'eta0': [0.0],  # The initial learning rate for 'constant', 'invscaling', and 'adaptive' schedules
        'max_iter': [1000]  # The maximum number of iterations
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

        Model.__init__(self, X, y, SGDClassifier(random_state=seed), self.param_grid)

    def train(self, enable_parameter_search=False):
        """
        Used for training the model, it just calls to the method in the superclass.
        """
        return super().train(enable_parameter_search)