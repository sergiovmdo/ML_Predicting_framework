from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression as LogisticRegressionModel

from model.models.Model import Model


class LogisticRegression(Model):
    """
    Subclass of Model that will represent the implementation of a Logistic Regression, containing all the needed
    information for the training of this model.
    """

    param_grid = {
        'C': [0.01, 0.1, 1, 10],  # Regularization parameter
        'penalty': ['l1', 'l2'],  # Regularization type
        'solver': ['liblinear', 'saga'],  # Solver algorithms
        'max_iter': [300]  # Maximum number of iterations for the solver to converge
    }

    def __init__(self, X, y):
        """
        Initialize a new instance of LogisticRegression which is a subclass of the Model class which is also
        instantiated inside this constructor.

        Args:
            X (dataframe): Dataframe containing the training information for the model.
            y (array): Array containing the training target variable.

        """
        Model.__init__(self, X, y, LogisticRegressionModel(), self.param_grid)

    def train(self):
        """
        Used for training the model, it just calls to the method in the superclass.
        """
        return super().train()




