from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression as LogisticRegressionModel

from model.models.Model import Model


class LogisticRegression(Model):
    param_grid = {
        'C': [0.01, 0.1, 1, 10],  # Regularization parameter
        'penalty': ['l1', 'l2'],  # Regularization type
        'solver': ['liblinear', 'saga'],  # Solver algorithms
        'max_iter': [300]  # Maximum number of iterations for the solver to converge
    }

    def __init__(self, X, y):
        Model.__init__(self, X, y, LogisticRegressionModel(), self.param_grid)

    def train(self):
        return super().train()




