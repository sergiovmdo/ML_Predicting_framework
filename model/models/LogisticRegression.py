from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression as LogisticRegressionModel


class LogisticRegression:
    param_grid = {
        'C': [0.01, 0.1, 1, 10], # Regularization parameter
        'penalty': ['l1', 'l2'], # Regularization type
        'solver': ['liblinear', 'saga'], # Solver algorithms
        'max_iter': [300] # Maximum number of iterations for the solver to converge
    }

    def __init__(self, X, y):
        self.X = X
        self.y = y

        self.best_score = 0.0

    def train(self):
        model = LogisticRegressionModel()

        grid_search = GridSearchCV(model, self.param_grid, cv=15, scoring='roc_auc')
        grid_search.fit(self.X, self.y)

        if grid_search.best_score_ > self.best_score:
            self.best_score = grid_search.best_score_
            self.best_parameters = grid_search.best_params_

            self.modify_grid_params()
            return self.train()

        else:
            return grid_search, self.best_parameters, grid_search.best_estimator_.coef_

    def modify_grid_params(self):
        pass
