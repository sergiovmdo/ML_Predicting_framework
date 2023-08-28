from sklearn.model_selection import GridSearchCV


class LogisticRegression:
    param_grid = {
        'C': [0.01, 0.1, 1, 10], # Regularization parameter
        'penalty': ['l1', 'l2'], # Regularization type
        'solver': ['liblinear', 'saga'], # Solver algorithms
        'max_iter': [300] # Maximum number of iterations for the solver to converge
    }

    def __int__(self, X, y):
        self.X = X
        self.y = y

        self.best_score = 0.0

        self.train()

    def train(self):
        model = LogisticRegression()

        grid_search = GridSearchCV(model, self.param_grid, cv=15, scoring='roc_auc')
        grid_search.fit(self.X, self.y)

        if grid_search.best_score_ > self.best_score:
            self.best_score = grid_search.best_score_
            self.best_parameters = grid_search.best_params_

            self.modify_grid_params()
            self.train()

        else:
            return model, self.best_parameters

    def modify_grid_params(self):
        pass
