from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression as LogisticRegressionModel


class LogisticRegression:
    param_grid = {
        'C': [0.01, 0.1, 1, 10],  # Regularization parameter
        'penalty': ['l1', 'l2'],  # Regularization type
        'solver': ['liblinear', 'saga'],  # Solver algorithms
        'max_iter': [300]  # Maximum number of iterations for the solver to converge
    }

    def __init__(self, X, y):
        self.X = X
        self.y = y

        self.best_score = 0.0

    def train(self):
        model = LogisticRegressionModel()

        feature_names = self.X.columns.tolist()

        grid_search = GridSearchCV(model, self.param_grid, cv=15, scoring='roc_auc')
        grid_search.fit(self.X, self.y)

        if round(grid_search.best_score_, 3) > round(self.best_score, 3):
            self.best_score = grid_search.best_score_
            self.best_parameters = grid_search.best_params_

            coefficients = grid_search.best_estimator_.coef_[0]
            coefficients_with_names = {}

            for i, feature in enumerate(feature_names):
                coefficients_with_names[feature] = coefficients[i]

            self.coefficients = coefficients_with_names

            self.modify_grid_params()
            return self.train()

        else:
            return grid_search, self.best_parameters, self.coefficients

    def generate_interval(self, x, y, n):
        """
        Taking x and y as two real numbers, it generates n more numbers in between
        """
        interval = abs(x - y)  # Calculate the interval between a and b
        step = interval / (n + 1)  # Divide the interval into 3 equal parts

        numbers = []
        for i in range(n):
            number = min(x, y) + (i + 1) * step
            numbers.append(number)

        return numbers

    def generate_previous_number(self, x):
        interval = abs(x) / 4
        prev = x - (3 * interval)

        return prev

    def generate_next_number(self, x):
        interval = abs(x) / 4
        prev = x + (3 * interval)

        return prev

    def modify_grid_params(self):
        for parameter, value in self.best_parameters.items():
            if any(isinstance(value, cls) for cls in [int, float]) and len(self.param_grid[parameter]) > 1:
                parameter_values = self.param_grid[parameter]
                position = parameter_values.index(value)

                # First we contemplate the case where the optimal value is the last in the initial range
                if position == len(parameter_values) - 1:
                    new_top = self.generate_next_number(value)
                    new_bottom = parameter_values[position - 1]
                # Optimal value is the first element
                elif position == 0:
                    new_bottom = self.generate_previous_number(value)
                    new_top = parameter_values[position + 1]
                # Optimal value in the middle
                else:
                    new_bottom = parameter_values[position - 1]
                    new_top = parameter_values[position + 1]

                new_range_bottom = self.generate_interval(new_bottom, value, 2)
                new_range_top = self.generate_interval(new_top, value, 2)

                self.param_grid[parameter] = new_range_bottom + new_range_top



