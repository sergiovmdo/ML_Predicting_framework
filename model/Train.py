from model.models.LogisticRegression import LogisticRegression


class Train:
    def __init__(self, X_train, y_train, parameters):
        self.X_train = X_train
        self.y_train = y_train
        self.parameters = parameters


    def train(self):
        if self.parameters['model'] == 'logistic_regression':
            model = LogisticRegression(self.X_train, self.y_train)
            model = model.train()
            best_params = model.best_params_

            coefficients = model.best_estimator_.coef_[0]
            feature_importances = {}
            feature_names = self.X_train.columns.tolist()

            for i, feature in enumerate(feature_names):
                feature_importances[feature] = coefficients[i]

            return model, feature_importances, best_params