from model.models.LogisticRegression import LogisticRegression


class Train:
    def __init__(self, X_train, y_train, parameters):
        self.X_train = X_train
        self.y_train = y_train
        self.parameters = parameters


    def train(self):
        if self.parameters['model'] == 'logistic_regression':
            model = LogisticRegression(self.X_train, self.y_train)
            model, best_params, feature_importances = model.train()

            return model, feature_importances, best_params