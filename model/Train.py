from model.models.LogisticRegression import LogisticRegression


class Train:
    def __int__(self, X_train, y_train, parameters):
        self.X_train = X_train
        self.y_train = y_train
        self.parameters = parameters


    def train(self):
        if self.parameters['model'] == 'logistic_regression':
            model = LogisticRegression(self.X_train, self.y_train)
            model, best_params = model.train()

            feature_importances = model.coef_[0]

            return model, feature_importances, best_params