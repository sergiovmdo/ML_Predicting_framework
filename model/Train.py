from model.models.LogisticRegression import LogisticRegression


class Train:
    """
    Object that represents the training procedure that will store all the needed information for it.
    """

    def __init__(self, X_train, y_train, parameters):
        """
        Initialize a new instance of Train

        Args:
            X_train (dataframe): training data.
            y_train (array): target array corresponding to training data.
            parameters (dictionary): contains all the needed parameters.

        """
        self.X_train = X_train
        self.y_train = y_train
        self.parameters = parameters

    def train(self):
        """
        Performs the training step which varies depending on the type of model that we will be using.

        Returns:
            model (object): trained model.
            feature_importances (dictionary): dictionary containing all the features used for training the model along
                                              with its respective coefficients.
            best_params (dictionary): dictionary containing the best hyperparameters for this model.

        """
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
