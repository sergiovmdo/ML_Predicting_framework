from model.models.GradientDescent import GradientDescent
from model.models.LogisticRegression import LogisticRegression
from model.models.RBF_SVM import RBF_SVM
from model.models.RandomForest import RandomForest
from model.models.XGBoost import XGBoost


class Train:
    """
    Object that represents the training procedure that will store all the needed information for it.
    """

    def __init__(self, parameters):
        """
        Initialize a new instance of Train

        Args:
            X_train (dataframe): training data.
            y_train (array): target array corresponding to training data.
            parameters (dictionary): contains all the needed parameters.
        """
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
            model = LogisticRegression(self.parameters['X_train'], self.parameters['y_train'], self.parameters['seed'])
            model = model.train(self.parameters['enable_parameter_search'])
            best_params = model.best_params_

        elif self.parameters['model'] == 'random_forest':
            model = RandomForest(self.parameters['X_train'], self.parameters['y_train'], self.parameters['seed'])
            model = model.train()
            best_params = model.best_params_

        elif self.parameters['model'] == 'xgboost':
            model = XGBoost(self.parameters['X_train'], self.parameters['y_train'], self.parameters['seed'])
            model = model.train()
            best_params = model.best_params_

        elif self.parameters['model'] == 'rbf_svm':
            model = RBF_SVM(self.parameters['X_train'], self.parameters['y_train'], self.parameters['seed'])
            model = model.train()
            best_params = model.best_params_

        elif self.parameters['model'] == 'gradient_descent':
            model = GradientDescent(self.parameters['X_train'], self.parameters['y_train'], self.parameters['seed'])
            model = model.train()
            best_params = model.best_params_

        return model, best_params
