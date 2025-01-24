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
        Performs the training step, which varies depending on the type of model specified in the parameters.

        Returns:
            model (object): Trained model.
            feature_importances (dictionary): Dictionary containing all the features used for training the model along
                                               with its respective coefficients (if applicable).
            best_params (dictionary): Dictionary containing the best hyperparameters for this model.
        """
        model_mapping = {
            'logistic_regression': LogisticRegression,
            'random_forest': RandomForest,
            'xgboost': XGBoost,
            'rbf_svm': RBF_SVM,
            'gradient_descent': GradientDescent
        }

        model_type = self.parameters.get('model')

        if model_type not in model_mapping:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Initialize and train the model
        model_class = model_mapping[model_type]
        model = model_class(self.parameters)
        model = model.train()

        # Retrieve the best parameters if the model supports it
        self.parameters['best_params'] = (model.best_params_
                                          if hasattr(model, 'best_params_') else
                                          model.get_params())

        return model

