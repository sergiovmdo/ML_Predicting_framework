from model.models.Model import Model
from sklearn.svm import SVC


class RBF_SVM(Model):

    param_grid = {
        'C': [0.1, 1],
        'gamma': [0.1, 1],
    }

    def __init__(self, X, y, seed):
        """
        Initialize a new instance of LogisticRegression which is a subclass of the Model class which is also
        instantiated inside this constructor.

        Args:
            X (dataframe): Dataframe containing the training information for the model.
            y (array): Array containing the training target variable.
            seed (int): Seed to be used in the LogisticRegression

        """
        Model.__init__(self, X, y, SVC(kernel='rbf', probability=True, random_state=seed), self.param_grid)

    def train(self, enable_parameter_search=False):
        """
        Used for training the model, it just calls to the method in the superclass.
        """
        return super().train(enable_parameter_search)