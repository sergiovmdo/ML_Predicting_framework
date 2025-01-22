from model.models.Model import Model
from sklearn.linear_model import SGDClassifier


class GradientDescent(Model):

    def __init__(self, parameters):
        """
        Initialize a new instance of XGBoost which is a subclass of the Model class which is also
        instantiated inside this constructor.

        Args:
            X (dataframe): Dataframe containing the training information for the model.
            y (array): Array containing the training target variable.
            seed (int): Seed to be used in the LogisticRegression

        """
        self.parameters = parameters

        Model.__init__(self, parameters, SGDClassifier(random_state=self.parameters['seed']))

    def train(self):
        """
        Used for training the model, it just calls to the method in the superclass.
        """
        return super().train()