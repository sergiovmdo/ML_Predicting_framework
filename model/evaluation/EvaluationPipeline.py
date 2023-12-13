from model.evaluation.BootstrapPoint632 import BootstrapPoint632
from model.evaluation.InternalValidation import InternalValidation
from model.evaluation.TrainTest import TrainTest


class EvaluationPipeline:
    """
    Object that represents the evaluation module that performs all the operations relating the evaluation of the
    model.
    """

    def __init__(self, model, parameters):
        """
        Initialize a new instance of

        Args:

        """
        self.model = model
        self.parameters = parameters

    def evaluate(self):
        """
        We use the trained model for predicting the part of the dataset that we kept for testing
        purposes.

        Once we get the predictions we compared against the truth in order to extract all the common metrics
        used in Machine Learning: Accuracy, precision, recall, f1-score, AUC, and the confussion matrix.

        Returns:
            A dictionary containing all the results.
        """

        if self.parameters['evaluation_technique'] == 'train_test':
            evaluation = TrainTest(self.parameters, self.model)

            return evaluation.evaluate()
        elif self.parameters['evaluation_technique'] == 'bootstrap':
            evaluation = InternalValidation(self.parameters, self.model)
            return evaluation.evaluate()
        elif self.parameters['evaluation_technique'] == '.632+':
            evaluation = BootstrapPoint632(self.parameters)
            return evaluation.evaluate()
