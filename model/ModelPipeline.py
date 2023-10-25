from sklearn.model_selection import train_test_split

from model.Train import Train
from model.EvaluateModel import EvaluateModel
from model.OutputModule import Output
from collections import OrderedDict

class ModelPipeline:
    """
    Pipeline in charge of running all the train/test/evaluation procedure.
    """
    def __init__(self, X_train, X_test, y_train, y_test, parameters):
        """
        Initialize a new instance of ModelPipeline

        Args:
            dataframe (dataframe): Complete data.
            parameters (dictionary): Set of parameters that contain all the needed information for
            running the pipeline.
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.parameters = parameters


    def run(self):
        """
        Trains the model for finally testing and generate a file that will contain all
        the information related to the process.
        """
        # We instantiate the training pipeline and we train the model
        training_pipeline = Train(self.X_train, self.y_train, self.parameters)
        model, feature_importance, best_params = training_pipeline.train()

        self.parameters['best_params'] = best_params

        sorted_feature_importance = OrderedDict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        self.parameters['feature_imporatances'] = sorted_feature_importance

        # We collect all the evaluation metrics from the trained model
        evaluation_pipeline = EvaluateModel(model, self.X_test, self.y_test)
        evaluation_results = evaluation_pipeline.evaluate()

        self.parameters['evaluation_results'] = evaluation_results

        # We generate a file containing all the details of this run
        output_generator = Output(self.parameters)
        return output_generator.generate_dataframe()
