import os

from sklearn.model_selection import GridSearchCV

from model.Train import Train
from model.evaluation.EvaluationPipeline import EvaluationPipeline
from model.OutputModule import Output
import joblib


class ModelPipeline:
    """
    Pipeline in charge of running all the train/test/evaluation procedure.
    """
    def __init__(self, parameters):
        """
        Initialize a new instance of ModelPipeline

        Args:
            dataframe (dataframe): Complete data.
            parameters (dictionary): Set of parameters that contain all the needed information for
            running the pipeline.
        """
        self.parameters = parameters


    def run(self):
        """
        Trains the model for finally testing and generate a file that will contain all
        the information related to the process.
        """
        # We instantiate the training pipeline and we train the model
        training_pipeline = Train(self.parameters)
        model = training_pipeline.train()

        if isinstance(model, GridSearchCV):
            model_to_save = model.best_estimator_
        else:
            model_to_save = model

        output_folder = self.parameters['output_folder']
        model_folder = f"{output_folder}/trained_models"
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        joblib.dump(model_to_save, f"{model_folder}/model_CV.joblib")

        # We collect all the evaluation metrics from the trained model
        evaluation_pipeline = EvaluationPipeline(model, self.parameters)
        evaluation_results = evaluation_pipeline.evaluate()

        self.parameters['evaluation_results'] = evaluation_results

        # We generate a file containing all the details of this run
        output_generator = Output(self.parameters)
        return output_generator.generate_dataframe()
