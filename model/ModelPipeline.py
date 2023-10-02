from sklearn.model_selection import train_test_split

from model.Train import Train
from model.EvaluateModel import EvaluateModel
from model.OutputModule import Output

class ModelPipeline:
    """
    Pipeline in charge of running all the train/test/evaluation procedure.
    """
    def __init__(self, dataframe, parameters):
        """
        Initialize a new instance of ModelPipeline

        Args:
            dataframe (dataframe): Complete data.
            parameters (dictionary): Set of parameters that contain all the needed information for
            running the pipeline.
        """
        self.dataframe = dataframe
        self.parameters = parameters


    def run(self):
        """
        Splits the data in train/test, trains the model for finally testing and generate a file that will contain all
        the information related to the process.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.dataframe.drop(self.parameters['target'], axis=1),
                                                            self.dataframe[self.parameters['target']], test_size=0.3,
                                                            random_state=self.parameters['seed'])

        # We instantiate the training pipeline and we train the model
        training_pipeline = Train(X_train, y_train, self.parameters)
        model, feature_importances, best_params = training_pipeline.train()

        self.parameters['best_params'] = best_params
        self.parameters['feature_imporatances'] = feature_importances

        # We collect all the evaluation metrics from the trained model
        evaluation_pipeline = EvaluateModel(model, X_test, y_test)
        evaluation_results = evaluation_pipeline.evaluate()

        self.parameters['evaluation_results'] = evaluation_results

        # We generate a file containing all the details of this run
        output_generator = Output(self.parameters)
        return output_generator.generate_dataframe()
