import pandas as pd
from sklearn.model_selection import train_test_split
from model.Train import Train
from model.models.EvaluateModel import EvaluateModel


class ModelPipeline:
    def __init__(self, dataframe, parameters):
        self.dataframe = dataframe
        self.parameters = parameters

    def run(self):
        X_train, X_test, y_train, y_test = train_test_split(self.dataframe.drop(self.parameters['target'], axis=1),
                                                            self.dataframe[self.parameters['target']], test_size=0.3)

        # We instantiate the training pipeline and we train the model
        training_pipeline = Train(X_train, y_train, self.parameters)
        model, best_params = training_pipeline.train()
        self.parameters['best_params'] = best_params

        # We collect all the evaluation metrics from the trained model
        evaluation_pipeline = EvaluateModel(model, X_test, y_test)
        evaluation_results = evaluation_pipeline.evaluate()
