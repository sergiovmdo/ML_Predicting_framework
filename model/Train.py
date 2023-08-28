from model.models.LogisticRegression import LogisticRegression


class Train:
    def __int__(self, dataframe, parameters):
        self.dataframe = dataframe
        self.parameters = parameters


    def train(self):
        if self.parameters['model'] == 'logistic_regression':
            model = LogisticRegression(self.dataframe)
            model, best_params = model.train()