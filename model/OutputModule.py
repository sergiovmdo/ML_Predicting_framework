import csv

import pandas as pd


class Output:
    """
    Class that represents the Output of the program, that will allocate all the information related to it.
    """

    def __init__(self, parameters):
        """
        Initialize a new instance of Output

        Args:
            parameters (dictionary): Set of parameters and results that have been collected during execution.
        """
        self.parameters = parameters

    def generate_dataframe(self):
        """
        Method used for generating the dataframe that will store the output of both pipelines.
        """
        columns = []
        values = []

        del self.parameters['X_train']
        del self.parameters['X_test']
        del self.parameters['y_train']
        del self.parameters['y_test']
        del self.parameters['dataframe']
        if 'parameters_grid' in self.parameters:
            del self.parameters['parameters_grid']

        if self.parameters['evaluation_technique'] != 'bootstrap':
            for key, value in self.parameters['evaluation_results'].items():
                self.parameters['evaluation_results'][key] = round(value, 4)
        else:
            metrics = self.parameters['evaluation_results'][0]
            c_stat = self.parameters['evaluation_results'][1]
            for key, value in metrics.items():
                metrics[key] = round(value, 4)

            self.parameters['evaluation_results'] = metrics

            for key, value in c_stat.items():
                c_stat[key] = round(value, 4)

            self.parameters['overfitting'] = c_stat

        for key, value in self.parameters.items():
            if isinstance(value, dict) and key is not 'feature_importances':
                for sub_key, sub_value in value.items():
                    columns.append(sub_key)
                    values.append(sub_value)
            else:
                columns.append(key)
                values.append(value)

        return pd.DataFrame([values], columns=columns)
