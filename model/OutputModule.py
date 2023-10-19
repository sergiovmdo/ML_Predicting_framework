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

        for key, value in self.parameters.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    columns.append(sub_key)
                    values.append(sub_value)
            else:
                columns.append(key)
                values.append(value)

        return pd.DataFrame([values], columns=columns)
