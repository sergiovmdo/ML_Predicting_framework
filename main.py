import sys
import pandas as pd
import ast
import json

from model.ModelPipeline import ModelPipeline
from preprocessing.PreprocessingPipeline import PreprocessingPipeline


def main():
    # Check if command-line arguments were provided
    if len(sys.argv) < 2:
        print("Usage: python script.py argument1 argument2")
        return

    # Access command-line arguments
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]

    # We need to convert our arg2 into a dictionary of parameters
    try:
        with open(arg2, "r") as json_file:
            parameters = json.load(json_file)
        if isinstance(parameters, dict):
            print("Converted dictionary:", parameters)
        else:
            print("The input is not a valid dictionary.")
    except (ValueError, SyntaxError):
        raise ValueError("Could not convert the parameters file to a dictionary.")

    # We read the data from our first parameter
    try:
        # We read our data from the path extracted from arg1
        dataframe = pd.read_csv(arg1, index_col=0)
    except (ValueError, SyntaxError):
        raise ValueError("Data has to be in a comma separated csv format")

    ### PREPROCESSING ###

    preprocessing_pipeline = PreprocessingPipeline(dataframe, parameters)
    dataframe = preprocessing_pipeline.run()

    ### MODEL TRAINING AND TESTING ###

    model_pipeline = ModelPipeline(dataframe, parameters)
    model_pipeline.run()


if __name__ == "__main__":
    main()
