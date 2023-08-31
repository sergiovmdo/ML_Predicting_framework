import sys
import pandas as pd
import ast

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
        parameters = ast.literal_eval(arg2)
        if isinstance(parameters, dict):
            print("Converted dictionary:", parameters)
        else:
            print("The input is not a valid dictionary.")
    except (ValueError, SyntaxError):
        print("Error: Could not convert the input string to a dictionary.")
        return

    # We read our data from the path extracted from arg1
    dataframe = pd.read_csv(arg1, index_col=0)

    ### PREPROCESSING ###

    preprocessing_pipeline = PreprocessingPipeline(dataframe, parameters)
    dataframe = preprocessing_pipeline.run()

    ### MODEL TRAINING AND TESTING ###

    model_pipeline = ModelPipeline(dataframe, parameters)
    model_pipeline.run()


if __name__ == "__main__":
    main()
