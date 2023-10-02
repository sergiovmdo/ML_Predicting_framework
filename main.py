import sys
import pandas as pd
import time
import json
import itertools
import copy

from model.ModelPipeline import ModelPipeline
from preprocessing.PreprocessingPipeline import PreprocessingPipeline


def generate_combinations(json_obj, current_combination, combinations_list):
    if not json_obj:
        combinations_list.append(current_combination)
        return

    key, value = json_obj.popitem()

    if isinstance(value, list):
        for item in value:
            new_combination = copy.deepcopy(current_combination)
            new_combination[key] = item
            generate_combinations(json_obj.copy(), new_combination, combinations_list)
    else:
        current_combination[key] = value
        generate_combinations(json_obj, current_combination, combinations_list)


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

    combinations = []
    generate_combinations(parameters.copy(), {}, combinations)

    output_dataframe = pd.DataFrame()
    # We iterate through all the possible parameter combinations.
    for combination in combinations:
        ### PREPROCESSING ###

        preprocessing_pipeline = PreprocessingPipeline(dataframe, combination)
        dataframe = preprocessing_pipeline.run()

        ### MODEL TRAINING AND TESTING ###

        model_pipeline = ModelPipeline(dataframe, combination)
        aux = model_pipeline.run()
        output_dataframe = pd.concat([output_dataframe, aux], ignore_index=True)

    output_dataframe.to_csv(parameters['output_file'])


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
