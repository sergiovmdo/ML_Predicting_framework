import sys
import pandas as pd
import time
import json
import copy

import os

# Adjust path for xgboost DLLs
if getattr(sys, 'frozen', False):  # If running from a PyInstaller EXE
    base_path = sys._MEIPASS
    os.environ["PATH"] = os.pathsep.join([
        os.path.join(base_path, "xgboost", "lib"),
        os.environ["PATH"]
    ])

from model.ModelPipeline import ModelPipeline
from preprocessing.PreprocessingPipeline import PreprocessingPipeline


def exception_control(parameters):
    valid_parameters = [
        'target', 'features', 'imputer', 'scaler', 'encoder', 'class_balancer', 'evaluation_technique',
        'model', 'enable_parameter_search', 'splitting_runs', 'bootstrap_runs', 'output_folder',
        'num_features', 'feature_selector', 'parameters_grid', 'plot_mean_roc', 'roc_color'
    ]

    valid_values = {
        'scaler': ['min_max', 'z_score'],
        'encoder': ['one_hot', 'target_encoding'],
        'class_balancer': ['smote', 'random_oversampling'],
        'imputer': ['simple_imputer'],
        'evaluation_technique': ['train_test', 'bootstrap', '.632+'],
        'model': ['logistic_regression', 'random_forest', 'xgboost', 'rbf_svm', 'gradient_descent']
    }

    type_checks = {
        'target': str,
        'features': (list, str),  # Allow list or empty string
        'imputer': str,  # Specific validation will be done later
        'scaler': str,  # Specific validation will be done later
        'encoder': str,  # Specific validation will be done later
        'class_balancer': str,  # Specific validation will be done later
        'evaluation_technique': str,  # Specific validation will be done later
        'model': str,  # Specific validation will be done later
        'enable_parameter_search': bool,
        'splitting_runs': int,
        'bootstrap_runs': int,
        'output_folder': str,
        'num_features': int,
        'feature_selector': str,  # No specific validation for now
        'parameters_grid': dict,  # Typically a dictionary for grid search
        'plot_mean_roc': bool,
        'roc_color': str  # Optional string for color
    }

    for k, v in parameters.items():
        if k not in valid_parameters:
            raise ValueError(f"This parameter name is not valid: {k}")

        # Type validation
        expected_type = type_checks.get(k)
        if expected_type and not isinstance(v, expected_type) and v != '':
            raise TypeError(f"Parameter '{k}' must be of type {expected_type.__name__}.")

        # Specific value validation
        if k in valid_values and v not in valid_values[k] and v != '':
            valid_options = ", ".join(valid_values[k])
            raise ValueError(f"Invalid value for parameter '{k}'. Valid options are: {valid_options}.")


def generate_combinations(json_obj, current_combination, combinations_list):
    if not json_obj:
        combinations_list.append(current_combination)
        return

    key, value = json_obj.popitem()

    if ("features" not in key) and isinstance(value, list):
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

    if len(sys.argv) < 3:
        # Access command-line arguments
        arg1 = sys.argv[0]
        arg2 = sys.argv[1]
    else:
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

    # Ensure the output directory exists before running the pipeline
    output_folder = parameters['output_folder']
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # We read the data from our first parameter
    try:
        # We read our data from the path extracted from arg1
        dataframe = pd.read_csv(arg1)
    except (ValueError, SyntaxError):
        raise ValueError("Data has to be in a comma separated csv format")

    combinations = []
    generate_combinations(parameters.copy(), {}, combinations)

    output_dataframe = pd.DataFrame()
    # We iterate through all the possible parameter combinations.
    for combination in combinations:
        # Raises an exception if any of the parameters is incorrect
        exception_control(combination)

        ### PREPROCESSING ###

        preprocessing_pipeline = PreprocessingPipeline(dataframe, combination)
        local_parameters = preprocessing_pipeline.run()

        ### MODEL TRAINING AND TESTING ###

        model_pipeline = ModelPipeline(local_parameters)
        aux = model_pipeline.run()
        output_dataframe = pd.concat([output_dataframe, aux], ignore_index=True)

    # Save the output dataframe to the specified folder
    output_path = os.path.join(output_folder, "output_dataframe.csv")
    output_dataframe.T.to_csv(output_path)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
