import sys
import pandas as pd
import time
import json
import copy

from model.ModelPipeline import ModelPipeline
from preprocessing.PreprocessingPipeline import PreprocessingPipeline


def exception_control(parameters):
    valid_parameters = ['target', 'features', 'scaler', 'encoder', 'class_balancer', 'evaluation_technique',
                        'model', 'enable_parameter_search', 'splitting_runs', 'bootstrap_runs', 'output_file',
                        'num_features', 'feature_selector', 'parameters_grid']

    valid_scalers = ['min_max', 'z_score']
    valid_encoders = ['one_hot', 'target_encoding']
    valid_balancers = ['smote', 'random_oversampling']
    valid_evaluation_techniques = ['train_test', 'bootstrap', '.632+']
    valid_models = ['logistic_regression', 'random_forest', 'xgboost', 'rbf_svm', 'gradient_descent']

    for k, v in parameters.items():
        if k not in valid_parameters:
            raise ValueError("This parameter name is not valid: {}".format(k))

        if k == valid_parameters[0]:
            if not isinstance(v, str):
                raise TypeError("Parameter: {} has to be a string".format(str(k)))
        elif k == valid_parameters[1]:
            if not isinstance(v, list) and v != '':
                raise TypeError("Parameter: {} has to be a list or the empty string".format(str(k)))
        elif k == valid_parameters[2]:
            if v not in valid_scalers and v != '':
                raise ValueError("The scaler is not available, the implemented scalers keys are: {}".format(", ".join(valid_scalers)))
        elif k == valid_parameters[3]:
            if v not in valid_encoders and v != '':
                raise ValueError("The encoder is not available, the implemented encoders keys are: {}".format(", ".join(valid_encoders)))
        elif k == valid_parameters[4]:
            if v not in valid_balancers and v != '':
                raise ValueError("The balancer is not available, the implemented balancer keys are: {}".format(", ".join(valid_balancers)))
        elif k == valid_parameters[5]:
            if v not in valid_evaluation_techniques and v != '':
                raise ValueError("The evaluation is not available, the implemented evaluation keys are: {}".format(", ".join(valid_evaluation_techniques)))
        elif k == valid_parameters[6]:
            if v not in valid_models and v != '':
                raise ValueError("The model is not available, the implemented model keys are: {}".format(", ".join(valid_models)))
        elif k == valid_parameters[7]:
            if not isinstance(v, bool):
                raise TypeError("Parameter: {} has to be a boolean".format(str(k)))
        elif k == valid_parameters[8]:
            if not isinstance(v, int):
                raise TypeError("Parameter: {} has to be an integer".format(str(k)))
        elif k == valid_parameters[9]:
            if not isinstance(v, int):
                raise TypeError("Parameter: {} has to be an integer".format(str(k)))
        elif k == valid_parameters[10]:
            if not isinstance(v, str):
                raise TypeError("Parameter: {} has to be a string".format(str(k)))


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

    output_dataframe.T.to_csv(parameters['output_file'])


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
