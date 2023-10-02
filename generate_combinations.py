import sys
import json
import copy


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


arg1 = sys.argv[1]

# We need to convert our arg2 into a dictionary of parameters
try:
    with open(arg1, "r") as json_file:
        parameters = json.load(json_file)
    if isinstance(parameters, dict):
        print("Converted dictionary:", parameters)
    else:
        print("The input is not a valid dictionary.")
except (ValueError, SyntaxError):
    raise ValueError("Could not convert the parameters file to a dictionary.")

combinations = []
generate_combinations(parameters.copy(), {}, combinations)

folder_path = '/home/sergiov/PycharmProjects/ICB_Response_Model/parameters2'

for i, combination in enumerate(combinations):
    file_name = f"{folder_path}/combination_{i}.json"  # Specify the full file path
    with open(file_name, "w") as json_file:
        json.dump(combination, json_file)
