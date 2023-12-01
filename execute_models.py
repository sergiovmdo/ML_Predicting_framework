import pandas as pd
import sys
import main
from GoogleDocsService import GoogleDocsService
import os
import ast
from collections import OrderedDict


def get_feature_importances_text(df):
    row_name = 'feature_importances'
    column_name = 'Unnamed: 0'
    train_test_features = df.loc[df[column_name] == row_name, '0'].iloc[0]

    dict_str = train_test_features.replace("OrderedDict", "")

    # Using ast.literal_eval to convert the string to an OrderedDict
    ordered_dict = ast.literal_eval(dict_str)

    # Converting to an actual OrderedDict
    ordered_dict = OrderedDict(ordered_dict)

    features_text = '\n'.join(f'{key}: {value}' for key, value in ordered_dict.items())
    return features_text


def get_auc_text_tt(df):
    row_name = 'auc'
    column_name = 'Unnamed: 0'
    auc = df.loc[df[column_name] == row_name, '0'].iloc[0]
    return f"The AUC for train/test is: {auc}\n"


def get_sample_size(df):
    row_name = 'sample_size'
    column_name = 'Unnamed: 0'
    sample_size = df.loc[df[column_name] == row_name, '0'].iloc[0]
    return f"The Sample size is: {sample_size}\n"


def get_auc_text_iv(df):
    row_name = 'evaluation_results'
    column_name = 'Unnamed: 0'
    results = df.loc[df[column_name] == row_name, '0'].iloc[0]

    tuple_of_dicts = ast.literal_eval(results)

    # Separate the tuple into two dictionaries
    dict1, dict2 = tuple_of_dicts

    return f"The AUC for internal validation is: {dict1['auc']} and the correction factor: {dict2['auc']}\n"


def print_model(service, file_id, model_type, features_iv, features_tt, auc_iv, auc_tt, sample_size, subset):
    # asdsad
    service.insert_feature_importances_table(file_id, features_tt, features_iv)
    service.insert_text(file_id, auc_iv)

    service.insert_text(file_id, auc_tt)

    service.insert_text(file_id, sample_size)

    if subset:
        model_type = model_type + subset
    service.insert_title(file_id, model_type)

def write_document(data_path, service, file_id, subset=''):
    for i in range(0, len(file_paths), 2):
        params_iv = file_paths[i]
        params_tt = file_paths[i + 1]

        model_type = params_iv.split('_')[3]
        model_type = model_type.split('/')[1]

        # Execution of internal validation
        sys.argv = [data_path, params_iv]
        main.main()

        df_iv = pd.read_csv('test.csv')

        features_iv = get_feature_importances_text(df_iv)
        auc_iv = get_auc_text_iv(df_iv)

        # Execution of train/test
        sys.argv = [data_path, params_tt]
        main.main()

        df_tt = pd.read_csv('test.csv')

        features_tt = get_feature_importances_text(df_tt)
        auc_tt = get_auc_text_tt(df_tt)

        sample_size = get_sample_size(df_tt)

        # Write document
        print_model(service, file_id, model_type, features_iv, features_tt, auc_iv, auc_tt, sample_size, subset)


# Complete data
complete_path = '/home/sergiov/Desktop/full_data_random.csv'

# Stratified data
infiltrated_data = '/home/sergiov/Desktop/infiltrated_patients.csv'
non_infiltrated_data = '/home/sergiov/Desktop/non-infiltrated_patients.csv'

parameters_path = '/home/sergiov/PycharmProjects/ICB_Response_Model/parameter_files'
file_paths = sorted([os.path.join(root, file) for root, dirs, files in os.walk(parameters_path) for file in files])

service = GoogleDocsService()

title = 'Diary 1'
file_id = service.create_document(title)

write_document(infiltrated_data, service, file_id, ' Infiltrated')

write_document(non_infiltrated_data, service, file_id, ' Non-Infiltrated')

write_document(complete_path, service, file_id)




