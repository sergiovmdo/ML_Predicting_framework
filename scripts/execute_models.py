import pandas as pd
import sys
import subprocess

import main
from GoogleDocsService import GoogleDocsService
import os
import ast
from collections import OrderedDict

# Specify the path to the main module in ProjectA
path_to_main_module = '/home/sergiov/PycharmProjects/ICB_Response_Model/main.py'


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
    return f"The AUC for train/test is: {auc}\n", auc


def get_sample_size(df):
    row_name = 'sample_size'
    column_name = 'Unnamed: 0'
    sample_size = df.loc[df[column_name] == row_name, '0'].iloc[0]
    return f"The Sample size is: {sample_size}\n", sample_size


def get_auc_text_iv(df):
    row_name = 'auc'
    column_name = 'Unnamed: 0'
    auc = df.loc[df[column_name] == row_name, '0'].iloc[0]

    row_name = 'auc_cs'
    column_name = 'Unnamed: 0'
    auc_cs = df.loc[df[column_name] == row_name, '0'].iloc[0]

    return f"The AUC for internal validation is: {auc} and the correction factor: {auc_cs}\n", auc, auc_cs

def get_auc_text_632(df):
    row_name = 'auc'
    column_name = 'Unnamed: 0'
    auc = df.loc[df[column_name] == row_name, '0'].iloc[0]

    return f"The AUC for Bootstrap .632+ is: {auc}", auc


def print_model(service, file_id, features_iv, features_tt, auc_iv, auc_tt, sample_size, model_name):
    # asdsad
    #service.insert_feature_importances_table(file_id, features_tt, features_iv)
    service.create_table(file_id, 2, 2, [["Train/Test", "Bootstrap .632+"], [features_tt, features_iv]])
    service.insert_text(file_id, auc_iv)

    service.insert_text(file_id, auc_tt)

    service.insert_text(file_id, sample_size)

    service.insert_heading(file_id, model_name, 'HEADING_2')


def write_summary_table(service, file_id, data):
    #service.insert_summary_table(file_id, data)
    service.create_table(file_id, rows=len(data), columns=len(data[0]), table_data=data)


def write_document(pipeline_output_path, file_paths, data_path, service, file_id, subset=''):
    summary_values = []
    for i in range(0, len(file_paths), 2):
    #for i in range(0, 1, 2):
        params_iv = file_paths[i]
        params_tt = file_paths[i + 1]

        model_type = os.path.basename(params_iv)
        model_type = model_type.split('_')[0]

        # Execution of train/test
        sys.argv = [data_path, params_tt]
        main.main()

        df_tt = pd.read_csv(pipeline_output_path)

        features_tt = get_feature_importances_text(df_tt)
        if not features_tt:
            features_tt = '-'

        auc_tt_text, auc_tt = get_auc_text_tt(df_tt)

        sample_size_text, sample_size = get_sample_size(df_tt)

        # Execution of internal validation
        sys.argv = [data_path, params_iv]
        main.main()

        df_iv = pd.read_csv(pipeline_output_path)

        features_iv = get_feature_importances_text(df_iv)
        if not features_iv:
            features_iv = '-'

        auc_iv_text, auc_iv = get_auc_text_632(df_iv)

        # Create model name and append values to summary list
        model_name = subset + model_type

        summary_values.append([model_name, sample_size, auc_tt, auc_iv])

        # Write document
        print_model(service, file_id, features_iv, features_tt, auc_iv_text, auc_tt_text, sample_size_text, model_name)

    return summary_values


# Complete data
complete_path = '/datasets/sergio/Integrated_data/df_WES+RNA_response.csv.csv'

# Stratified data
infiltrated_data = '/datasets/sergio/Integrated_data/df_WES+RNA_response.csv_inf.csv'
non_infiltrated_data = '/datasets/sergio/Integrated_data/df_WES+RNA_response.csv_non-inf.csv'

parameters_path = '/home/sergiov/PycharmProjects/Reporting_Tool/parameters/parameter_files_xgb_632'
output_path = '/home/sergiov/PycharmProjects/ICB_Response_Model/test.csv'

file_paths = sorted([os.path.join(root, file) for root, dirs, files in os.walk(parameters_path) for file in files])

service = GoogleDocsService(token_path='/home/sergiov/PycharmProjects/ICB_Response_Model/scripts/token.pickle',
                            secret_path='/home/sergiov/Downloads/client_secret.json')

title = '1412_xgboost'
file_id = service.create_document(title)

summary_values = [['Model type', 'Sample Size', 'AUC Train/Test', 'AUC Bootstrap .632+']]

summary_values += write_document(output_path, file_paths, infiltrated_data, service, file_id, 'Inf ')

summary_values += write_document(output_path, file_paths, non_infiltrated_data, service, file_id, 'NInf ')

summary_values += write_document(output_path, file_paths, complete_path, service, file_id, 'All ')

write_summary_table(service, file_id, summary_values)


