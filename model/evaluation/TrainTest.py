import os
import random

import joblib
from sklearn.model_selection import train_test_split
from model.evaluation.EvaluateModel import EvaluateModel
from collections import OrderedDict
import pandas as pd

from preprocessing.ClassBalancer import ClassBalancer


class TrainTest(EvaluateModel):
    def __init__(self, parameters, model):
        self.runs = parameters['splitting_runs']
        self.X = parameters['X_test']
        self.y = parameters['y_test']
        self.parameters = parameters
        self.model = model

    def evaluate(self):
        # Store trained model
        output_folder = self.parameters['output_folder']
        model_folder = f"{output_folder}/trained_models"

        y_pred = self.model.predict(self.X)
        y_pred_proba = self.model.predict_proba(self.X)[:, 1]

        metrics = self.compute_metrics(y_true=self.y, y_pred=y_pred, y_pred_proba=y_pred_proba)

        model_path = f"{model_folder}/model_CV.joblib"
        metrics["model_path"] = model_path

        metrics_df = pd.DataFrame([metrics])

        feature_importances_df = pd.DataFrame()

        seeds = [random.randint(1, 99999) for _ in range(self.runs - 1)]
        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                self.parameters['dataframe'].drop(self.parameters['target'], axis=1),
                self.parameters['dataframe'][self.parameters['target']],
                test_size=self.parameters['test_size'], random_state=seed)

            if 'class_balancer' in self.parameters and self.parameters['class_balancer']:
                class_balancer = ClassBalancer(X_train, y_train,
                                               self.parameters['class_balancer'], self.parameters['seed'])
                X_train, y_train = class_balancer.balance_classes()

            model, feature_importances = self.instantiate_model(X_train, y_train, self.parameters['model'], self.parameters['best_params'])
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            model_path = f"{model_folder}/model_{seed}.joblib"
            joblib.dump(model, model_path)

            metrics = self.compute_metrics(y_true=y_test, y_pred=y_pred, y_pred_proba=y_pred_proba)
            metrics["model_path"] = model_path

            new_row = pd.DataFrame([metrics])

            feature_importances_df = pd.concat([feature_importances_df, pd.DataFrame([feature_importances])], axis=0)
            # Append the new row to the original DataFrame
            metrics_df = pd.concat([metrics_df, new_row], axis=0, ignore_index=True)

        roc_curves_df = metrics_df[['fpr', 'tpr']]

        # Calculate the average of each column
        average_values = metrics_df.drop(['fpr', 'tpr', 'model_path'], axis=1).mean()

        average_feature_importances = feature_importances_df.mean()

        # Convert the Series with average values to a dictionary
        average_dict = average_values.to_dict()

        average_feature_importances = average_feature_importances.to_dict()

        # Round the values to 3 decimals and convert to float
        rounded_feature_importances = {key: round(float(value), 3) for key, value in
                                       average_feature_importances.items()}

        # Sort the dictionary by values in descending order
        sorted_feature_importances = OrderedDict(
            sorted(rounded_feature_importances.items(), key=lambda x: x[1], reverse=True))

        self.parameters['feature_importances'] = sorted_feature_importances

        self.plot_roc(roc_curves_df, average_dict['auc'], self.parameters['output_folder'])
        # Save metrics_df
        output_dir = os.path.join(self.parameters['output_folder'], "full_metrics_df.csv")
        metrics_df.to_csv(output_dir)

        # Store the closest model to mean AUC
        closest_index = (metrics_df['auc'] - average_dict['auc']).abs().nsmallest(1).index[0]
        chosen_model_path = metrics_df.loc[closest_index, 'model_path']

        for path in metrics_df['model_path'].values:
            if path != chosen_model_path and "model_CV" not in path:
                full_path = os.path.abspath(path)  # Convert to absolute path
                if os.path.exists(full_path):
                    os.remove(full_path)
                else:
                    print(f"File not found: {full_path}")

        return average_dict