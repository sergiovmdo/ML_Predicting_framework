from collections import OrderedDict

from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


class EvaluateModel:
    """
    Object that represents the evaluation module that performs all the operations relating the evaluation of the
    model.
    """

    def __init__(self, model, parameters):
        """
        Initialize a new instance of

        Args:

        """
        self.model = model
        self.parameters = parameters

    def get_feature_importances(self, model):
        """
        Retrieves the feature importances from a trained model.

        Args:
            model (object): trained model.

        Returns:
            A dictionary containing the feature importances.
        """
        coefficients = model.feature_importances_
        feature_importances = {}
        feature_names = self.parameters['X_train'].columns.tolist()

        for i, feature in enumerate(feature_names):
            feature_importances[feature] = coefficients[i]

        return feature_importances


    def instantiate_model(self, X_train, y_train, model_type, params):
        if model_type == 'xgboost':
            model = xgb.XGBClassifier(**params)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(**params)
        elif model_type == 'logistic_regression':
            model = LogisticRegression(**params)

        model.fit(X_train, y_train)

        if model_type == 'logistic_regression':
            coefficients = model.coef_[0]
            feature_importances = {}
            feature_names = self.parameters['X_train'].columns.tolist()

            for i, feature in enumerate(feature_names):
                feature_importances[feature] = coefficients[i]
        else:
            feature_importances = self.get_feature_importances(model)

        return model, feature_importances

    def plot_roc(self, metrics_df, averaged_auc, overoptimistic_auc=0, overoptimistic_curve=[]):
        plt.figure(figsize=(8, 8))

        if overoptimistic_curve:
            plt.plot(overoptimistic_curve[0], overoptimistic_curve[1], color='navy', lw=2, linestyle='dotted',
                     label='Overoptimistic AUC: ' + str(round(overoptimistic_auc, 2)))


        tpr_values = metrics_df['tpr'].tolist()
        fpr_values = metrics_df['fpr'].tolist()

        # Find the minimum and maximum false positive rates across all runs
        min_fpr = min(np.min(run_fpr) for run_fpr in fpr_values)
        max_fpr = max(np.max(run_fpr) for run_fpr in fpr_values)

        # Choose a common set of mean_fpr values within the range
        mean_fpr = np.linspace(min_fpr, max_fpr, 100)

        # Interpolate individual ROC curves to the common set of mean_fpr values
        interp_tpr_values = [interp1d(fpr, tpr, kind='linear', fill_value='extrapolate')(mean_fpr) for fpr, tpr in
                             zip(fpr_values, tpr_values)]

        for index, row in metrics_df.iterrows():
            tpr = row['tpr']
            fpr = row['fpr']

            plt.plot(fpr, tpr, lw=1, color='grey')

        mean_tpr = np.mean(interp_tpr_values, axis=0)

        if overoptimistic_auc != 0:
            averaged_auc = overoptimistic_auc-averaged_auc

        plt.plot(mean_fpr, mean_tpr, color='red', lw=2, linestyle='--',
                 label='Averaged AUC: ' + str(round(averaged_auc, 2)))

        plt.xlabel('1 - Specificity (FPR)')
        plt.ylabel('Sensitivity (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')

        plt.savefig('roc.pdf')

    def compute_metrics(self, y_true, y_pred, y_pred_proba):
        """
        Once we get the predictions we compared against the truth in order to extract all the common metrics
        used in Machine Learning: Accuracy, precision, recall, f1-score, AUC, and the confussion matrix.

        Args:
            y_pred (array): binary predictions
            y_pred_proba (array): probabilistic predictions

        Returns:
            The dictionary containing all the evaluation metrics
        """
        # Initialize the dictionary to store evaluation results
        evaluation_results = {}

        # Calculate and store accuracy
        accuracy = accuracy_score(y_true, y_pred)
        evaluation_results['accuracy'] = accuracy

        # Calculate and store precision
        precision = precision_score(y_true, y_pred, average='weighted')
        evaluation_results['precision'] = precision

        # Calculate and store recall
        recall = recall_score(y_true, y_pred, average='weighted')
        evaluation_results['recall'] = recall

        # Calculate and store F1-score
        f1 = f1_score(y_true, y_pred, average='weighted')
        evaluation_results['f1'] = f1

        # # Calculate and store confusion matrix
        # conf_matrix = confusion_matrix(y_true, y_pred)
        # evaluation_results['confusion_matrix'] = conf_matrix

        # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        evaluation_results['fpr'] = fpr
        evaluation_results['tpr'] = tpr

        roc_auc = auc(fpr, tpr)
        evaluation_results['auc'] = roc_auc

        return evaluation_results

    def evaluate_by_splitting(self, X, y, runs):
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]

        metrics = self.compute_metrics(y_true=y, y_pred=y_pred, y_pred_proba=y_pred_proba)

        metrics_df = pd.DataFrame([metrics])

        feature_importances_df = pd.DataFrame()

        seeds = [random.randint(1, 99999) for _ in range(runs)]
        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                self.parameters['dataframe'].drop(self.parameters['target'], axis=1),
                self.parameters['dataframe'][self.parameters['target']],
                test_size=0.3, random_state=seed)

            model, feature_importances = self.instantiate_model(X_train, y_train, self.parameters['model'], self.parameters['best_params'])
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            metrics = self.compute_metrics(y_true=y_test, y_pred=y_pred, y_pred_proba=y_pred_proba)

            new_row = pd.DataFrame([metrics])

            feature_importances_df = pd.concat([feature_importances_df, pd.DataFrame([feature_importances])], axis=0)
            # Append the new row to the original DataFrame
            metrics_df = pd.concat([metrics_df, new_row], axis=0)

        roc_curves_df = metrics_df[['fpr', 'tpr']]

        # Calculate the average of each column
        average_values = metrics_df.drop(['fpr', 'tpr'], axis=1).mean()

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

        self.plot_roc(roc_curves_df, average_dict['auc'])

        return average_dict

    def evaluate_by_overoptimism(self, runs):
        # Original data
        X = self.parameters['X_train']
        y = self.parameters['y_train']

        original_data = pd.concat([X, y], axis=1)

        # Predict over the initial dataset to obtain the overoptimistic metrics
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]

        overoptimistic_metrics = self.compute_metrics(y_true=y, y_pred=y_pred, y_pred_proba=y_pred_proba)
        tpr = overoptimistic_metrics['tpr']
        fpr = overoptimistic_metrics['fpr']

        del overoptimistic_metrics['tpr']
        del overoptimistic_metrics['fpr']


        overfitting_df = pd.DataFrame()
        feature_importances_df = pd.DataFrame()

        # Sample n bootstrap samples
        for i in range(runs):
            # Bootstrap sample of size n
            bootstrap_sample = original_data.sample(n=len(original_data), replace=True)

            boostrapped_X = bootstrap_sample.drop(self.parameters['target'], axis=1)
            boostrapped_y = bootstrap_sample[self.parameters['target']]

            # Fit the model on the bootstrap sample
            model, feature_importances  = self.instantiate_model(boostrapped_X, boostrapped_y, self.parameters['model'],
                                           self.parameters['best_params'])

            # Use the fitted model over the original data
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1]

            metrics = self.compute_metrics(y_true=y, y_pred=y_pred, y_pred_proba=y_pred_proba)

            # Substract the obtained metrics to the original ones --> Estimation of overfitting
            result_dict = {key: overoptimistic_metrics[key] - metrics[key] for key in overoptimistic_metrics}
            result_dict['tpr'] = metrics['tpr']
            result_dict['fpr'] = metrics['fpr']

            new_row = pd.DataFrame([result_dict])

            feature_importances_df = pd.concat([feature_importances_df, pd.DataFrame([feature_importances])], axis=0)

            # Append the new row to the original DataFrame
            overfitting_df = pd.concat([overfitting_df, new_row], axis=0)

        # Calculate the average of each column
        average_values = overfitting_df.drop(['fpr', 'tpr'], axis=1).mean()

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

        # Substract the correction factor to the original measures
        unbiased_metrics = {key: overoptimistic_metrics[key] - average_dict[key] for key in overoptimistic_metrics}

        self.plot_roc(overfitting_df, average_dict['auc'], overoptimistic_metrics['auc'], [fpr, tpr])

        # We return the metrics and the overfitting measures
        return unbiased_metrics, average_dict

    def evaluate(self):
        """
        We use the trained model for predicting the part of the dataset that we kept for testing
        purposes.

        Once we get the predictions we compared against the truth in order to extract all the common metrics
        used in Machine Learning: Accuracy, precision, recall, f1-score, AUC, and the confussion matrix.

        Returns:
            A dictionary containing all the results.
        """

        if self.parameters['split_validation']:
            X = self.parameters['X_test']
            y = self.parameters['y_test']

            return self.evaluate_by_splitting(X, y, self.parameters['splitting_runs'])
        else:
            return self.evaluate_by_overoptimism(self.parameters['bootstrap_runs'])
