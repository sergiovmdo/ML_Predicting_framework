from collections import OrderedDict
import pandas as pd
from model.evaluation.EvaluateModel import EvaluateModel


class InternalValidation(EvaluateModel):
    def __init__(self, parameters, model):
        self.runs = parameters['bootstrap_runs']
        self.parameters = parameters
        self.model = model


    def evaluate(self):
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
        for i in range(self.runs):
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
            result_dict = {key+'_cs': overoptimistic_metrics[key] - metrics[key] for key in overoptimistic_metrics}
            result_dict['tpr'] = metrics['tpr']
            result_dict['fpr'] = metrics['fpr']

            new_row = pd.DataFrame([result_dict])

            # Append the new row to the original DataFrame
            overfitting_df = pd.concat([overfitting_df, new_row], axis=0)

            feature_importances_df = pd.concat([feature_importances_df, pd.DataFrame([feature_importances])], axis=0)

        # Calculate the average of each column
        average_values = overfitting_df.drop(['fpr', 'tpr'], axis=1).mean()

        average_feature_importances = feature_importances_df.mean()

        # Convert the Series with average values to a dictionary
        average_dict = average_values.to_dict()

        average_feature_importances = average_feature_importances.to_dict()

        # Round the values to 3 decimals and convert them to float
        rounded_feature_importances = {key: round(float(value), 3) for key, value in
                                       average_feature_importances.items()}

        # Sort the dictionary by values in descending order
        sorted_feature_importances = OrderedDict(
            sorted(rounded_feature_importances.items(), key=lambda x: x[1], reverse=True))

        self.parameters['feature_importances'] = sorted_feature_importances

        # Substract the correction factor to the original measures
        corrected_metrics = {key: overoptimistic_metrics[key] - average_dict[key+'_cs'] for key in overoptimistic_metrics}

        self.plot_roc(overfitting_df, average_dict['auc_cs'], overoptimistic_metrics['auc'], [fpr, tpr])

        # We return the metrics and the overfitting measures
        return corrected_metrics, average_dict


