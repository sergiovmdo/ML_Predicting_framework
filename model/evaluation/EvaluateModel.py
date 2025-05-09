import os

from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC


def get_feature_importances(model, model_type, feature_names):
    """
    Retrieves the feature importances from a trained model.

    Args:
        model (object): trained model.

    Returns:
        A dictionary containing the feature importances.
    """
    feature_importances = {}

    if model_type == 'logistic_regression':
        coefficients = model.coef_[0]

        for i, feature in enumerate(feature_names):
            feature_importances[feature] = coefficients[i]

    elif model_type == 'rbf_svm':
        feature_importances = {}
    elif model_type == 'gradient_descent':
        coefficients = model.coef_[0]

        for i, feature in enumerate(feature_names):
            feature_importances[feature] = coefficients[i]

    else:
        coefficients = model.feature_importances_
        feature_importances = {}

        for i, feature in enumerate(feature_names):
            feature_importances[feature] = coefficients[i]

        return feature_importances

    return feature_importances

class EvaluateModel:
    """
    Superclass for the different evaluation techniques implemented in the pipeline, contains all the common methods
    """
    def instantiate_model(self, X_train, y_train, model_type, params=None):
        """
        Fits the model to the data passed as parameters and retrieves the feature importances

        Args:
            X_train (dataframe): Training data.
            y_train (array): Target data.
            model_type (string): Model to be instantiated.
            params (dictionary, optional): Dictionary containing the optimized hyperparameters of the model. Defaults to None.

        Returns:
            The fitted model and a dictionary containing the feature importances.
        """
        model_mapping = {
            'xgboost': xgb.XGBClassifier,
            'random_forest': RandomForestClassifier,
            'logistic_regression': LogisticRegression,
            'rbf_svm': lambda **kwargs: SVC(kernel='rbf', probability=True, **kwargs),
            'gradient_descent': SGDClassifier
        }

        if model_type not in model_mapping:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Instantiate the model with default parameters if params is None
        model_class = model_mapping[model_type]
        model = model_class(**(params or {}))

        model.fit(X_train, y_train)

        feature_names = X_train.columns.tolist()
        feature_importances = get_feature_importances(model, model_type, feature_names)

        return model, feature_importances


    def plot_roc(self, metrics_df, averaged_auc, output_path, overoptimistic_auc=0, overoptimistic_curve=[]):
        """
        Creates the ROC curve plot.

        Args:
            metrics_df (DataFrame): The dataframe containing all the metrics.
            averaged_auc (float): The mean Area Under the Curve of the different runs.
            output_path (str): Path to the output directory.
            overoptimistic_auc (float): The overoptimistic AUC value.
            overoptimistic_curve (list): [fpr, tpr] for an overoptimistic ROC curve (optional).

        Returns:
            None
        """

        plt.figure(figsize=(8, 8))

        if overoptimistic_curve:
            plt.plot(
                overoptimistic_curve[0],
                overoptimistic_curve[1],
                color='navy',
                lw=2,
                linestyle='dotted',
                label='Overoptimistic AUC: ' + str(round(overoptimistic_auc, 2))
            )

        tpr_values = metrics_df['tpr'].tolist()
        fpr_values = metrics_df['fpr'].tolist()

        # -- Fix each run's ROC to start at (0,0), ensure sorted FPR, and replace NaNs --
        for i in range(len(fpr_values)):
            fpr = np.array(fpr_values[i], dtype=float)
            tpr = np.array(tpr_values[i], dtype=float)

            # If empty or not starting at (0,0), insert it
            if fpr.size == 0 or fpr[0] != 0.0 or tpr[0] != 0.0:
                fpr = np.insert(fpr, 0, 0.0)
                tpr = np.insert(tpr, 0, 0.0)

            # Sort by ascending FPR to avoid interpolation issues
            sorted_idx = np.argsort(fpr)
            fpr = fpr[sorted_idx]
            tpr = tpr[sorted_idx]

            # Replace NaNs with 0.0
            fpr = np.nan_to_num(fpr, nan=0.0)
            tpr = np.nan_to_num(tpr, nan=0.0)

            fpr_values[i] = fpr.tolist()
            tpr_values[i] = tpr.tolist()

        # Find the minimum and maximum false-positive rates across all runs
        min_fpr = min(np.min(run_fpr) for run_fpr in fpr_values)
        max_fpr = max(np.max(run_fpr) for run_fpr in fpr_values)

        # Choose a common set of mean_fpr values within the range
        mean_fpr = np.linspace(min_fpr, max_fpr, 100)

        # Interpolate individual ROC curves to the common set of mean_fpr values
        interp_tpr_values = [
            interp1d(fpr, tpr, kind='linear', bounds_error=False, fill_value='extrapolate')(mean_fpr)
            for fpr, tpr in zip(fpr_values, tpr_values)
        ]

        # Plot all runs if plot_mean_roc is False, otherwise only do the mean curve
        if not self.parameters['plot_mean_roc']:
            linestyle = '--'
            for index, row in metrics_df.iterrows():
                tpr = row['tpr']
                fpr = row['fpr']
                plt.plot(fpr, tpr, lw=1, color='grey')
        else:
            linestyle = 'solid'

        # Adjust the displayed AUC if there's an "overoptimistic" value
        if overoptimistic_auc != 0:
            averaged_auc = overoptimistic_auc - averaged_auc

        # Choose color if set
        color = self.parameters['roc_color'] if self.parameters['roc_color'] else 'red'

        interp_tpr_values = np.array(interp_tpr_values, dtype=float)
        interp_tpr_values = np.nan_to_num(interp_tpr_values, nan=0.0)
        mean_tpr = np.mean(interp_tpr_values, axis=0)
        calculated_auc = np.trapz(mean_tpr, mean_fpr)

        self.parameters['AUC_Integral'] = calculated_auc

        # Plot the mean ROC curve
        plt.plot(
            mean_fpr,
            mean_tpr,
            color=color,
            lw=2,
            linestyle=linestyle,
            label='Averaged AUC: ' + str(round(averaged_auc, 2))
        )

        # -----------------------------------------------------
        # Add the diagonal "chance" line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle=':', label='Chance')

        padding = 0.05
        plt.xlim([-padding, 1 + padding])
        plt.ylim([-padding, 1 + padding])
        # -----------------------------------------------------
        plt.grid(True)
        plt.xlabel('1 - Specificity (FPR)')
        plt.ylabel('Sensitivity (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')

        output_dir = os.path.join(output_path, "roc.pdf")
        plt.savefig(output_dir)

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
