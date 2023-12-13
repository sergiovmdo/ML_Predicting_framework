import numpy as np
from mlxtend.evaluate import bootstrap_point632_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_curve, roc_auc_score

from model.evaluation.EvaluateModel import EvaluateModel


class BootstrapPoint632(EvaluateModel):
    def __init__(self, parameters):
        self.runs = parameters['bootstrap_runs']
        self.parameters = parameters

    def evaluate(self):
        callable_metrics = {'auc': roc_auc_score, 'accuracy': accuracy_score, 'precision': precision_score,
                            'recall': recall_score, 'f1': f1_score}
        metrics = {'auc': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}

        X = self.parameters['X_train']
        y = self.parameters['y_train']

        model, _ = self.instantiate_model(X, y, model_type=self.parameters['model'],
                                           params=self.parameters['best_params'])

        for k, v in callable_metrics.items():
            scores = bootstrap_point632_score(model, X, y, n_splits=self.runs, method='.632+', scoring_func=v,
                                              predict_proba=(k == 'auc'))
            score = np.mean(scores)
            metrics[k] = score

        self.parameters['feature_importances'] = {}

        return metrics
