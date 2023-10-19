from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


class EvaluateModel:
    """
    Object that represents the evaluation module that performs all the operations relating the evaluation of the
    model.
    """
    def __init__(self, model, X_test, y_test):
        """
        Initialize a new instance of

        Args:

        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self):
        """
        We use the trained model for predicting the part of the dataset that we kept for testing
        purposes.

        Once we get the predictions we compared against the truth in order to extract all the common metrics
        used in Machine Learning: Accuracy, precision, recall, f1-score, AUC, and the confussion matrix.

        Returns:
            A dictionary containing all the results.
        """
        # First of all we perform predictions
        y_pred = self.model.predict(self.X_test)

        # Initialize the dictionary to store evaluation results
        evaluation_results = {}

        # Calculate and store accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        evaluation_results['accuracy'] = accuracy

        # Calculate and store precision
        precision = precision_score(self.y_test, y_pred, average='weighted')
        evaluation_results['precision'] = precision

        # Calculate and store recall
        recall = recall_score(self.y_test, y_pred, average='weighted')
        evaluation_results['recall'] = recall

        # Calculate and store F1-score
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        evaluation_results['f1'] = f1

        # Calculate and store confusion matrix
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        evaluation_results['confusion_matrix'] = conf_matrix

        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        auc = roc_auc_score(self.y_test, y_pred_proba)  # y_pred should be probability scores for positive class
        evaluation_results['auc'] = auc

        return evaluation_results
