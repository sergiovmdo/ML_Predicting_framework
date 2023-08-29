from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class EvaluateModel:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self):
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

        return evaluation_results
