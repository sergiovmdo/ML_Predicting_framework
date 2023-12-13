import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create a sample DataFrame
data = {
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
    'Target': [0, 1, 0, 1, 1]  # Binary target variable (0 or 1)
}

df = pd.DataFrame(data)

# Split the data into features (X) and target (y)
X = df[['Feature1', 'Feature2']]
y = df['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an XGBoost classifier
clf = xgb.XGBClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
