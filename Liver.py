import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('./liver.csv')

# Preprocess the data
data.dropna(inplace=True)
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
X = data.drop(columns=['Dataset'])
Y = data['Dataset']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialize models
models = {
    'Support Vector Machine': SVC(C=1, tol=0.0001, gamma='scale', kernel='rbf'),
    'Naive Bayes': GaussianNB()
}

# Train models and evaluate accuracy
accuracy_scores = {}
for model_name, model in models.items():
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    accuracy_scores[model_name] = accuracy_score(Y_test, predictions)

# Plotting the accuracy scores
plt.figure(figsize=(8, 5))
plt.bar(accuracy_scores.keys(), accuracy_scores.values(), color=['blue', 'orange'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.axhline(y=0.5, color='r', linestyle='--')  # Optional: Add a baseline for reference
plt.show()