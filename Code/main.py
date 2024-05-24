import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Importing Data
data = pd.read_csv("/kaggle/input/heart-disease/heart.csv")
data.head(6)

# Correlation plot
plt.figure(figsize=(20, 12))
sns.set_context('notebook', font_scale=1.3)
sns.heatmap(data.corr(), annot=True, linewidth=2)
plt.tight_layout()

# One-hot encoding
dfs = pd.get_dummies(data, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# Scaling
sc = StandardScaler()
col_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dfs[col_to_scale] = sc.fit_transform(dfs[col_to_scale])

# Splitting the data
X = dfs.drop('target', axis=1)
y = dfs.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest Classifier
rfc = RandomForestClassifier(random_state=42)
rfcs = RandomizedSearchCV(estimator=rfc, param_distributions=params2, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)
rfcs.fit(X_train, y_train)
y_pred2 = rfcs.predict(X_test)

# SVM Classifier
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf']
}

svm = SVC()
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_svm = grid_search.best_estimator_
y_pred_svm = best_svm.predict(X_test)

# Evaluation metrics for Random Forest
precision_rf = precision_score(y_test, y_pred2)
recall_rf = recall_score(y_test, y_pred2)
f1_rf = f1_score(y_test, y_pred2)
acc_rf = accuracy_score(y_test, y_pred2)

# Evaluation metrics for SVM
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
acc_svm = accuracy_score(y_test, y_pred_svm)

print("Random Forest Accuracy:", acc_rf)
print("SVM Accuracy:", acc_svm)
