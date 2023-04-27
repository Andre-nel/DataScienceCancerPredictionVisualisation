import pickle
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


current_file_path = Path(__file__).resolve()
root_path = Path(current_file_path / "../../../..").resolve()

# Load the data
pre_pca_data = pd.read_csv(root_path / ("src/visualization/cancer/"
                                        "data/cancer_data.csv"))
pre_all_data = pd.read_csv(root_path / ("src/visualization/cancer/"
                                        "data/cancer_data_all_features.csv"))

# Data to test on
test_on = "all"

if test_on == "pca":
    data = pre_pca_data
else:
    data = pre_all_data

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


log_reg = LogisticRegression(random_state=42, max_iter=1000)

# Tune hyperparameters using GridSearchCV
params = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
grid_log_reg = GridSearchCV(log_reg, param_grid=params, scoring='f1', cv=5)
grid_log_reg.fit(X_train, y_train)

# Best model
log_reg_best = grid_log_reg.best_estimator_

# save the trained model to a pickle file
if test_on == "pca":
    with open(root_path / "src/visualization/cancer/model/logistic_regression_model.pkl", 'wb') as f:
        pickle.dump(log_reg_best, f)
else:
    with open(root_path / "src/visualization/cancer/model/logistic_regression_model_all_features.pkl", 'wb') as f:
        pickle.dump(log_reg_best, f)


svc = SVC(random_state=42)

# Tune hyperparameters using GridSearchCV
params = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
grid_svc = GridSearchCV(svc, param_grid=params, scoring='f1', cv=5)
grid_svc.fit(X_train, y_train)

# Best model
svc_best = grid_svc.best_estimator_


rf = RandomForestClassifier(random_state=42)

# Tune hyperparameters using GridSearchCV
params = {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20, 30], "min_samples_split": [2, 5, 10]}
grid_rf = GridSearchCV(rf, param_grid=params, scoring='f1', cv=5)
grid_rf.fit(X_train, y_train)

# Best model
rf_best = grid_rf.best_estimator_


gb = GradientBoostingClassifier(random_state=42)

# Tune hyperparameters using GridSearchCV
params = {"n_estimators": [100, 200, 300], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 7]}
grid_gb = GridSearchCV(gb, param_grid=params, scoring='f1', cv=5)
grid_gb.fit(X_train, y_train)

# Best model
gb_best = grid_gb.best_estimator_


# # Evaluate models
log_reg_eval = evaluate_model(log_reg_best, X_test, y_test)
svc_eval = evaluate_model(svc_best, X_test, y_test)
rf_eval = evaluate_model(rf_best, X_test, y_test)
gb_eval = evaluate_model(gb_best, X_test, y_test)

# Print results
print("Logistic Regression Performance:", log_reg_eval)
print("Support Vector Machine Performance:", svc_eval)
print("Random Forest Performance:", rf_eval)
print("Gradient Boosting Performance:", gb_eval)

"""
Logistic Regression Performance: {'accuracy': 0.9491525423728814, 'precision': 0.9814814814814815,
                                  'recall': 0.9137931034482759, 'f1': 0.9464285714285714}
Support Vector Machine Performance: {'accuracy': 0.940677966101695, 'precision': 0.9636363636363636,
                                     'recall': 0.9137931034482759, 'f1': 0.9380530973451328}
Random Forest Performance: {'accuracy': 0.9322033898305084, 'precision': 0.9464285714285714,
                            'recall': 0.9137931034482759, 'f1': 0.9298245614035087}
Gradient Boosting Performance: {'accuracy': 0.940677966101695, 'precision': 0.9473684210526315,
                                'recall': 0.9310344827586207, 'f1': 0.9391304347826087}

All leatures
Logistic Regression:

Accuracy: 0.9927
Precision: 1.0
Recall: 0.9853
F1 Score: 0.9926

Logistic Regression Performance: {'accuracy': 0.9927007299270073, 'precision': 1.0,
'recall': 0.9852941176470589, 'f1': 0.9925925925925926}

Support Vector Machine:

Accuracy: 0.9708
Precision: 1.0
Recall: 0.9412
F1 Score: 0.9697
Random Forest:

Accuracy: 0.9854
Precision: 0.9853
Recall: 0.9853
F1 Score: 0.9853
Gradient Boosting:

Accuracy: 0.9635
Precision: 1.0
Recall: 0.9265
F1 Score: 0.9618
The Logistic Regression model has the highest accuracy, recall, and F1 score among all models. 
Additionally, it has a perfect precision score, indicating no false positives. 
"""
