# ------------------ Imports ------------------

from calc import *
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from features import get_feature_matrices
import joblib

# ------------------ Prepare Data ------------------

X_train, y_train, X_test, y_test = get_feature_matrices()  # Get the feature matrices (X_train, y_train, X_test, y_test)

# ------------------ Create Pipeline ------------------

pipeline = Pipeline([
    ("scaler", StandardScaler()),   # Standardizes the values
    ("svc", SVC(kernel="rbf"))    # SVC = Support Vector Classifier, using the radial basis function (RBF) Kernel
])

# ------------------ Parameter Optimization ------------------

param_grid = {
    'svc__C': [0.1, 1, 10],     # Selected values for parameter C (Penalty for misclassification)
    'svc__gamma': [0.001, 0.01, 0.1, 'scale']
    # Selected values for parameter Gamma (Influence of a single training example)
    # 'scale' is an automatic default value
}

grid_search = GridSearchCV(
    estimator=pipeline,  # The model to be evaluated -> Pipeline (StandardScaler + SVC)
    param_grid=param_grid,  # The parameter combinations to be tested
    cv=5,  # Number of Folds for Cross-Validation, 5 is often considered an optimal value
    scoring='accuracy',  # Evaluation metric
    verbose=1  # Display progress & status during the search, 1 = simple output
)

# ------------------ Training ------------------

grid_search.fit(X_train, y_train)  # Training the model

# ------------------ Save Model ------------------

best_model = grid_search.best_estimator_    # Stores the model with the best parameters found by GridSearch
filename = "model.joblib"  # Filename for saving
joblib.dump(best_model, filename)  # Serialize and save the model

# ------------------ Entry Point ------------------

if __name__ == "__main__":
    pass
