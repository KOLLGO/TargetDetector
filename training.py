import sys
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline
from features import get_model_matrices
import joblib

# ------------------ Prepare Data ------------------

X_train, y_train, X_test, y_test = get_model_matrices(
    sys.argv[1], sys.argv[2]
)  # Get the training feature matrix and labels

# ------------------ Create Pipeline ------------------

pipeline = Pipeline(
    [
        ("scaler", MaxAbsScaler()),  # scales the values
        (
            "svc",
            SVC(kernel="rbf"),
        ),  # SVC = Support Vector Classifier, using the radial basis function (RBF) Kernel
    ]
)

# ------------------ Parameter Optimization ------------------

param_grid = {
    "svc__C": [
        0.1,
        1,
        10,
    ],  # Selected values for parameter C (Penalty for misclassification)
    "svc__gamma": [0.001, 0.01, 0.1, "scale"],
    # Selected values for parameter Gamma (Influence of a single training example)
    # 'scale' is an automatic default value
}

grid_search = GridSearchCV(
    estimator=pipeline,  # The model to be evaluated -> Pipeline (StandardScaler + SVC)
    param_grid=param_grid,  # The parameter combinations to be tested
    cv=5,  # Number of Folds for Cross-Validation, 5 is often considered an optimal value
    scoring="accuracy",  # Evaluation metric
    verbose=1,  # Display progress & status during the search, 1 = simple output
)

# ------------------ Training ------------------

grid_search.fit(X_train, y_train)  # Training the model

# ------------------ Save Model ------------------

best_model = (
    grid_search.best_estimator_
)  # Stores the model with the best parameters found by GridSearch
filename = (
    sys.argv[2] + "model.joblib"
)  # Filename for saving, given as second run argument

joblib.dump(best_model, filename)  # Serialize and save the model

# evaluation = evaluate(X_test, y_test)

# ------------------ Entry Point ------------------

if __name__ == "__main__":
    pass
