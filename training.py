import sys

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline
from features import get_model_matrices
import joblib
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV

# ------------------ Prepare Data ------------------

csv_path = sys.argv[1]
model_folder = sys.argv[2]

if not model_folder.endswith("/"):
    model_folder += "/"

print("----------------- Model Training -----------------")
print("data file: " + csv_path)
print("model folder: " + model_folder)
print()

X_train, y_train = get_model_matrices(
    csv_path, model_folder
)  # Get the training feature matrix and labels

# ------------------ Create Pipelines ------------------

svc_pipeline = Pipeline(
    [
        ("scaler", MaxAbsScaler()),  # scales the values
        (
            "svc",
            SVC(kernel="rbf", probability=True),
        ),  # SVC = Support Vector Classifier, using the radial basis function (RBF) Kernel
    ]
)
"""
linear_svc_pipeline = Pipeline(
    [
        ("scaler", MaxAbsScaler()),
        (
            "classifier",
            CalibratedClassifierCV(
                LinearSVC(max_iter=10000, dual="auto", multi_class="ovr"), cv=3
            ),
        ),
    ]
)"""

naive_bayes_pipeline = Pipeline(
    [
        ("scaler", MaxAbsScaler()),
        ("classifier", MultinomialNB()),
    ]
)

logistic_pipeline = Pipeline(
    [
        ("scaler", MaxAbsScaler()),
        ("classifier", LogisticRegression(max_iter=10000, multi_class="multinomial")),
    ]
)

# ------------------ Meta Classifier -------------------
meta_classifier = VotingClassifier(
    estimators=[
        ("svc", svc_pipeline),
        ("naive_bayes", naive_bayes_pipeline),
        ("logistic", logistic_pipeline),
    ],
    voting="soft",  # weighted probabilities
    n_jobs=-1,  # all available CPU-cores
)

# ------------------ Parameter Optimization ------------------

param_grid = {
    "svc__svc__C": [0.1, 1, 10],
    "svc__svc__gamma": [0.001, 0.01, 0.1, "scale"],
    "naive_bayes__classifier__alpha": [1.0, 0.1, 0.01],
    "logistic__classifier__C": [0.1, 1, 10],
    "logistic__classifier__solver": ["lbfgs", "newton-cg"],
    "voting": ["soft", "hard"],
}
"""param_grid = {
    "svc__C": [
        0.1,
        1,
        10,
    ],  # Selected values for parameter C (Penalty for misclassification)
    "svc__gamma": [0.001, 0.01, 0.1, "scale"],
    # Selected values for parameter Gamma (Influence of a single training example)
    # 'scale' is an automatic default value
}"""

print("Searching hyperparameters...")
grid_search = GridSearchCV(
    estimator=meta_classifier,  # The model to be evaluated -> Pipeline (StandardScaler + SVC)
    param_grid=param_grid,  # The parameter combinations to be tested
    cv=5,  # Number of Folds for Cross-Validation, 5 is often considered an optimal value
    scoring="accuracy",  # Evaluation metric
    verbose=2,  # Display progress & status during the search, 2 = detailed output
    n_jobs=-1,  # Use all available CPU-cores
)

# ------------------ Training ------------------

print("Training model...")
grid_search.fit(X_train, y_train)  # Training the model

# ------------------ Save Model ------------------

best_model = (
    grid_search.best_estimator_
)  # Stores the model with the best parameters found by GridSearch
filename = (
    model_folder + "model.joblib"
)  # Filename for saving, given as second run argument

joblib.dump(best_model, filename)  # Serialize and save the model

print("Evaluating model...")
evaluation = None
# y_pred = best_model.predict(X_test)
# ---------------- serialize Evaluation Results ----------------
with open(model_folder + "evaluation.txt", "w") as f:
    f.write(str(evaluation))

print("All data saved to your model folder!")

# ------------------ Entry Point ------------------

if __name__ == "__main__":
    pass
