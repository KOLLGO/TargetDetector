import os
import sys
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler
from imblearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support
from imblearn.over_sampling import RandomOverSampler
from features import get_model_matrices


# ------------------ Prepare Data ------------------

csv_path = sys.argv[1]
model_folder = sys.argv[2]

if not model_folder.endswith("/"):
    model_folder += "/"

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

print("----------------- Model Training -----------------")
print("data file: " + csv_path)
print("model folder: " + model_folder)
print()

X_train, y_train = get_model_matrices(
    csv_path, model_folder
)  # Get the training feature matrix and labels

# ------------------ Create Pipelines ------------------
oversampling_seed: int = 42
svc_pipeline = Pipeline(
    [
        ("scaler", MaxAbsScaler()),  # scales the values
        (
            "sampler",
            RandomOverSampler(random_state=oversampling_seed),
        ),  # handle class imbalance
        (
            "svc",
            SVC(kernel="rbf", probability=True),
        ),  # SVC = Support Vector Classifier, using the radial basis function (RBF) Kernel
    ]
)

naive_bayes_pipeline = Pipeline(
    [
        ("scaler", MaxAbsScaler()),
        (
            "sampler",
            RandomOverSampler(random_state=oversampling_seed),
        ),  # handle class imbalance
        ("classifier", MultinomialNB()),
    ]
)

logistic_pipeline = Pipeline(
    [
        ("scaler", MaxAbsScaler()),
        (
            "sampler",
            RandomOverSampler(random_state=oversampling_seed),
        ),  # handle class imbalance
        ("classifier", LogisticRegression(max_iter=10000)),
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


# ------------------ K-Fold CV ------------------
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

param_grid = {
    "svc__svc__C": [0.1, 1, 10],
    "svc__svc__gamma": [0.001, 0.01, 0.1, "scale"],
    "naive_bayes__classifier__alpha": [1.0, 0.1, 0.01],
    "logistic__classifier__C": [0.1, 1, 10],
    "logistic__classifier__solver": ["lbfgs", "newton-cg"],
    "voting": ["soft", "hard"],
}

fold_metrics = {"precision": [], "recall": [], "f1": []}

for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train)):
    print(f"----------------Outer Fold--------------\n {fold + 1}/5")

    # Split for this fold
    X_train_fold = X_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_train_fold = y_train[train_idx]
    y_val_fold = y_train[val_idx]

    print(f"Searching hyperparameters  for fold {fold + 1}...")
    grid_search = GridSearchCV(
        estimator=meta_classifier,  # The model to be evaluated -> Pipeline (StandardScaler + SVC)
        param_grid=param_grid,  # The parameter combinations to be tested
        cv=inner_cv,  # Number of Folds for Cross-Validation, 5 is often considered an optimal value
        scoring="f1_macro",  # Evaluation metric
        verbose=2,  # Display progress & status during the search, 2 = detailed output
        n_jobs=-1,  # Use all available CPU-cores
    )
    grid_search.fit(X_train_fold, y_train_fold)
    meta_classifier = grid_search.best_estimator_
    # Predictions
    y_pred = meta_classifier.predict(X_val_fold)

    # macro average
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val_fold, y_pred, average="macro", zero_division=0
    )

    # save results
    fold_metrics["precision"].append(precision)
    fold_metrics["recall"].append(recall)
    fold_metrics["f1"].append(f1)

    print(f"\nFold {fold + 1} Results:")
    print(f"  Precision (macro): {precision:.4f}")
    print(f"  Recall (macro):    {recall:.4f}")
    print(f"  F1-Score (macro):  {f1:.4f}")

# ------------------ final averaged results ------------------
print("--------------- final results ---------------")
print(f"{'='*50}")
print(
    f"Average Precision (macro): {np.mean(fold_metrics['precision']):.4f} (+/- {np.std(fold_metrics['precision']):.4f})"
)
print(
    f"Average Recall (macro):    {np.mean(fold_metrics['recall']):.4f} (+/- {np.std(fold_metrics['recall']):.4f})"
)
print(
    f"Average F1-Score (macro):  {np.mean(fold_metrics['f1']):.4f} (+/- {np.std(fold_metrics['f1']):.4f})"
)

print("\nPer-Fold Results:")
for i in range(5):
    print(
        f"Fold {i+1}: P={fold_metrics['precision'][i]:.4f}, R={fold_metrics['recall'][i]:.4f}, F1={fold_metrics['f1'][i]:.4f}"
    )


"""# ------------------ Training ------------------

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

print("All data saved to your model folder!")"""

# ------------------ Entry Point ------------------1

if __name__ == "__main__":
    pass
