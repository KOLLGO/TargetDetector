# =============== IMPORTS =============== #

# General
import os
import sys

# Data processing
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import precision_recall_fscore_support
from scipy.sparse import save_npz

# GridSearch, K-Fold CV
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# Base models
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Meta classifier
from sklearn.ensemble import VotingClassifier, StackingClassifier

# Pipelining
from imblearn.pipeline import Pipeline

# Own functions
from features import get_model_matrices


# =============== DATA PREPARATION =============== #
csv_path = sys.argv[1]
model_folder = sys.argv[2]

CORES = -1  # Use all available CPU cores

if not model_folder.endswith("/"):
    model_folder += "/"

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

print("=============== MODEL TRAINING ===============")
print(f"Data file: {csv_path}")
print(f"Model folder: {model_folder}")
print()

# Get training feature matrix and labels
X_train, y_train = get_model_matrices(csv_path, model_folder)

# Get class distribution and largest class
print("Class distribution:")
unique, counts = np.unique(y_train, return_counts=True)

for u, c in zip(unique, counts):
    print(f"{u}: {c}")

max_count_index = np.argmax(counts)
largest_class = unique[max_count_index]
largest_count = counts[max_count_index]

print(f"Largest class: {largest_class}, count: {largest_count}")


# =============== GRIDSEARCH PARAMETERS =============== #

# SVC
svc_param_grid = {
    "svc__C": [0.1, 1, 3, 10, 30],
    "svc__gamma": [0.001, 0.003, 0.01, 0.1, "scale"],
    "svc__class_weight": ["balanced"],
}

# Logistic regression
log_param_grid = {
    "log__C": [0.1, 1, 3, 10],
    "log__solver": ["lbfgs", "newton-cg"],
    "log__class_weight": ["balanced"],
}

# MulitnomialNB
nb_param_grid = {
    "nb__alpha": [0.01, 0.1, 1],
    "nb__fit_prior": [True, False],
}

# RandomForest
rf_param_grid = {
    "rf__n_estimators": [100, 200, 300, 400, 500],
    "rf__max_depth": [None, 10, 20, 30, 40, 50],
    "rf__min_samples_split": [2, 5, 10],
    "rf__class_weight": ["balanced"],
}


# =============== OVERSAMPLING STRATEGY =============== #
seed = 42  # Oversampling seed

strategy = {
    "group": int(largest_count * 0.4),  # group
    "individual": int(largest_count * 0.6),  # individual
    "public": int(largest_count),  # public
}

# =============== PIPELINE CREATION =============== #

# SVC
svc_pipe = Pipeline(
    [
        ("scaler", MaxAbsScaler()),
        ("svc", SVC(kernel="rbf", probability=True)),
    ]
)

# Logistic Regression
log_pipe = Pipeline(
    [
        ("scaler", MaxAbsScaler()),
        ("log", LogisticRegression(max_iter=10000)),
    ]
)

# MultinomialNB
nb_pipe = Pipeline(
    [
        ("nb", MultinomialNB()),
    ]
)

# RandomForest
rf_pipe = Pipeline(
    [
        ("rf", RandomForestClassifier()),
    ]
)


# =============== EVALUATION METRICS =============== #

metrics_voting = {"precision": [], "recall": [], "f1": []}
metrics_stacking = {"precision": [], "recall": [], "f1": []}

metrics_base_models = {
    "svc": {"precision": [], "recall": [], "f1": []},
    "log": {"precision": [], "recall": [], "f1": []},
    "nb": {"precision": [], "recall": [], "f1": []},
    "rf": {"precision": [], "recall": [], "f1": []},
}

# Splits for K-Fold CV
inner_k = 3
outer_k = 5
k_base_models = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=seed)
k_meta_classifier = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=seed)

# Open results file for writing
results_file = open(model_folder + "training_results.txt", "w")
results_file.write("=============== MODEL TRAINING RESULTS ===============\n")
results_file.write(f"Inner K-Fold splits: {inner_k}\n")
results_file.write(f"Outer K-Fold splits: {outer_k}\n")
results_file.write(f"Seed: {seed}\n")
results_file.write("Oversampling_strategy: " + str(strategy) + "\n")

# =============== OUTER FOLD LOOP =============== #

for fold, (train_idx, val_idx) in enumerate(k_meta_classifier.split(X_train, y_train)):
    print(f"=============== Fold {fold+1} of 5 ===============")
    results_file.write(f"=============== Fold {fold+1} of 5 ===============\n")

    # Split for this fold
    X_train_fold = X_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_train_fold = y_train[train_idx]
    y_val_fold = y_train[val_idx]

    # GridSearchCV for SVC
    gs_svc = GridSearchCV(
        estimator=svc_pipe,
        param_grid=svc_param_grid,
        cv=k_base_models,
        scoring="f1_macro",
        verbose=2,
        n_jobs=CORES,
    )

    gs_svc.fit(X_train_fold, y_train_fold)
    best_svc = gs_svc.best_estimator_

    # GridSearchCV for Logistic Regression
    gs_log = GridSearchCV(
        estimator=log_pipe,
        param_grid=log_param_grid,
        cv=k_base_models,
        scoring="f1_macro",
        verbose=2,
        n_jobs=CORES,
    )

    gs_log.fit(X_train_fold, y_train_fold)
    best_log = gs_log.best_estimator_

    # GridSearchCV for MultinomialNB
    gs_nb = GridSearchCV(
        estimator=nb_pipe,
        param_grid=nb_param_grid,
        cv=k_base_models,
        scoring="f1_macro",
        verbose=2,
        n_jobs=CORES,
    )

    gs_nb.fit(X_train_fold, y_train_fold)
    best_nb = gs_nb.best_estimator_

    # GridSearchCV for RandomForest
    gs_rf = GridSearchCV(
        estimator=rf_pipe,
        param_grid=rf_param_grid,
        cv=k_base_models,
        scoring="f1_macro",
        verbose=2,
        n_jobs=CORES,
    )

    gs_rf.fit(X_train_fold, y_train_fold)
    best_rf = gs_rf.best_estimator_

    # =============== BASE MODEL EVALUATION (IN OUTER FOLD LOOP) =============== #
    for name, model in [
        ("svc", best_svc),
        ("log", best_log),
        ("nb", best_nb),
        ("rf", best_rf),
    ]:
        y_pred_base = model.predict(X_val_fold)
        p_bm, r_bm, f_bm, _ = precision_recall_fscore_support(
            y_val_fold, y_pred_base, average="macro", zero_division=0
        )

        # Save results
        metrics_base_models[name]["precision"].append(p_bm)
        metrics_base_models[name]["recall"].append(r_bm)
        metrics_base_models[name]["f1"].append(f_bm)

        msg = f"Macro values for {name}:\n"
        msg += f"Precision: {p_bm:.4f}\n"
        msg += f"Recall: {r_bm:.4f}\n"
        msg += f"F1 Score: {f_bm:.4f}\n"
        msg += f"=" * 50 + "\n"
        print(msg, end="")
        results_file.write(msg)

    # =============== VOTING CLASSIFIER AS META CLASSIFIER (IN OUTER FOLD LOOP) =============== #
    """
    base_models_vc = [
        ("svc", best_svc),
        ("log", best_log),
        ("naive_bayes", best_nb),
        ("rf", best_rf),
    ]

    # Define VotingClassifier
    voting_clf = VotingClassifier(
        estimators=base_models_vc, voting="soft", weights=[1, 1, 1, 1], n_jobs=CORES
    )

    # Train VotingClassifier on (resampled) train split for this fold
    voting_clf.fit(X_train_fold, y_train_fold)

    # Predict and evaluate on validation fold
    y_pred_vc = voting_clf.predict(X_val_fold)
    p_vc, r_vc, f_vc, _ = precision_recall_fscore_support(
        y_val_fold, y_pred_vc, average="macro", zero_division=0
    )

    # Save results for this fold
    metrics_voting["precision"].append(p_vc)
    metrics_voting["recall"].append(r_vc)
    metrics_voting["f1"].append(f_vc)

    msg = f"VotingClassifier macro results for fold {fold+1}:\n"
    msg += f"Precision: {p_vc:.4f}\n"
    msg += f"Recall: {r_vc:.4f}\n"
    msg += f"F1 Score: {f_vc:.4f}\n"
    msg += f"=" * 50 + "\n"

    print(msg, end="")
    results_file.write(msg)
    """

    # =============== STACKING CLASSIFIER AS META CLASSIFIER (IN OUTER FOLD LOOP) =============== #
    base_models_sc = [
        ("svc", best_svc),
        ("log", best_log),
        ("nb", best_nb),
        ("rf", best_rf),
    ]

    # Define StackingClassifier
    stacking_clf = StackingClassifier(
        estimators=base_models_sc,
        final_estimator=LogisticRegression(max_iter=1000),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=CORES,
        verbose=0,
    )

    # Train StackingClassifier on (resampled) train split for this fold
    stacking_clf.fit(X_train_fold, y_train_fold)

    # Predict and evaluate on validation fold
    y_pred_sc = stacking_clf.predict(X_val_fold)
    p_sc, r_sc, f_sc, _ = precision_recall_fscore_support(
        y_val_fold, y_pred_sc, average="macro", zero_division=0
    )

    # Save results for this fold
    metrics_stacking["precision"].append(p_sc)
    metrics_stacking["recall"].append(r_sc)
    metrics_stacking["f1"].append(f_sc)

    msg = f"StackingClassifier macro results for fold {fold+1}:\n"
    msg += f"Precision: {p_sc:.4f}\n"
    msg += f"Recall: {r_sc:.4f}\n"
    msg += f"F1 Score: {f_sc:.4f}\n"
    msg += f"=" * 50 + "\n"

    print(msg, end="")
    results_file.write(msg)
    results_file.write("\n")

# =============== FINAL AVERAGED RESULTS =============== #
print("=============== FINAL RESULTS ===============")
results_file.write("=============== FINAL RESULTS ===============\n")

# VotingClassifier macro results
"""
msg = f"VotingClassifier macro values:\n"
msg += f"    Average precision: {np.mean(metrics_voting['precision']):.4f} (+/- {np.std(metrics_voting['precision']):.4f})\n"
msg += f"    Average recall: {np.mean(metrics_voting['recall']):.4f} (+/- {np.std(metrics_voting['recall']):.4f})\n"
msg += f"    Average f1 score: {np.mean(metrics_voting['f1']):.4f} (+/- {np.std(metrics_voting['f1']):.4f})\n"

print(msg, end="")
results_file.write(msg)
"""

# StackingClassifier macro results
msg = f"StackingClassifier macro values:\n"
msg += f"    Average precision: {np.mean(metrics_stacking['precision']):.4f} (+/- {np.std(metrics_stacking['precision']):.4f})\n"
msg += f"    Average recall: {np.mean(metrics_stacking['recall']):.4f} (+/- {np.std(metrics_stacking['recall']):.4f})\n"
msg += f"    Average f1 score: {np.mean(metrics_stacking['f1']):.4f} (+/- {np.std(metrics_stacking['f1']):.4f})\n"

print(msg, end="")
results_file.write(msg)

# Base models results
msg = f"Base Models macro values (averaged across folds):\n"
for name in metrics_base_models:
    msg += f"\n{name.upper()}:\n"
    msg += f"    Average precision: {np.mean(metrics_base_models[name]['precision']):.4f} (+/- {np.std(metrics_base_models[name]['precision']):.4f})\n"
    msg += f"    Average recall: {np.mean(metrics_base_models[name]['recall']):.4f} (+/- {np.std(metrics_base_models[name]['recall']):.4f})\n"
    msg += f"    Average f1 score: {np.mean(metrics_base_models[name]['f1']):.4f} (+/- {np.std(metrics_base_models[name]['f1']):.4f})\n"

print(msg, end="")
results_file.write(msg)

results_file.close()
print(f"\nResults saved to: {model_folder}training_results.txt")
