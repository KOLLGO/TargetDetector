# ------------ Imports ------------

# General
import argparse
import os

# Data processing
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import precision_recall_fscore_support

# GridSearch, K-Fold CV
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# Base models
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Meta classifier
from sklearn.ensemble import StackingClassifier

# Pipelining
from imblearn.pipeline import Pipeline

# Own functions
from features import get_model_matrices


# ------------ Runtime Arguments ------------
parser = argparse.ArgumentParser()  # parser for command line arguments

# Input CSV File
parser.add_argument(
    "-i",  # Option 1
    "--input",  # Option 2
    type=str,  # Data Type
    help="Path to the CSV data file",  # Help Message
    required=True,  # is required
)

# Model Folder
parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="Folder to save the model components and results",
    required=True,
)

# Seed
parser.add_argument(
    "-s",
    "--seed",
    type=int,
    help="Seed for reproducibibility (default: None for random seed)",
    required=False,
    default=None,  # if not provided, use random seed
)

# Cores
parser.add_argument(
    "-c",
    "--cores",
    type=int,
    help="Number of CPU cores to use (default: -1 for all available cores)",
    required=False,
    default=-1,  # if not provided, use all availadble cores
)

args = parser.parse_args()  # parse arguments from command line

# assign values
csv_path: str = args.input
model_folder: str = args.model
seed: int = args.seed
CORES: int = args.cores

# ------------ Data preparation ------------
if not model_folder.endswith("/"):  # waterproofing model folder path
    model_folder += "/"

if not os.path.exists(model_folder):  # if model folder not existing, create it
    os.makedirs(model_folder)

# Print Config Info
print(f"-" * 50 + "Model Training" + f"-" * 50 + "\n")
print(f"Data file: {csv_path}")
print(f"Model folder: {model_folder}")
print()

# Get training feature matrix and labels from features.py
X_train, y_train = get_model_matrices(csv_path, model_folder)

# ------------ Grid Search Parameters ------------

# SVC
svc_param_grid = {
    "svc__C": [
        0.1, # lower - stronger (underfitting)
        1,
        3,
        10,
        30 # higher - weaker (overfitting)
    ],  # regularization parameter
    "svc__gamma": [
        0.001, # lower - far
        0.003,
        0.01,
        0.1, # higher - close
        "scale" # scale - auto-fit to data
    ],  # influence of single training examples
    "svc__class_weight": ["balanced"]  # handle class imbalance
}

# Logistic Regression
log_param_grid = {
    "log__C": [0.1, 1, 3, 10], # see SVC C parameter
    "log__solver": [
        "lbfgs", # general purpose
        "newton-cg" # multiclass or larger datasets
        ], # optimization algorithm
    "log__class_weight": ["balanced"] # handle class imbalance
}

# MulitnomialNB
nb_param_grid = {
    "nb__alpha": [
        0.01, # less smoothing
        0.1, 
        1 # standard smooting
        ], # smooting parameter against zero probabilities
    "nb__fit_prior": [True, False] # prior from class distribution (True) or uniform (False)
}

# RandomForest
rf_param_grid = {
    "rf__n_estimators": [
        100, # faster but less accurate 
        200, 
        300, 
        400, 
        500 # slower but more accurate
        ],# tree count
    "rf__max_depth": [
        None, # no limit, possible overfitting
        10, # less depth, more generalization, possible underfitting 
        20,
        30, 
        40, 
        50
        ], # max tree depth
    "rf__min_samples_split": [
        2, # small, more specific, possible overfitting
        5, 
        10 # larger, more general, possible underfitting
        ], # min samples to split a node
    "rf__class_weight": ["balanced"] # handle class imbalance
}


# ------------ Pipeline Initialization ------------

# SVC
svc_pipe = Pipeline(
    [
        ("scaler", MaxAbsScaler()), # scale features to handle for SVC
        ("svc", SVC(
                kernel="rbf", # radial basis 
                probability=True # probability instead of class labels (more info for stacking clf)
                )), # base model
    ]
)

# Logistic Regression
log_pipe = Pipeline(
    [
        ("scaler", MaxAbsScaler()), # scale features to handle for Logistic Regression
        ("log", LogisticRegression(max_iter=10000)), # base model, more iterations
    ]
)

# MultinomialNB
nb_pipe = Pipeline(
    [
        ("nb", MultinomialNB()), # base model, no scaling needed (negative values not allowed)
    ]
)

# RandomForest
rf_pipe = Pipeline(
    [
        ("rf", RandomForestClassifier()), # base model, no scaling needed (tree-based)
    ]
)


# ------------ Evaluation Metrics ------------
metrics_stacking: dict[str, list] = {"precision": [], "recall": [], "f1": []} # metrics storage (meta classifier)

metrics_base_models: dict[str, dict[str, list]] = {
    "svc": {"precision": [], "recall": [], "f1": []},
    "log": {"precision": [], "recall": [], "f1": []},
    "nb": {"precision": [], "recall": [], "f1": []},
    "rf": {"precision": [], "recall": [], "f1": []},
} # metrics storage (base models)

# ------------ K-Fold Cross-Validation ------------
# Splits for K-Fold CV
inner_k: int = 3
outer_k: int = 5
k_base_models = StratifiedKFold(
    n_splits=inner_k, # folds
    shuffle=True, # shuffle data randomly before splitting
    random_state=seed # seed for reproducibility
    ) # base model for k-Fold-CV, stratified - keep class distribution
k_meta_classifier = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=seed)# see above

# Open results file for saving
results_file = open(model_folder + "training_results.txt", "w")
# write session info
results_file.write(f"-" * 50 + "Training Results" + f"-" * 50 +"\n")
results_file.write(f"Inner K-Fold splits: {inner_k}\n")
results_file.write(f"Outer K-Fold splits: {outer_k}\n")
results_file.write(f"Seed: {seed}\n")

# ------------ Outer Fold Loop ------------
for fold, (train_idx, val_idx) in enumerate(k_meta_classifier.split(X_train, y_train)):
    print(f"-"*50 + f"Fold {fold+1} of 5" + f"-"*50 + "\n")
    results_file.write(f"-"*50 + f"Fold {fold+1} of 5" + f"-"*50 + "\n")

    # Split for this fold
    X_train_fold = X_train[train_idx] # train samples
    X_val_fold = X_train[val_idx] # test samples
    y_train_fold = y_train[train_idx] # train classes
    y_val_fold = y_train[val_idx] # test classes

    # GridSearchCV for SVC
    gs_svc = GridSearchCV(
        estimator=svc_pipe, # use model pipeline defined earlier
        param_grid=svc_param_grid, # use param grid defined earlier
        cv=k_base_models, # use cv-model defined earlier
        scoring="f1_macro", # metric to optimize
        verbose=2, # more info on processes
        n_jobs=CORES # CPU core count
    ) # hyperparameter search

    gs_svc.fit(X_train_fold, y_train_fold) # train model
    best_svc = gs_svc.best_estimator_ # select best model to put into meta classifier

    # GridSearchCV for Logistic Regression (see above)
    gs_log = GridSearchCV(
        estimator=log_pipe,
        param_grid=log_param_grid,
        cv=k_base_models,
        scoring="f1_macro",
        verbose=2,
        n_jobs=CORES
    )

    gs_log.fit(X_train_fold, y_train_fold)
    best_log = gs_log.best_estimator_

    # GridSearchCV for MultinomialNB (see above)
    gs_nb = GridSearchCV(
        estimator=nb_pipe,
        param_grid=nb_param_grid,
        cv=k_base_models,
        scoring="f1_macro",
        verbose=2,
        n_jobs=CORES
    )

    gs_nb.fit(X_train_fold, y_train_fold)
    best_nb = gs_nb.best_estimator_

    # GridSearchCV for RandomForest (see above)
    gs_rf = GridSearchCV(
        estimator=rf_pipe,
        param_grid=rf_param_grid,
        cv=k_base_models,
        scoring="f1_macro",
        verbose=2,
        n_jobs=CORES
    )

    gs_rf.fit(X_train_fold, y_train_fold)
    best_rf = gs_rf.best_estimator_

    # ------------ Base Models Evaluation ------------
    for name, model in [
        ("svc", best_svc),
        ("log", best_log),
        ("nb", best_nb),
        ("rf", best_rf),
    ]: # loop through every model
        y_pred_base = model.predict(X_val_fold) # predict test data with model
        p_bm, r_bm, f_bm, _ = precision_recall_fscore_support(
            y_val_fold, # true classes
            y_pred_base, # classes predicted by model
            average="macro", # use macro average
            zero_division=0 # if division by 0, return 0
        ) # support ignored by _

        # Save results
        metrics_base_models[name]["precision"].append(p_bm) # put precision into storage
        metrics_base_models[name]["recall"].append(r_bm) # recall
        metrics_base_models[name]["f1"].append(f_bm) # f1

        msg = f"Macro values for {name}:\n"
        msg += f"Precision: {p_bm:.4f}\n" # :.4f rounds to 4 decimal digits
        msg += f"Recall: {r_bm:.4f}\n"
        msg += f"F1 Score: {f_bm:.4f}\n"
        msg += f"-" * 100 + "\n"
        print(msg, end="") # console output, end "" so no extra empty line
        results_file.write(msg) # write to output file

    # ------------ Stacking Classifier as Meta Classifier ------------
    base_models_sc = [
        ("svc", best_svc),
        ("log", best_log),
        ("nb", best_nb),
        ("rf", best_rf),
    ] # base model storage

    # Define StackingClassifier
    stacking_clf = StackingClassifier(
        estimators=base_models_sc, # base model storage
        final_estimator=LogisticRegression(max_iter=1000), # use logistic regression for stacking
        stack_method="predict_proba", # use probabilities
        passthrough=False, # only outputs of base models
        n_jobs=CORES,# use all available cores
        verbose=0, # no info
    )
    stacking_clf.fit(X_train_fold, y_train_fold) # train meta classifier

    # ------------ Stacking Classifier Evaluation ------------
    y_pred_sc = stacking_clf.predict(X_val_fold) # predict test data
    p_sc, r_sc, f_sc, _ = precision_recall_fscore_support(
        y_val_fold, y_pred_sc, average="macro", zero_division=0
    ) # see above

    # Save results for this fold
    metrics_stacking["precision"].append(p_sc) # add values to stacking classifer storage
    metrics_stacking["recall"].append(r_sc)
    metrics_stacking["f1"].append(f_sc)

    msg = f"Stacking Classifier (Meta) macro results for fold {fold+1}:\n"
    msg += f"Precision: {p_sc:.4f}\n"
    msg += f"Recall: {r_sc:.4f}\n"
    msg += f"F1 Score: {f_sc:.4f}\n"

    print(msg, end="")
    results_file.write(msg)
    results_file.write("\n")

# ------------ Final Evaluation Results ------------
print(f"-"*50 + "Final Results" + f"-"*50 + "\n")
results_file.write(f"-"*50 + "Final Results" + f"-"*50 + "\n")

# Stacking Classifier macro results (see above)
msg = f"StackingClassifier macro values:\n"
msg += f"    Average precision: {np.mean(metrics_stacking['precision']):.4f} (+/- {np.std(metrics_stacking['precision']):.4f})\n" # mean average of macros
msg += f"    Average recall: {np.mean(metrics_stacking['recall']):.4f} (+/- {np.std(metrics_stacking['recall']):.4f})\n"
msg += f"    Average f1 score: {np.mean(metrics_stacking['f1']):.4f} (+/- {np.std(metrics_stacking['f1']):.4f})\n"

print(msg, end="")
results_file.write(msg)

# Base models results
msg = f"Base Models macro values (averaged across folds):\n"
for name in metrics_base_models:
    msg += f"\n{name.upper()}:\n"
    msg += f"    Average precision: {np.mean(metrics_base_models[name]['precision']):.4f} (+/- {np.std(metrics_base_models[name]['precision']):.4f})\n" # mean average of macros
    msg += f"    Average recall: {np.mean(metrics_base_models[name]['recall']):.4f} (+/- {np.std(metrics_base_models[name]['recall']):.4f})\n" # np.std: show standard deviance
    msg += f"    Average f1 score: {np.mean(metrics_base_models[name]['f1']):.4f} (+/- {np.std(metrics_base_models[name]['f1']):.4f})\n"

print(msg, end="")
results_file.write(msg)

results_file.close()
print(f"\nResults saved to: {model_folder}training_results.txt")
