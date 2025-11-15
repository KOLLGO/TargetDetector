# ------------------ Imports ------------------

from calc import *  # Stelle sicher, dass df hier definiert ist
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# ------------------ Daten vorbereiten ------------------

X = df[["feature1", "feature2"]]  # Passe die Feature-Namen an
y = df["target"]                  # Passe die Zielvariable an

# ------------------ Pipeline erstellen ------------------

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="rbf", probability=True))
])

# ------------------ Parameteroptimierung ------------------

param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__gamma': [0.001, 0.01, 0.1, 'scale']  # 'scale' ist ein automatischer Standardwert von 0,22
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1
)

# ------------------ Training ------------------

grid_search.fit(X_train, y_train)

# ------------------ Modell speichern ------------------

filename = "grid_search.joblib"
joblib.dump(grid_search, filename)

# ------------------ Modell laden (optional) ------------------

# grid_search = joblib.load("grid_search.joblib")
# best_model = grid_search.best_estimator_
# predictions = best_model.predict(X_test)

# ------------------ Einstiegspunkt ------------------

if __name__ == "__main__":
    pass
