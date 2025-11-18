# ------------------ Imports ------------------

from calc import *  # Stelle sicher, dass df hier definiert ist
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from features import get_feature_matrices
import joblib

# ------------------ Daten vorbereiten ------------------

X_train, y_train, X_test, y_test = get_feature_matrices() # Uebergabe Feature-Matrix

# ------------------ Pipeline erstellen ------------------

pipeline = Pipeline([
    ("scaler", StandardScaler()),   # standartisiert Werte
    ("svc", SVC(kernel="rbf"))    # SVC = Support Vector Classifyer, radial basis function Kernel
])

# ------------------ Parameteroptimierung ------------------

param_grid = {
    'svc__C': [0.1, 1, 10],     # gewaehlte Werte fuer Parameter C (Bestrafung Fehlklassifikation)
    'svc__gamma': [0.001, 0.01, 0.1, 'scale']
    # gewaehlte Werte fuer Parameter Gamma (Einfluss einzelnes Trainingsbeispiel)
    # 'scale' ist ein automatischer Standardwert von 0,22
}

grid_search = GridSearchCV(
    estimator=pipeline,  # zu bewertende Modell -> Pipeline (Standardsclaer + SVC)
    param_grid=param_grid,  # geteste Parameterkombination
    cv=5,  # Anzahl Folds Cross-Validation, 5 als optimaler Wert
    scoring='accuracy',  # Bewertungsmetrik
    verbose=1  # Anzeige Fortschritt & Statuts während Suche, 1 = einfache Ausgabe
)

# ------------------ Training ------------------

grid_search.fit(X_train, y_train)  # Training

# ------------------ Modell speichern ------------------

best_model = grid_search.best_estimator_    # speichert Model mit besten Paramtern aus GridSearch
filename = "model.joblib"  # Dateiname für Speicherung
joblib.dump(best_model, filename)  # Serialisierung Model 

# ------------------ Einstiegspunkt ------------------

if __name__ == "__main__":
    pass
