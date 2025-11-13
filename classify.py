from calc import *
from sklearn.svm import svc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ------ Data Frame Zeugs -----

X = df["feature1", "feature 2", ...]
y = df["target1", "target2", ...]

# ------ Test/Train -------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# -------- dings -------

scaler = StandardScaler()
svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)

# ------ Pipeline -------

svm_model = Pipeline([
    ("scaler", scaler),
    ("svm", svm)
])


# ------- training ------

svm_model.fit(X_train, y_train)

if __name__ == "__main__":
    pass
