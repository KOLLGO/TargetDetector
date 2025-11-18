import sys
import joblib
from features import build_feature_matrix
from pathlib import Path

if __name__ == "__main__":
    model_path = Path(sys.argv[1])  # model path from command line
    file_path = sys.argv[2]  # data file from command line
    model = joblib.load(model_path)  # load model from path
    X_test = build_feature_matrix(file_path)  # build feature matrix from data file
    predictions = model.predict(X_test)  # predict data
    print(predictions)
