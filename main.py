import datetime
import sys
import os
import joblib
import pandas as pd
from features import build_feature_matrix

if __name__ == "__main__":
    # --------------- get command line arguments ---------------
    model_path_str = sys.argv[1]  # model path from command line
    file_path = sys.argv[2]  # data file from command line
    if not model_path_str.endswith("/"):
        model_path_str += "/"

    print("----------------- Prediction -----------------")
    print("model folder: " + model_path_str)
    print("data file: " + file_path)

    # --------------- load model and predict ---------------
    print("Loading model...")
    model = joblib.load(model_path_str + "model.joblib")  # load model from path
    X_test = build_feature_matrix(
        file_path, model_path_str
    )  # build feature matrix from data file
    print("Predicting...")
    predictions = model.predict(X_test)  # predict data

    # --------------- save predictions to file ---------------
    if not os.path.exists("./outputs"):
        os.makedirs("./outputs")
    df_in = pd.read_csv(file_path, sep=";")
    df_in["prediction"] = predictions
    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df_in.to_csv(f"./outputs/{date}.csv", sep=";")
    print(f"Predictions saved to ./outputs/{date}.csv!")
