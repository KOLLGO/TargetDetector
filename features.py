import joblib
import pandas as pd
import re
from collections import Counter
from scipy.sparse import csr_matrix, hstack
from preprocessing import get_processed_dfs, data_handling
from pathlib import Path


# --------Feature Extraction---------
def extract_pronouns(df):
    """
    in: df
    out: df with pronouns features
    """
    pronouns_list = [
        "ich",
        "du",
        "er",
        "sie",
        "es",
        "wir",
        "ihr",
        "mich",
        "dich",
        "ihn",
        "uns",
        "euch",
        "mir",
        "dir",
        "ihm",
        "ihnen",
        "mein",
        "meine",
        "meiner",
        "meines",
        "meinem",
        "meinen",
        "dein",
        "deine",
        "deiner",
        "deines",
        "deinem",
        "deinen",
        "sein",
        "seine",
        "seiner",
        "seines",
        "seinem",
        "seinen",
        "ihr",
        "ihre",
        "ihrer",
        "ihres",
        "ihrem",
        "ihren",
        "unser",
        "unsere",
        "unserer",
        "unseres",
        "unserem",
        "unseren",
        "euer",
        "eure",
        "eurer",
        "eures",
        "eurem",
        "euren",
    ]  # List of all pronouns

    # create df + columns for pronouns
    df_pronouns = pd.DataFrame()
    df_pronouns["id"] = 0  # id column
    # all pronoun columns
    for pronoun in pronouns_list:
        df_pronouns[f"pronoun_{pronoun}"] = 0

    # loop through rows
    for idx, row in df.iterrows():
        id = row["id"]  # get id
        text = row["description"]  # get description

        # loop through pronouns
        for pronoun in pronouns_list:
            column_name = f"pronoun_{pronoun}"  # set column name
            count = text.lower().split().count(pronoun)  # count occurrences
            df_pronouns.loc[idx, "id"] = id  # place id in pronoun df
            df_pronouns.loc[idx, column_name] = count  # enter counts

    return df_pronouns


def extract_generics(df):
    """
    in: df
    out: df with generics features
    """
    generics_list = [
        "jeder",
        "alle",
        "leute",
        "man",
        "lieber",
        "liebe",
        "freunde",
        "gruppe",
        "jemand",
        "politik",
        "menschen",
        "gesellschaft",
        "gemeinschaft",
        "volk",
        "bürger",
        "welt",
        "nation",
        "bevölkerung",
        "der",
        "die",
        "das",
    ]  # list of generics
    df_generics = pd.DataFrame()
    df_generics["id"] = 0  # id column
    # all generics columns
    for generic in generics_list:
        df_generics[f"generic_{generic}"] = 0

    # loop through rows
    for idx, row in df.iterrows():
        id = row["id"]  # get id
        text = row["description"]  # get description

        # loop through generics
        for generic in generics_list:
            column_name = f"generic_{generic}"  # set column name
            count = text.lower().split().count(generic)  # count occurrences
            df_generics.loc[idx, "id"] = id  # place id in generic df
            df_generics.loc[idx, column_name] = count  # enter counts

    return df_generics


def extract_mentions(df):
    """
    in: df
    out: df with mentions features
    todo: maybe other mentions depending on final dataset
    """
    mentions_list = ["ind", "pre", "pol", "grp"]  # list of mentions
    df_mentions = pd.DataFrame()
    df_mentions["id"] = 0  # id column
    # all mentions columns
    for mention in mentions_list:
        df_mentions[f"mention_{mention}"] = 0

    # loop through rows
    for idx, row in df.iterrows():
        id = row["id"]  # get id
        text = row["description"]  # get description

        # loop through mentions
        for mention in mentions_list:
            column_name = f"mention_{mention}"  # set column name
            count = len(re.findall(rf"@{mention}\b", text))  # count occurrences
            df_mentions.loc[idx, "id"] = id  # place id in mention df
            df_mentions.loc[idx, column_name] = count  # enter counts

    return df_mentions


def extract_word_n_grams(df):
    """
    in: df
    out: df with word n-grams
    """

    # helper to produce n-grams from token list
    def ngrams(tokens, n):
        return [
            " ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
        ]  # ngram loop

    all_ngrams = set()
    rows_ngrams = []  # store per-row lists of ngrams

    for idx, row in df.iterrows():
        text = row.get("description_clean", row.get("description", ""))  # get text
        tokens = text.split()  # slice by space
        row_ngrams = ngrams(tokens, 2) + ngrams(tokens, 3)  # create ngrams
        rows_ngrams.append(row_ngrams)  # add to list
        all_ngrams.update(row_ngrams)  # update set

    # sort ngrams ascending
    all_ngrams = sorted(all_ngrams)

    # build df with ngram columns
    col_names = [
        f"ngram_{ng.replace(' ', '_')}" for ng in all_ngrams
    ]  # replace space for names
    df_ngrams = pd.DataFrame(
        0, index=df.index, columns=["id"] + col_names
    )  # add id for joining later

    # fill id column and counts per row (Foreign Code, by GitHub Copilot)
    for idx, row in df.iterrows():
        df_ngrams.at[idx, "id"] = row.get("id", idx)
        counter = Counter(rows_ngrams[idx])
        for ng, col in zip(all_ngrams, col_names):
            if counter.get(ng, 0):
                df_ngrams.at[idx, col] = counter[ng]

    return df_ngrams


# --------Feature Extraction Pipeline---------
def feature_extraction_pipeline(df):
    """
    in: df
    out: df with all features
    """
    # get all feature dfs
    df_pronouns = extract_pronouns(df)
    df_generics = extract_generics(df)
    df_mentions = extract_mentions(df)
    df_word_ngrams = extract_word_n_grams(df)  # not implemented yet

    # merge all feature dataframes using 'id'
    df_features = (
        df_pronouns.merge(df_generics, on="id")
        .merge(df_mentions, on="id")
        .merge(df_word_ngrams, on="id")
    )  # join all feature dfs
    return df_features


# --------Get Feature Matrices---------
def get_model_matrices(csv_path, vec_path):
    """
    in: csv file path
    out: dfs: X_train, y_train, X_test, y_test
    """
    _, X_train, X_test, X_train_tfidf, _, y_train, y_test = get_processed_dfs(
        csv_path, vec_path
    )  # function from preprocessing.py
    X_train = get_train_matrix(X_train, X_train_tfidf, vec_path)
    X_test = get_test_matrix(X_test, vec_path)
    return X_train, y_train, X_test, y_test


def get_train_matrix(X_train, X_train_tfidf, vec_path):
    """
    in: train df, train tfidf matrix, vectorizer path
    out: X_train
    """
    # convert feature Series to DFs (Foreign Code, by GitHub Copilot)
    df_train = pd.DataFrame({"id": range(len(X_train)), "description": X_train.values})
    # feature extraction pipeline for train data
    df_features_train = feature_extraction_pipeline(df_train)
    feature_names = list(df_features_train.columns)  # save feature column names
    joblib.dump(
        feature_names, vec_path + "feature_names.pkl"
    )  # Serialize feature column names
    # convert and combine features and tfidf to sparse matrix
    mat_features_train = csr_matrix(df_features_train.drop(columns=["id"]).values)
    X_train = hstack([mat_features_train, X_train_tfidf])
    return X_train


def get_test_matrix(X_test, vec_path):
    """
    in: test df, vectorizer path
    out: X_test
    """
    texts = None
    # decide how to extract texts based on data type
    # pandas Series (from split_data)
    if isinstance(X_test, pd.Series):
        texts = X_test.values
    # pandas DataFrame (from regular usage)
    elif isinstance(X_test, pd.DataFrame):
        texts = X_test["description_clean"].values

    # build feature df from texts
    df_features = pd.DataFrame({"id": range(len(texts)), "description": texts})
    df_features = feature_extraction_pipeline(df_features)

    # load saved feature column order and vectorizer
    feature_names_path = vec_path + "feature_names.pkl"
    vectorizer_path = vec_path + "tfidf_vectorizer.pkl"
    feature_names = joblib.load(feature_names_path)
    vectorizer = joblib.load(vectorizer_path)

    # remove 'id' and reindex df_features to match training columns
    feature_cols = [c for c in feature_names if c != "id"]
    df_features = df_features.reindex(
        columns=["id"] + feature_cols, fill_value=0
    )  # fill missing with 0

    # convert df features to sparse matrix without id
    mat_features_all = csr_matrix(df_features.drop(columns=["id"]).values)

    # build TF-IDF matrix using saved vectorizer
    tfidf_mat = vectorizer.transform(texts)

    # combine features (feature columns first, then TF-IDF columns)
    X_test = hstack([mat_features_all, tfidf_mat])
    return X_test


# --------- Build Feature Matrix for regular usage --------
def build_feature_matrix(csv_path, vec_path):
    """
    in:
        csv_path to csv with raw data
        vec_path: prefix used when training to save vectorizer and feature names
    out: sparse feature matrix
    """
    # load and preprocess data
    df_preprocessed = data_handling(csv_path)
    X_test = get_test_matrix(df_preprocessed, vec_path)

    return X_test
