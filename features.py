# Feature Extraction
import pandas as pd
import re
from scipy.sparse import hstack
from preprocessing import test_data_handling(), train_data_handling(), train_tfidf_matrix, test_tfidf_matrix

def extract_labels(df):
    """
    in: df
    out: df with labels
    """
    df_labels = df[["id", "TAR"]].copy()  # extract id and target columns
    return df_labels


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
        "euren"]  # List of all pronouns

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
        "das"
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
    pass


# --------Feature Extraction Pipeline---------
def feature_extraction_pipeline(df, tfidf_matrix):
    """
    in: df
    out: df with all features
    """
    # get all feature dfs
    df_labels = extract_labels(df)
    df_pronouns = extract_pronouns(df)
    df_generics = extract_generics(df)
    df_mentions = extract_mentions(df)
    df_word_ngrams = extract_word_n_grams(df)  # not implemented yet

    # Merge all feature dataframes on using 'id'
    df_features = df_pronouns.merge(df_generics, on="id").merge(
        df_mentions, on="id"
    ).merge(df_word_ngrams, on="id") # join all feature dfs
    df_features = df_features.merge(df_labels, on="id")  # add labels to last column
    return df_features

def get_train_data():
    """
    in: none
    out: dfs: X_train, y_train
    """
    df_features = feature_extraction_pipeline(train_data_handling())
    train_tfidf_matrix = pd.DataFrame  # function from preprocessing goes here
    X_train = hstack(df_features, train_tfidf_matrix.values)
    y_train = df_features["TAR"]
    return X_train, y_train


def get_test_data():
    """
    in: none
    out: dfs: X_test, y_test
    """
    df_features = feature_extraction_pipeline(test_data_handling())
    test_tfidf_matrix = pd.DataFrame  # function from preprocessing goes here
    X_test = hstack(df_features, test_tfidf_matrix.values)
    y_test = df_features["TAR"]
    return X_test, y_test
