import joblib
import pandas as pd
import re
from collections import Counter
from scipy.sparse import csr_matrix, hstack
from preprocessing import get_processed_dfs, data_handling


# --------Feature Extraction---------
def extract_pronouns(df: pd.DataFrame, pronouns_list: list[str]):
    """
    in: df
    out: df with pronouns features
    """
    print("Extracting pronouns...")
    # create df + columns for pronouns
    df_pronouns = pd.DataFrame()
    df_pronouns["id"] = 0  # id column
    # all pronoun columns
    for pronoun in pronouns_list:
        df_pronouns[f"pronoun_{pronoun}"] = 0

    # loop through rows
    for idx, row in df.iterrows():
        id = row["id"]  # get id
        text = row["description_clean"]  # get description

        # loop through pronouns
        for pronoun in pronouns_list:
            column_name = f"pronoun_{pronoun}"  # set column name
            count = text.lower().split().count(pronoun)  # count occurrences
            df_pronouns.loc[idx, "id"] = id  # place id in pronoun df
            df_pronouns.loc[idx, column_name] = count  # enter counts

    return df_pronouns


def extract_generics(df: pd.DataFrame, generics_list: list[str]):
    """
    in: df
    out: df with generics features
    """
    print("Extracting generics...")
    df_generics = pd.DataFrame()
    df_generics["id"] = 0  # id column
    # all generics columns
    for generic in generics_list:
        df_generics[f"generic_{generic}"] = 0

    # loop through rows
    for idx, row in df.iterrows():
        id = row["id"]  # get id
        text = row["description_clean"]  # get description

        # loop through generics
        for generic in generics_list:
            column_name = f"generic_{generic}"  # set column name
            count = text.lower().split().count(generic)  # count occurrences
            df_generics.loc[idx, "id"] = id  # place id in generic df
            df_generics.loc[idx, column_name] = count  # enter counts

    return df_generics


def extract_mentions(df: pd.DataFrame):
    """
    in: df
    out: df with mentions features
    todo: maybe other mentions depending on final dataset
    """
    print("Extracting mentions...")
    mentions_list = ["ind", "pre", "pol", "grp"]  # list of mentions
    df_mentions = pd.DataFrame()
    df_mentions["id"] = 0  # id column
    # all mentions columns
    for mention in mentions_list:
        df_mentions[f"mention_{mention}"] = 0

    # loop through rows
    for idx, row in df.iterrows():
        id = row["id"]  # get id
        text = row["description_clean"]  # get description

        # loop through mentions
        for mention in mentions_list:
            column_name = f"mention_{mention}"  # set column name
            count = len(re.findall(rf"@{mention}\b", text))  # count occurrences
            df_mentions.loc[idx, "id"] = id  # place id in mention df
            df_mentions.loc[idx, column_name] = count  # enter counts

    return df_mentions


def extract_word_n_grams(df: pd.DataFrame, words_of_interest: list[str]):
    """
    in: df
    out: df with word n-grams
    """
    print("Extracting word n-grams...")
    # Parameters: keep top K n-grams separately for bigrams and trigrams
    TOP_BIGRAMS = 500
    TOP_TRIGRAMS = 100

    # helper to produce n-grams from token list
    def ngrams(tokens, n):
        return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    # init counters and storage
    bigram_counter = Counter()
    trigram_counter = Counter()
    rows_bigrams = []
    rows_trigrams = []
    interest_set: set[str] = {w.lower() for w in words_of_interest}

    for idx, row in df.iterrows():
        text = row.get(
            "description_clean"  # , row.get("description_clean", "")
        )  # get cleaned text
        tokens = str(text).strip().split()  # tokenize by words

        # generate bigrams and trigrams per row
        bigrams_all = ngrams(tokens, 2) if len(tokens) >= 2 else []
        trigrams_all = ngrams(tokens, 3) if len(tokens) >= 3 else []

        # filter n-grams by words of interest
        bigrams = [
            bigram
            for bigram in bigrams_all
            if any(word in interest_set for word in bigram.split())
        ]
        trigrams = [
            trigram
            for trigram in trigrams_all
            if any(word in interest_set for word in trigram.split())
        ]

        # add to storage
        rows_bigrams.append(bigrams)
        rows_trigrams.append(trigrams)

        # update counters
        bigram_counter.update(bigrams)
        trigram_counter.update(trigrams)

    # pick top-k most common n-grams
    top_bigrams = [ng for ng, _ in bigram_counter.most_common(TOP_BIGRAMS)]
    top_trigrams = [ng for ng, _ in trigram_counter.most_common(TOP_TRIGRAMS)]

    # build column names
    bigram_cols = [f"ngram_bi_{bg.replace(' ', '_')}" for bg in top_bigrams]
    trigram_cols = [f"ngram_tri_{tg.replace(' ', '_')}" for tg in top_trigrams]
    col_names = bigram_cols + trigram_cols

    # prepare dataframe with zeros
    df_ngrams = pd.DataFrame(0, index=df.index, columns=["id"] + col_names)

    # fill id and counts per row (fixed by GitHub Copilot)
    for idx, row in df.iterrows():
        df_ngrams.at[idx, "id"] = row.get("id", idx)
        b_counter = Counter(rows_bigrams[idx])
        t_counter = Counter(rows_trigrams[idx])

        # top k
        for bigram, col in zip(top_bigrams, bigram_cols):
            df_ngrams.at[idx, col] = b_counter.get(bigram, 0)
        for trigram, col in zip(top_trigrams, trigram_cols):
            df_ngrams.at[idx, col] = t_counter.get(trigram, 0)

    return df_ngrams


# --------Feature Extraction Pipeline---------
def feature_extraction_pipeline(df):
    """
    in: df
    out: df with all features
    """
    pronouns_list = [
        "du",
        "er",
        "sie",
        "es",
        "wir",
        "ihr",
        "dich",
        "ihn",
        "uns",
        "euch",
        "dir",
        "ihm",
        "ihnen",
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
        "pegida",
        "bevölkerung",
        "der",
        "die",
        "das",
    ]  # list of generics

    words_of_interest = pronouns_list + generics_list
    # get all feature dfs
    df_pronouns = extract_pronouns(df, pronouns_list)
    df_generics = extract_generics(df, generics_list)
    df_mentions = extract_mentions(df)
    df_word_ngrams = extract_word_n_grams(df, words_of_interest)  # not implemented yet

    # merge all feature dataframes using 'id'
    df_features = (
        df_pronouns.merge(df_generics, on="id")
        .merge(df_mentions, on="id")
        .merge(df_word_ngrams, on="id")
    )  # join all feature dfs
    return df_features


# --------Get Feature Matrices---------
def get_model_matrices(csv_path: str, vec_path: str):
    """
    in: csv file path, verctorizer save path
    out: X_train, y_train
    """
    print("Preprocessing data...")
    X_train, X_train_tfidf = get_processed_dfs(csv_path, vec_path)
    y_train = X_train["TAR"]  # get labels
    print("Starting feature extraction...")
    df_features_train = feature_extraction_pipeline(X_train)
    feature_names: list[str] = list(
        df_features_train.columns
    )  # save feature column names
    joblib.dump(
        feature_names, vec_path + "feature_names.pkl"
    )  # Serialize feature column names
    # convert and combine features and tfidf to sparse matrix
    mat_features_train = csr_matrix(df_features_train.drop(columns=["id"]).values)
    X_train: csr_matrix = hstack([mat_features_train, X_train_tfidf])
    return X_train, y_train


if __name__ == "__main__":
    X_train, y_train = get_model_matrices("../tar.csv", "./model")
    print(X_train)
    print(y_train)
