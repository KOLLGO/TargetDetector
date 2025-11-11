# Feature Extraction
import pandas as pd
import re

# --------DataFrame---------
df = pd.read_csv(
    "../tar.csv", sep=";", nrows=50, on_bad_lines="skip", na_filter=True
)  # first 50 rows for testing
df["description"] = df["description"].str.lower()  # lowercase descriptions
print(df.head())


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
    pronouns_list = ["ich", "du", "er", "sie", "wir", "ihr", "es"]  # List of pronouns

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
        "euch",
        "uns",
        "dir",
        "ihnen",
        "mir",
        "freunde",
        "gruppe",
        "jemand",
        "politik",
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
def feature_extraction_pipeline(df):
    """
    in: df
    out: df with all features
    """
    # get all feature dfs
    df_labels = extract_labels(df)
    df_pronouns = extract_pronouns(df)
    df_generics = extract_generics(df)
    df_mentions = extract_mentions(df)

    # df_word_ngrams = extract_word_n_grams(df)  # not implemented yet

    # Merge all feature dataframes on using 'id'
    df_features = df_pronouns.merge(df_generics, on="id").merge(
        df_mentions, on="id"
    )  # ToDo: add word n-grams when implemented
    df_features = df_features.merge(df_labels, on="id")  # add labels to last column
    return df_features


print(feature_extraction_pipeline(df))
