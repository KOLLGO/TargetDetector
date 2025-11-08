# Feature Extraction
import pandas as pd

# --------Pseudo DataFrame---------
df = pd.read_csv(
    "tar.csv", sep=";", nrows=50, on_bad_lines="skip", na_filter=True
)  # first 50 rows
print(df.head())


# --------Feature Extraction---------
def extract_pronouns(df):
    """
    in: df
    current out: df with pronouns features
    desired out: feature vector with counts of pronouns
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
    out: feature vector with counts of generics
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
    pass


def extract_mentions(df):
    """
    in: df
    out: feature vector with counts of mentions
    """
    pass


def extract_word_n_grams(df):
    """
    in: df
    out: word n-grams
    """
    pass


print(extract_pronouns(df))
