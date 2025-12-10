# ------------ Imports ------------
# Serializaton
import joblib

# Data processing
import pandas as pd
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------ Load Spacy Model ------------
try:
    nlp = spacy.load("de_core_news_lg")  # loads german language model
except OSError:
    from spacy.cli import download  # download model if not present

    download("de_core_news_lg")
    nlp = spacy.load("de_core_news_lg")


# ------------ Preprocessing Pipeline ------------
def data_handling(file: str):
    """
    in: csv file path
    out: cleaned dataframe
    """

    # open csv file
    df_data: pd.DataFrame = pd.read_csv(file, sep=";")
    # df_data = df_data.head(100)  # optional limit for faster testing

    # new column for cleaned description + convert to lowercase
    df_data["description_clean"] = df_data["description"].str.lower()

    # whitelist of pronouns to keep
    whitelist: set[str] = {
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
    }

    # function to remove stopwords
    def remove_stopwords(description: str):
        """
        in: description text
        out: description text without stopwords
        """

        doc = nlp(description)  # converts text to spacy-doc object with tokens
        cleaned_tokens: list[str] = [
            token.text
            for token in doc  # for every token in doc
            if not token.is_stop
            or token.text in whitelist  # keep token only if its not a stopword or is whitelisted
        ]
        return " ".join(cleaned_tokens)  # joins tokens back to string

    # remove hyperlinks
    def remove_hyperlinks(description: str):
        """
        in: description text
        out: description text without hyperlinks
        """
        return re.sub(r"http[s]?://\S+", "", description) # regex for link structure, sub with empty string

    # function to remove special characters
    def remove_special_chars(description: str):
        """
        in: description text
        out: description text without special characters
        """
        # map umlauts to their replacements
        umlaut_map: dict[str, str] = {
            "ä": "ae",
            "ö": "oe",
            "ü": "ue",
            "ß": "ss",
        }

        # replace all umlauts
        for umlaut, replacement in umlaut_map.items():
            description = description.replace(
                umlaut, replacement
            )  # replace umlaut with replacement

        description = re.sub(r",(?=\S)", ", ", description)  # space after commas
        description = re.sub(r"\.(?=\S)", ". ", description)  # space after dot

        description = re.sub(
            r"[^a-z\s@]", "", description
        )  # remove non-alphanumeric characters except spaces
        description = re.sub(
            r"\s+", " ", description
        )  # replace multiple spaces with single space
        return description.strip()  # remove leading/trailing spaces

    # apply preprocessing functions
    df_data["description_clean"] = (
        df_data["description_clean"]
        .apply(remove_hyperlinks)
        .apply(remove_special_chars)
        .apply(remove_stopwords)
    )

    return df_data


# ------------ TF-IDF Feature Engineering ------------
def tfidf_vectorizer(df: pd.DataFrame, filepath: str):
    """
    in: df - cleaned DataFrame
        filepath - path to save the vectorizer
    out: X_train_tfidf - TF-IDF transformed train features (sparse matrix)
    """

    vectorizer: TfidfVectorizer = TfidfVectorizer(
        max_features=10000,  # limit feature space dimensionality
        min_df=1,  # mininmal document frequency for a token to be included (could be increased to avoid overfitting)
        sublinear_tf=True,  # logarithmic scaling for better for better weighting
        norm="l2",  # L2 normalization
        stop_words=None,  # doesnt exlude more stopwords
    )

    X_train_tfidf = vectorizer.fit_transform(
        df["description_clean"]
    )  # fit and transform train data

    print("Train shape:", X_train_tfidf.shape)  # print dimensions of tf-idf matrices

    joblib.dump(
        vectorizer, filepath + "tfidf_vectorizer.pkl"
    )  # serialize vectorizer
    return X_train_tfidf


# ------------ Data to import for features ------------
def get_processed_dfs(csv_path: str, vec_path: str):
    """
    in: csv file path, vectorizer save path
    out: df with preprocessed data, vectorized dataset (tf-idf sparse matrix)
    """
    df_preprocessed: pd.DataFrame = data_handling(csv_path)
    X_train_tfidf = tfidf_vectorizer(df_preprocessed, vec_path)
    return (df_preprocessed, X_train_tfidf)


# ------------ Test ------------
if __name__ == "__main__":
    df_test, X_train_tfidf = get_processed_dfs("../tar.csv", "./model")
    print(df_test.head())
    print(X_train_tfidf)
