# ======== Preprocessing ======== #

import pandas as pd
import spacy
import re
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer

"""
with if-statement to avoid errors if model is not present ???
"""
try:
    nlp = spacy.load("de_core_news_lg")  # loads german language model
except OSError:
    from spacy.cli import download  # download model if not present

    download("de_core_news_lg")
    nlp = spacy.load("de_core_news_lg")


# ======== Preprocessing Pipeline ======== #
def data_handling(file):
    """
    in: csv file
    out: cleaned dataframe
    """

    # data input: csv -> df
    df_data = pd.read_csv(file, sep=";")
    df_data = df_data.head(10)  # limit to first 50 rows for testing

    # convert to lowercase
    df_data["description_clean"] = df_data["description"].str.lower()

    # whitelist of pronouns to keep
    whitelist = {
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
    }

    # function to remove stopwords
    def remove_stopwords(description):
        """
        in: description text
        out: description text without stopwords
        """

        doc = nlp(description)  # converts text to spacy-doc object with tokens
        cleaned_tokens = [
            token.text
            for token in doc  # for every token in doc
            if not token.is_stop
            or token.text in whitelist  # remove stopwords and keeps words in whitelist
        ]
        return " ".join(cleaned_tokens)  # joins tokens back to string

    # remove hyperlinks
    def remove_hyperlinks(description):
        """
        in: description text
        out: description text without hyperlinks
        """
        return re.sub(r"http[s]?://\S+", "", description)

    # map umlauts to their replacements
    umlaut_map = {
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "ß": "ss",
    }

    # function to remove special characters
    def remove_special_chars(description):
        """
        in: description text
        out: description text without special characters
        """

        # replace all umlauts
        for umlaut, replacement in umlaut_map.items():
            description = description.replace(
                umlaut, replacement
            )  # replace umlaut with replacement

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


# ======== Data Split ======== #
def split_data(df):
    """
    in: dataframe (cleaned)
    out: X_train -> train features
         X_test -> test features
         y_train -> train labels
         y_test -> test labels
    """

    X = df["description_clean"]  # features
    y = df["TAR"]  # labels

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,  # test data size: 20 %
        random_state=42,  # seed for reproducibility
        stratify=y,  # ensures balanced class distribution in train/test split
    )

    print(f"Training set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    return X_train, X_test, y_train, y_test


# ======== Oversampling ======== #
def random_oversampling(X_train, y_train):
    """
    in: X_train -> training features
        y_train -> training labels
    out: X_resampled -> oversampled training features
         y_resampled -> oversampled training labels
    """
    train_data = pd.DataFrame(
        {"description_clean": X_train, "TAR": y_train}
    )  # combining X and y into df

    max_class = train_data["TAR"].value_counts().idxmax()  # find majority class
    max_class_size = train_data["TAR"].value_counts().max()  # size of majority class
    print(f"Majority class: {max_class}")
    print(f"Majority class size: {max_class_size}")

    lst = []  # list to hold resampled dataframes

    # for every class
    for class_index, group in train_data.groupby("TAR"):
        if class_index == max_class:
            lst.append(group)  # keep majority class as is
            continue
        # oversample minority classes
        group_resampled = resample(
            group,
            replace=True,  # sample with replacement
            n_samples=max_class_size,  # to match majority class size
            random_state=42,  # seed for reproducibility
        )
        lst.append(group_resampled)

    train_data_resampled = pd.concat(lst)  # concatenate all resampled dataframes

    train_data_resampled = train_data_resampled.sample(
        frac=1, random_state=42
    ).reset_index(
        drop=True
    )  # shuffle dataframe (better for training)

    # split back into X (data) and y (labels)
    X_train_resampled = train_data_resampled["description_clean"]
    y_train_resampled = train_data_resampled["TAR"]

    print("\nClass distribution before oversampling:")
    print(y_train.value_counts())
    print("\nClass distribution after oversampling:")
    print(y_train_resampled.value_counts())

    return X_train_resampled, y_train_resampled


# ======= TF-IDF feature engineering ======== #
def tfidf_vectorizer(X_train_resampled, X_test):
    """
    in: X_train_resampled -> oversampled training features
        X_test -> test features
    out: X_train_tfidf -> TF-IDF transformed training features (sparse matrix)
         X_test_tfidf -> TF-IDF transformed test features (sparse matrix)
    """

    vectorizer = TfidfVectorizer(
        max_features=10000,  # limit feature space dimensionality
        min_df=2,  # mininmal document frequency for a token to be included (avoids overfitting)
        sublinear_tf=True,  # logarithmic scaling for better for better weighting
        norm="l2",  # L2 normalization
        stop_words=None,  # doesnt exlude more stopwords
    )

    X_train_tfidf = vectorizer.fit_transform(
        X_train_resampled
    )  # fit and transform train data
    X_test_tfidf = vectorizer.transform(X_test)  # only transform test data

    # print dimensions of tf-idf matrices
    print("Train shape:", X_train_tfidf.shape)
    print("Test shape:", X_test_tfidf.shape)

    # save tf-idf features to csv (optional)
    """X_train_tfidf_df = pd.DataFrame(
        X_train_tfidf.toarray(), columns=vectorizer.get_feature_names_out()
    )
    X_train_tfidf_df.to_csv("tfidf_train_oversampled.csv", index=False)"""

    return X_train_tfidf, X_test_tfidf


# ======= Data for feature extraction ======== #
def get_processed_dfs(csv_path):
    """
    in: none
    out: df with preprocessed data
    """
    df_preprocessed = data_handling(csv_path)
    X_train, X_test, y_train, y_test = split_data(df_preprocessed)
    X_train_resampled, y_train_resampled = random_oversampling(X_train, y_train)
    x_train_tfidf, x_test_tfidf = tfidf_vectorizer(X_train_resampled, X_test)
    return (
        df_preprocessed,
        X_train_resampled,
        X_test,
        x_train_tfidf,
        x_test_tfidf,
        y_train_resampled,
        y_test,
    )


# ======= Test ======== #
if __name__ == "__main__":
    df_preprocessed = data_handling("tar.csv")
    print(df_preprocessed)

    X_train, X_test, y_train, y_test = split_data(df_preprocessed)
    X_train_resampled, y_train_resampled = random_oversampling(X_train, y_train)

    tfidf_vectorizer(X_train_resampled, X_test)
