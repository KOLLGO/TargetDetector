# ======== Preprocessing ======== #
import joblib
import pandas as pd
import spacy
import re
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer

# load spacy german language model
try:
    nlp = spacy.load("de_core_news_lg")  # loads german language model
except OSError:
    from spacy.cli import download  # download model if not present

    download("de_core_news_lg")
    nlp = spacy.load("de_core_news_lg")


# ======== Preprocessing Pipeline ======== #
def data_handling(file: str):
    """
    in: csv file path
    out: cleaned dataframe
    """

    # data input: csv -> df
    df_data: pd.DataFrame = pd.read_csv(file, sep=";")
    # df_data = df_data.head(100)  # optional limit for faster testing

    # convert to lowercase
    df_data["description_clean"] = df_data["description"].str.lower()

    # whitelist of pronouns to keep
    whitelist: set[str] = {
        # "ich",
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
            or token.text in whitelist  # remove stopwords and keeps words in whitelist
        ]
        return " ".join(cleaned_tokens)  # joins tokens back to string

    # remove hyperlinks
    def remove_hyperlinks(description: str):
        """
        in: description text
        out: description text without hyperlinks
        """
        return re.sub(r"http[s]?://\S+", "", description)

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


# ======== Oversampling (obsolete) ======== #
def random_oversampling(X_train: csr_matrix, y_train: csr_matrix):
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
def tfidf_vectorizer(df: pd.DataFrame, filepath: str):
    """
    in: df -> cleaned DataFrame
        filepath -> path to save the vectorizer
    out: X_train_tfidf -> TF-IDF transformed train features (sparse matrix)
    """

    vectorizer: TfidfVectorizer = TfidfVectorizer(
        max_features=10000,  # limit feature space dimensionality
        min_df=1,  # mininmal document frequency for a token to be included (avoids overfitting)
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
    )  # optional: save vectorizer
    return X_train_tfidf


# ======= Data for feature extraction ======== #
def get_processed_dfs(csv_path: str, vec_path: str):
    """
    in: csv file path, vectorizer save path
    out: df with preprocessed data, vectorized training dataset
    """
    df_preprocessed: pd.DataFrame = data_handling(csv_path)
    X_train_tfidf = tfidf_vectorizer(df_preprocessed, vec_path)
    return (df_preprocessed, X_train_tfidf)


# ======= Test ======== #
if __name__ == "__main__":
    df_test, X_train_tfidf = get_processed_dfs("../tar.csv", "./model")
    print(df_test.head())
    print(X_train_tfidf)
