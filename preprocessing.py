# ======== Preprocessing ======== #

import pandas as pd
import spacy
import re

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
    df_data = df_data.head(50)  # limit to first 50 rows for testing

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
            r"[^a-z0-9\s]", "", description
        )  # remove non-alphanumeric characters except spaces
        description = re.sub(
            r"\s+", " ", description
        )  # replace multiple spaces with single space
        return description.strip()  # remove leading/trailing spaces

    # apply preprocessing functions
    df_data["description_clean"] = (
        df_data["description_clean"]
        .apply(remove_stopwords)
        .apply(remove_hyperlinks)
        .apply(remove_special_chars)
    )

    return df_data


# ======= Test ======== #
print(data_handling("tar.csv"))
