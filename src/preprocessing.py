import re
import pandas as pd


def clean_text(text):

    text=str(text)
    text=re.sub(r"\s+"," ",text)
    return text.strip()


def preprocess_dataframe(df):

    df=df.copy()

    df["comment_text"]=(
        df["comment_text"]
        .apply(clean_text)
    )

    return df