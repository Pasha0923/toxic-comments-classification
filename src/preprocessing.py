import re

def clean_text(text):
    """
    Cleans text by removing extra spaces
    """
    text=str(text)
    text=re.sub(r"\s+"," ",text)
    return text.strip()

def preprocess_dataframe(df):
    """
    Applies text preprocessing
    to all comments in dataframe
    """
    df=df.copy()
    df["comment_text"]=(df["comment_text"].apply(clean_text))

    return df