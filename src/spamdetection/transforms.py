import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()


def init_nltk():
    nltk.download("punkt")
    nltk.download('stopwords')


def tokenize_words(text):
    """Tokenize words in text and remove punctuation"""
    text = word_tokenize(str(text).lower())
    text = [token for token in text if token.isalnum()]
    return text


def remove_stopwords(text):
    """Remove stopwords from the text"""
    text = [token for token in text if token not in stopwords.words("english")]
    return text


def stem(text):
    """Stem the text (originate => origin)"""
    text = [ps.stem(token) for token in text]
    return text


def transform(text):
    """Tokenize, remove stopwords, stem the text"""
    text = tokenize_words(text)
    text = remove_stopwords(text)
    text = stem(text)
    text = " ".join(text)
    return text


def transform_df(df):
    """Apply the transform function to the dataframe"""
    df["transformed_text"] = df["text"].apply(transform)
    return df


def encode_df(df, encoder=None):
    """Encode the features for training set"""
    if hasattr(encoder, "vocabulary_"):
        X = encoder.transform(df["transformed_text"]).toarray()
    else:
        X = encoder.fit_transform(df["transformed_text"]).toarray()
    y = df["label"].values
    return X, y, encoder


def tokenize(dataset, tokenizer):
    """Tokenize dataset"""

    def tokenization(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    def tokenization_t5(examples, padding="max_length"):
        # Add T5 prefix to the text
        text = ["classify as ham or spam: " +
                item for item in examples["text"]]

        # Tokenize text and labels
        inputs = tokenizer(text, max_length=tokenizer.model_max_length,
                           padding=padding, truncation=True)
        labels = tokenizer(
            text_target=examples["label"], max_length=max_label_length, padding=True, truncation=True)

        # Replace tokenizer.pad_token_id in the labels by -100 to ignore padding in the loss
        inputs["labels"] = [
            [(x if x != tokenizer.pad_token_id else -100) for x in label] for label in labels["input_ids"]
        ]
        return inputs

    if tokenizer is None:
        return dataset

    elif "T5" in type(tokenizer).__name__:
        # Extra step to convert our 0/1 labels into "ham"/"spam" strings
        dataset = dataset.map(
            lambda x: {"label": "ham" if x["label"] == 0 else "spam"})

        # Calculate the max label length after tokenization
        tokenized_label = dataset["train"].map(
            lambda x: tokenizer(x["label"], truncation=True), batched=True)
        max_label_length = max([len(x) for x in tokenized_label["input_ids"]])

        return dataset.map(tokenization_t5, batched=True, remove_columns=["label"])

    else:
        return dataset.map(tokenization, batched=True)
