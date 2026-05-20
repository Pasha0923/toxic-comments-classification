MODEL_NAME="bert-base-uncased"

MAX_LENGTH=128

BATCH_SIZE=16

EPOCHS=3

LEARNING_RATE=2e-5

RANDOM_STATE=42

LABEL_COLUMNS=[
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]