import torch
import pandas as pd

from sklearn.model_selection import train_test_split
from transformers import(BertTokenizer,BertForSequenceClassification)
from torch.utils.data import DataLoader

from src.dataset import ToxicDataset
from src.preprocessing import preprocess_dataframe
from configuration.config import *


device=(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


df=pd.read_csv(
    "data/train.csv"
)

df=preprocess_dataframe(df)


X_train,X_valid,y_train,y_valid=(
    train_test_split(
        df["comment_text"],
        df[LABEL_COLUMNS].values,
        test_size=0.2,
        random_state=RANDOM_STATE
    )
)


tokenizer=(
    BertTokenizer.from_pretrained(
        MODEL_NAME
    )
)


train_dataset=ToxicDataset(
    texts=X_train.values,
    labels=y_train,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH
)


valid_dataset=ToxicDataset(
    texts=X_valid.values,
    labels=y_valid,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH
)


train_loader=DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


valid_loader=DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE
)


model=(
    BertForSequenceClassification
    .from_pretrained(
        MODEL_NAME,
        num_labels=6,
        problem_type="multi_label_classification"
    )
)

model=model.to(device)


optimizer=torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)



for epoch in range(EPOCHS):

    model.train()

    total_loss=0


    for batch in train_loader:

        input_ids=(
            batch["input_ids"]
            .to(device)
        )

        attention_mask=(
            batch["attention_mask"]
            .to(device)
        )

        labels=(
            batch["labels"]
            .to(device)
        )


        outputs=model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )


        loss=outputs.loss

        total_loss+=loss.item()


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


    print(
        f"Epoch:{epoch+1}"
    )

    print(
        f"Loss:{total_loss}"
    )


torch.save(
    model.state_dict(),
    "models/bert_model.pt"
)