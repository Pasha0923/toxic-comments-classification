import torch
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import ToxicDataset
from src.preprocessing import preprocess_dataframe
from configuration.config import *
from src.evaluation import evaluate


# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load data
df = pd.read_csv("data/train.csv")
df = preprocess_dataframe(df)

df = df.sample(10000, random_state=42)

# X / y (ВАЖНО: читаемо и правильно)
X = df["comment_text"]
y = df[LABEL_COLUMNS].values


# split
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE
)


# tokenizer
# tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
tokenizer = BertTokenizer.from_pretrained("/content/drive/MyDrive/bert_cache")

# datasets
train_dataset = ToxicDataset(
    texts=X_train.values,
    labels=y_train,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH
)

valid_dataset = ToxicDataset(
    texts=X_valid.values,
    labels=y_valid,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH
)


# dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# model
# model = BertForSequenceClassification.from_pretrained(
#     MODEL_NAME,
#     num_labels=len(LABEL_COLUMNS),
#     problem_type="multi_label_classification"
# )
model = BertForSequenceClassification.from_pretrained(
    "/content/drive/MyDrive/bert_cache",
    num_labels=len(LABEL_COLUMNS),
    problem_type="multi_label_classification"
)

# 🔥 ВРЕМЕННО ДОБАВИТЬ (один раз)
# model.save_pretrained("/content/drive/MyDrive/bert_cache")
# tokenizer.save_pretrained("/content/drive/MyDrive/bert_cache")

model.to(device)

# 🔥 CLASS WEIGHTS (FIX IMBALANCE)
label_counts = df[LABEL_COLUMNS].sum().values
total = len(df)

pos_weight = (total - label_counts) / (label_counts + 1e-6)
pos_weight = torch.tensor(pos_weight, dtype=torch.float).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# tracking best model
best_f1 = 0.0


# training loop
for epoch in range(EPOCHS):

    print(f"\n===== Epoch {epoch + 1}/{EPOCHS} =====")

    model.train()
    total_loss = 0

    for batch in tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # outputs = model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     labels=labels
        # )
        # loss = outputs.loss

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        loss = criterion(outputs.logits, labels.float())


        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
    print(f"Train Loss: {avg_loss:.4f}")


    # =========================
    # VALIDATION
    # =========================
    f1 = evaluate(model, valid_loader, device)

    print(f"Validation F1 (macro): {f1:.4f}")

    # save best model
    if f1 > best_f1:
        best_f1 = f1
        model.save_pretrained("/content/drive/MyDrive/best_model")
        tokenizer.save_pretrained("/content/drive/MyDrive/best_model")
        print("✅ Best model saved")

print("\nTraining finished.")
print(f"Best F1: {best_f1:.4f}")