import json
import torch
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from transformers import (BertTokenizer,BertForSequenceClassification,get_linear_schedule_with_warmup)
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import ToxicDataset
from src.preprocessing import preprocess_dataframe
from src.evaluation import evaluate
from src.focalloss import FocalLoss
from iterstrat.ml_stratifiers import (MultilabelStratifiedShuffleSplit)
from configuration.config import *

# ===================================
# DEVICE
# ===================================

device = torch.device("cuda" if torch.cuda.is_available()else "cpu")

# ===================================
# LOAD DATA
# ===================================

df = pd.read_csv("data/train.csv")

df = preprocess_dataframe(df)

df = df.sample(1000,random_state=42)

X = df["comment_text"]
y = df[LABEL_COLUMNS].values

# compute class weights for Focal Loss

label_counts = (df[LABEL_COLUMNS].sum().values)

total = len(df)

alpha = ((total-label_counts)/(label_counts+1e-6))

alpha = np.clip(alpha,1,20)

alpha = torch.tensor(alpha,dtype=torch.float).to(device)

print("Alpha weights:",alpha)

# ===================================
# MultilabelStratifiedShuffleSplit SPLIT
# ===================================

splitter=MultilabelStratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=RANDOM_STATE
)

for train_idx,valid_idx in splitter.split(X,y):

    X_train=X.iloc[train_idx]
    X_valid=X.iloc[valid_idx]

    y_train=y[train_idx]
    y_valid=y[valid_idx]


with open("outputs/valid_split.pkl", "wb") as f:
    pickle.dump((X_valid, y_valid), f)

# ===================================
# TOKENIZER
# ===================================

# tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
tokenizer = BertTokenizer.from_pretrained("/content/drive/MyDrive/bert_cache")

# ===================================
# DATASETS
# ===================================

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

# ===================================
# DATALOADERS
# ===================================

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

# ===================================
# MODEL
# ===================================

model = BertForSequenceClassification.from_pretrained(
    "/content/drive/MyDrive/bert_cache",
    num_labels=len(LABEL_COLUMNS),
    problem_type="multi_label_classification"
)

model.to(device)

# ===================================
# LOSS
# ===================================

criterion=FocalLoss(alpha=alpha,gamma=2)

# ===================================
# OPTIMIZER
# ===================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

# ===================================
# SCHEDULER
# ===================================

total_steps = (len(train_loader)* EPOCHS)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps*0.1),
    num_training_steps=total_steps
)

# ===================================
# HISTORY
# ===================================

train_losses = []
val_losses = []
val_f1_scores = []
best_f1 = 0
best_epoch=0
best_thresholds=None

# ===================================
# VALIDATION LOSS
# ===================================

def compute_val_loss():

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for batch in valid_loader:

            input_ids = batch["input_ids"].to(device)

            attention_mask = batch["attention_mask"].to(device)

            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = criterion(
                outputs.logits,
                labels.float()
            )

            total_loss += loss.item()

    return total_loss / len(valid_loader)

# ===================================
# TRAINING LOOP
# ===================================

for epoch in range(EPOCHS):

    print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

    model.train()

    total_loss = 0

    for batch in tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    ):

        input_ids = batch[
            "input_ids"
        ].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        labels = batch[
            "labels"
        ].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = criterion(
            outputs.logits,
            labels.float()
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

        optimizer.step()

        scheduler.step()

        total_loss += loss.item()


    avg_loss = (total_loss/ len(train_loader))

    train_losses.append(avg_loss)

    print(f"Train Loss: {avg_loss:.4f}")


    # ====================
    # VALIDATION
    # ====================

    val_loss = compute_val_loss()
    val_losses.append(val_loss)

    print(f"Validation Loss: {val_loss:.4f}")

    f1,thresholds=evaluate(model,valid_loader,device)
    val_f1_scores.append(f1)

    print(f"Validation F1: {f1:.4f}")

    # ====================
    # SAVE MODEL
    # ====================
    if f1 > best_f1:

        best_f1=f1
        best_epoch=epoch+1
        best_thresholds=thresholds

        model.save_pretrained("/content/drive/MyDrive/best_model_v4")
        tokenizer.save_pretrained("/content/drive/MyDrive/best_model_v4")
        print("✅ Best model saved")


# ===================================
# SAVE HISTORY
# ===================================
history={

    "train_losses":train_losses,

    "val_losses":val_losses,

    "val_f1":val_f1_scores,

    "best_f1":best_f1,

    "best_epoch":best_epoch,

    "thresholds":best_thresholds
}

with open("history.json","w") as f:
    json.dump(history,f,indent=4)

print("\n========== Training Finished ==========")

print(f"Best Validation F1: {best_f1:.4f}")

print(f"Best Epoch: {best_epoch}")