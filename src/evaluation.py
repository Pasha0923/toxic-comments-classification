import torch
import numpy as np
from sklearn.metrics import classification_report, f1_score
from configuration.config import LABEL_COLUMNS

# function to find best thresholds for each class
def find_best_thresholds(y_true, probs):

    thresholds = np.arange(0.1,0.9,0.05)

    best_t = [0.5] * len(LABEL_COLUMNS)

    for i in range(len(LABEL_COLUMNS)):

        best_f1 = 0

        for t in thresholds:

            preds = (probs[:, i] > t).astype(int)

            f1 = f1_score(
                y_true[:, i],
                preds,
                zero_division=0
            )

            if f1 > best_f1:

                best_f1 = f1
                best_t[i] = t

    return best_t


# evaluation function
def evaluate(model,dataloader,device,threshold=None):

    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():

        for batch in dataloader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            labels = batch["labels"].cpu().numpy()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            probs = torch.sigmoid(outputs.logits).cpu().numpy()

            all_probs.append(probs)
            all_labels.append(labels)

    y_true = np.vstack(all_labels)
    y_probs = np.vstack(all_probs)

# auto threshold search
    
    if threshold is None:
        threshold = find_best_thresholds(
            y_true,
            y_probs
        )
        print("\nBest thresholds:")

        for label, t in zip(LABEL_COLUMNS,threshold):
            print(f"{label}: {t:.2f}")
   
    # predictions based on thresholds
    y_pred = np.zeros_like(y_probs)

    for i in range(len(LABEL_COLUMNS)):

        y_pred[:, i] = (y_probs[:, i] > threshold[i]).astype(int)
   
    # Classification Report

    print("\nClassification Report:\n")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=LABEL_COLUMNS,
            zero_division=0
        )
    )
    f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    return f1