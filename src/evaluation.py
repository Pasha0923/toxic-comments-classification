import torch
import numpy as np

from sklearn.metrics import classification_report, f1_score


def evaluate(model, dataloader, device, threshold=0.5):

    model.eval()

    all_preds = []
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

            # sigmoid for multi-label
            probs = torch.sigmoid(outputs.logits).cpu().numpy()

            preds = (probs > threshold).astype(int)

            all_preds.append(preds)
            all_labels.append(labels)

    # flatten
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=[
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate"
    ]))

    # return macro F1 for checkpointing
    f1 = f1_score(y_true, y_pred, average="macro")

    return f1