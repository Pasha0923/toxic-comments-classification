import torch
import numpy as np

from sklearn.metrics import classification_report


def evaluate(model, dataloader,device):
    model.eval()

    predictions=[]
    true_labels=[]

    with torch.no_grad():

        for batch in dataloader:

            input_ids=(batch["input_ids"].to(device))

            attention_mask=(batch["attention_mask"].to(device))

            labels=(batch["labels"].cpu().numpy())

            outputs=model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            probs=torch.sigmoid(outputs.logits)

            preds=(probs>0.5).cpu().numpy()

            predictions.extend(preds)

            true_labels.extend(labels)

    print(classification_report(true_labels,predictions))