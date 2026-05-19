import torch

from torch.utils.data import Dataset


class ToxicDataset(Dataset):

    def __init__(self,texts,labels,tokenizer,max_length):
        self.texts=texts
        self.labels=labels
        self.tokenizer=tokenizer
        self.max_length=max_length

    def __len__(self):

        return len(self.texts)
    
    def __getitem__(self,idx):

        text=str(self.texts[idx])

        encoding=self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return{

            "input_ids": encoding["input_ids"].flatten(),

            "attention_mask": encoding["attention_mask"].flatten(),

            "labels":
            torch.tensor(
                self.labels[idx],
                dtype=torch.float
            )

        }