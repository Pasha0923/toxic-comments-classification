# Toxic Comment Classification using BERT

## Project Overview

This project focuses on **multi-label toxic comment classification** using a fine-tuned BERT model.

The model analyzes user comments and predicts whether a comment belongs to one or more toxic categories.

Unlike standard binary classification ("toxic" vs "non-toxic"), this is a **multi-label classification task**, meaning one comment can belong to several categories simultaneously.

## Project Goal
The main objective of this project is:

- build a complete NLP pipeline for toxic comment detection
- fine-tune a pretrained Transformer model
- improve rare class prediction
- analyze model performance
- create an interactive web interface using Streamlit

## 🗂️ Project Structure
```bash
Toxic Comment Classification/
│
├── configuration/
│   └── config.py
│ 
├── notebook/
│   ├── eda.ipynb
│   └── toxic_demo_v3.ipynb
│
├── src/
├── ├──train.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── evaluation.py
│   ├── focalloss.py
│ 
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── test_labels.csv
│
├── models/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
│
├── outputs/
│   ├── history.json
│   ├── thresholds.json
│
│── .gitignore
├── app.py
├── requirements.txt
├── README.md   
```
## Project Pipeline
The project includes several stages:

1. Exploratory Data Analysis (EDA)

Notebook: eda.ipynb

Performed analysis:

- Missing values check
- Duplicate analysis
- Class imbalance analysis
- Toxic / non-toxic distribution
- Comment length analysis
- Histograms and visualizations
- Dataset statistics

2. Data preprocessing

The preprocessing pipeline was intentionally kept minimal to preserve the original semantic information in toxic comments.

Implemented preprocessing steps:

- converting text to string format
- removing extra whitespaces
- trimming leading and trailing spaces

Minimal preprocessing was used because BERT already contains its own tokenization and language representation mechanisms.

3. Transfer Learning

Instead of training from scratch:

- used pretrained BERT model
- applied Fine-Tuning for toxic classification task

## Model Architecture

The project uses a pretrained **BERT-base-uncased model** from Hugging Face for multi-label toxic comment classification.

**Workflow:**

1) A pretrained BERT model was loaded **bert-base-uncased**
2) A classification head for six toxic categories was added:

- toxic 
- severe_toxic 
- obscene
- threat 
- insult 
- identity_hate

3) The model was fine-tuned on the toxic comments dataset.

Fine-tuning allows the pretrained language model to adapt to the specific task while preserving the language understanding learned from large-scale text corpora.

**Architecture:**

Input Text
        ↓
BERT Tokenizer
        ↓
Pretrained BERT Encoder
        ↓
Dropout
        ↓
Linear Classification Layer
        ↓
Sigmoid activation
        ↓
6 probability outputs

Since the task is multi-label classification, sigmoid activation was used instead of softmax, allowing multiple classes to be predicted simultaneously.

## Download model weights

Model weights are not included in the repository because of file size limitations.

Download pretrained weights from:
```bash
https://drive.google.com/drive/folders/12TDF9qmEYu99A2PNVNRylADEYe0BYRUm?usp=sharing
```

After downloading:

Extract the downloaded archive
Copy the folder into:
models/

Final project structure:

```bash
├── models/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
```

## Imbalance with Focal Loss

The dataset contains a strong class imbalance problem: classes such as **threat**, **identity_hate**, and **severe_toxic** occur much less frequently than common classes toxic or obscene.

To improve performance on rare classes, the project uses Focal Loss instead of the standard Binary Cross Entropy loss.

Focal Loss dynamically reduces the contribution of easy examples and focuses training on difficult or misclassified samples.

**Parameters:**

alpha=1 — class balancing factor
gamma=2 — focuses learning on hard examples

**Benefits:**

Improves learning for underrepresented classes
Reduces the dominance of majority classes
Helps increase Macro F1 score
Makes the model more sensitive to rare toxic categories

## Training Strategy

To improve rare classes:

✔ Focal Loss
✔ Threshold optimization
✔ Learning rate scheduler
✔ Gradient clipping
✔ Fine-tuning pretrained BERT

## Model Training Configuration

| Parameter | Value |
|--------|-------|
| MAX_LENGTH | 128 |
| BATCH_SIZE | 16 |
| LEARNING_RATE | 2e-5 |
| EPOCHS | 5 |
| Optimizer | AdamW |
| Loss Function | FocalLoss(alpha=1, gamma=2) |
| Gradient Clipping | 1.0 |

## Dataset

Jigsaw Toxic Comment Classification Dataset
```bash
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
```
Full dataset:  **159,571** samples
Training size used: **20,000** samples

Although the original dataset is significantly larger, the model was intentionally trained on a smaller subset to reduce training time and demonstrate the complete workflow.

## 📊 Final Evaluation Summary

    |  Class   |  precision | recall | f1-score | support |
    |-------   |------------| -------|----------| --------|
       toxic   |    0.84    |  0.84  |   0.84   |   399   |
 severe_toxic  |    0.48    |  0.66  |   0.56   |   38    |
       obscene |    0.86    |  0.79  |   0.82   |   218   |
       threat  |    1.00    |  0.50  |   0.67   |    6    |
       insult  |    0.65    |  0.88  |   0.75   |   204   |
 identity_hate |    0.81    |  0.36  |   0.50   |   36    | 

 Best Validation Macro F1  | 69% 
 Test Macro F1             | 52% 

 ## Notebook (demo.ipynb)

The toxic_demo_v3.ipynb notebook contains:

- Model training
- Classification Report
- Visualization:  Train/Val Loss curves , F1 validation per epochs
- Final Test Evaluation
- ROC-AUC curves per class
- Example prediction classifier comments

## Streamlit Application

The project also contains an interactive Streamlit interface.

Features:

✔ Input custom comments
✔ Toxic / Non-toxic prediction
✔ Probability visualization
✔ Threshold-based classification
✔ Interactive charts

## ⚡ Local Installation 

1. **Clone the repository:**
```bash
git clone https://github.com/Pasha0923/toxic-comments-classification.git
cd toxic-comments-classification
```
2. **Create virtual environment (recommended)**
```bash
python -m venv venv
venv\Scripts\activate
```
3. **Install dependencies:**
```bash
pip install -r requirements.txt
```
4. **Install PyTorch CPU versions:**
```bash
pip install torch==2.5.1+cpu torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu
```
4. **Run the Streamlit application:**
```bash
streamlit run app.py
```