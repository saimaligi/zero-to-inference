"""
BERT Fine-tuning — IMDB Sentiment Classification
Dataset : stanfordnlp/imdb  (public, no token needed)
Model   : google-bert/bert-base-uncased
"""

import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification

#device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

#hyperparameters
max_length = 512
batch_size = 16
epochs     = 2
lr         = 2e-5

#dataset
imdb = load_dataset('stanfordnlp/imdb')

#tokenizer
tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
print("tokenizer loaded")

#tokenizing the dataset
def tokenize_function(data):
    return tokenizer(
        data['text'],
        padding='max_length',
        truncation=True,
        max_length=max_length
    )

tokenized_dataset = imdb.map(tokenize_function, batched=True, num_proc=1)
print("Data is tokenized according to the auto tokenizer")

#splits, convert to tensors and dataloaders
train_val  = tokenized_dataset['train'].train_test_split(test_size=0.2, seed=42)
train_data = train_val['train']
val_data   = train_val['test']
test_data  = tokenized_dataset['test']

columns = ['label', 'input_ids', 'token_type_ids', 'attention_mask']
train_data.set_format(type='torch', columns=columns)
val_data.set_format(type='torch', columns=columns)
test_data.set_format(type='torch', columns=columns)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader   = DataLoader(val_data, batch_size=batch_size)

#model
model = BertForSequenceClassification.from_pretrained('google-bert/bert-base-uncased', num_labels=2)
print("model loaded")
model = model.to(device)

#loss and optimizer
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

#training
print("training started")
for i in range(epochs):

    model.train()
    total_train_loss = 0

    for data in train_dataloader:
        optimizer.zero_grad()

        input_ids      = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        labels         = data['label'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        logits = outputs.logits
        loss   = F.cross_entropy(logits, labels, reduction='mean')
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(f"Epoch {i+1}/{epochs}  Train Loss: {total_train_loss / len(train_dataloader):.4f}")

    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for data in val_dataloader:
            input_ids      = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            labels         = data['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            logits = outputs.logits
            loss   = F.cross_entropy(logits, labels, reduction='mean')
            total_val_loss += loss.item()

    print(f"Epoch {i+1}/{epochs}  Val Loss:   {total_val_loss / len(val_dataloader):.4f}")
    print("-" * 30)

#accuracy matrix -- confusion matrix uses test data
print("metrics info")
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

test_dataloader = DataLoader(test_data, batch_size=batch_size)

model.eval()
all_predictions = []
all_labels      = []

with torch.no_grad():
    for data in test_dataloader:
        input_ids      = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        labels         = data['label'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        predictions = torch.argmax(outputs.logits, dim=-1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"Test Accuracy  : {accuracy_score(all_labels, all_predictions):.4f}")
print(f"Test F1 Macro  : {f1_score(all_labels, all_predictions, average='macro'):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(all_labels, all_predictions)}")