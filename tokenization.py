import pandas as pd
import torch
from transformers import BertTokenizer

# Load the dataset
df = pd.read_csv("News.csv")
df = df.iloc[0:30000]
df = df.sample(frac=1) 
texts = df['text'].astype(str).tolist()
labels = df['label'].tolist()

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the inputs
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# Save tokenized inputs and labels
torch.save(inputs, 'tokenized_inputs.pt')
torch.save(labels, 'labels.pt')
