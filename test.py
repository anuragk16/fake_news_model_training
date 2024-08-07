import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader

print("step1")
# Load the dataset
df = pd.read_csv("News.csv")
df = df.sample(frac=1) 
# Preprocess the dataset
texts = df['text'].astype(str).tolist()  # Ensure all elements are strings
labels = df['label'].tolist()
print("step2")
# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("step3")
# Tokenize the inputs
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
print("step4")
# Create a dataset
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

dataset = NewsDataset(inputs, labels)
print("step5")
# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

# Load the model
model = BertForSequenceClassification.from_pretrained('./bert_fake_news_classifier_3')
model.eval()
print("step5")
# Define the evaluation function
def evaluate(model, dataloader):
    model.eval()
    total, correct = 0, 0
    for batch in dataloader:
        inputs = {key: val.to(model.device) for key, val in batch.items() if key != 'labels'}
        labels = batch['labels'].to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    accuracy = correct / total
    return accuracy

# Move model to the appropriate device (CPU or GPU)
print(torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
print("step7")
# Evaluate the model
accuracy = evaluate(model, dataloader)
print(f'Accuracy: {accuracy}')
