# !pip install transformers
# !pip install torch
# !pip install scikit-learn
# !pip install transformers[torch] -U
# !pip show accelerate



import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
true_news = pd.read_csv('True.csv')
fake_news = pd.read_csv('Fake.csv')

# Add class labels
true_news['class'] = 1
fake_news['class'] = 0

# Merge data
news = pd.concat([true_news, fake_news]).reset_index(drop=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(news['text'], news['class'], test_size=0.2, random_state=42)

# Display the shape of the datasets
print(f'Training set shape: {X_train.shape}')
print(f'Testing set shape: {X_test.shape}')


from transformers import BertTokenizer
# TIME TAKEN 10-15 min
# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize data
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=512)

# Display the size of the tokenized data
print(f'Train encodings size: {len(train_encodings["input_ids"])}')
print(f'Test encodings size: {len(test_encodings["input_ids"])}')


import torch



class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create the datasets
train_dataset = NewsDataset(train_encodings, list(y_train))
test_dataset = NewsDataset(test_encodings, list(y_test))

# Display the size of the datasets
print(f'Training dataset size: {len(train_dataset)}')
print(f'Testing dataset size: {len(test_dataset)}')



from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Subset
import numpy as np

# Define a smaller subset of your dataset for quick experimentation
subset_train_indices = range(1000)  # Adjust the subset size as needed
subset_test_indices = range(200)   # Adjust the subset size as needed

subset_train_dataset = Subset(train_dataset, subset_train_indices)
subset_test_dataset = Subset(test_dataset, subset_test_indices)
# Load BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
print(torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',                # Output directory
    num_train_epochs=10,                    # Number of training epochs
    per_device_train_batch_size=4,         # Reduced batch size
    per_device_eval_batch_size=4,          # Reduced batch size for evaluation
    warmup_steps=500,                      # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,                     # Strength of weight decay
    logging_dir='./logs',                  # Directory for storing logs
    logging_steps=10,                      # Log every 10 steps
    evaluation_strategy="epoch",           # Evaluate at the end of every epoch
    save_strategy="epoch",                 # Save checkpoints at the end of every epoch
    load_best_model_at_end=True,           # Load the best model saved at the end of training
    metric_for_best_model="eval_loss",     # Use eval_loss to determine the best model
    greater_is_better=False                # Minimize the eval_loss metric
)


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=subset_train_dataset,    # Use the subset of training dataset
    eval_dataset=subset_test_dataset     # Use the subset of test dataset

)

# Train the model
trainer.train()


# Save the model
model_path = "./bert_fake_news_classifier"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)
print(f"Model saved to {model_path}")



# Evaluate the model
results = trainer.evaluate()

# Display evaluation results
for key, value in results.items():
    print(f'{key}: {value}')
