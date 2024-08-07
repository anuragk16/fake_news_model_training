
import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Load the model and tokenizer from the saved folder
model_save_path = './bert_fake_news_classifier'
model = BertForSequenceClassification.from_pretrained(model_save_path)
tokenizer = BertTokenizer.from_pretrained(model_save_path)

# Move model to the appropriate device (CPU or GPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Prediction for a single input
def predict(text):
    # Tokenize the input text
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    inputs = {key: val.to(model.device) for key, val in inputs.items()}
    
    # Make a prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1)
    return "Fake" if prediction.item() == 0 else "True"

# Example prediction
example_text = """Factbox: Contenders for senior jobs in Trump's administration (Reuters) - The following people are mentioned as contenders for senior roles as U.S. President-elect Donald Trump puts together his administration before taking office on Jan. 20, according to Reuters sources and other media reports. Trump already has named a number of people for other top jobs in his administration. * Chuck Conner, a former acting secretary of the U.S. Agriculture Department and current head of the National Council of Farmer Cooperatives * Tim Huelskamp, Republican U.S. representative from Kansas * Sid Miller, Texas agriculture commissioner  * Sonny Perdue, former Georgia governor * Navy Admiral Mike Rogers, director of the National Security Agency * Ronald Burgess, retired U.S. Army lieutenant general and former Defense Intelligence Agency chief  * Robert Cardillo, director of the National Geospatial-Intelligence Agency * Pete Hoekstra, Republican former U.S. representative from Michigan     * John Allison, a former chief executive officer of regional bank BB&T Corp and former head of the Cato Institute, a libertarian think tank * Paul Atkins, former SEC commissioner  * Thomas Hoenig, Federal Deposit Insurance Corp vice chairman and former head of the Kansas City Federal Reserve Bank * Dr. Scott Gottlieb, a venture capitalist, resident fellow at the American Enterprise Institute and former Food and Drug Administration deputy commissioner * Jim Oâ€™Neill, a Silicon Valley investor who previously served in the Department of Health and Human Services * Debra Wong Yang, a former U.S. attorney who was appointed by former President George W. Bush * Ralph Ferrara, a securities attorney at Proskauer Rose LLP * Paul Atkins, a former SEC commissioner who heads Trumpâ€™s transition team for independent financial regulatory agencies * Daniel Gallagher, Republican former SEC commissioner The Trump transition team confirmed the president-elect would choose from a list of 21 names he drew up during his campaign, including Republican U.S. Senator Mike Lee of Utah and William Pryor, a federal judge with the 11th U.S. Circuit Court of Appeals. * Dan DiMicco, former CEO of steel producer Nucor Corp * Robert Lighthizer, former deputy U.S. trade representative during the Reagan administration * Wayne Berman, senior executive with private equity and financial services firm Blackstone Group LP  * David McCormick, president of investment manager Bridgewater Associates LP  * Pete Hegseth, CEO of Concerned Veterans for America and Fox News commentator  * Navy Admiral Michelle Howard  * Scott Brown, former Republican U.S. senator from Massachusetts * Sarah Palin, former Alaska governor and Republican nominee for vice president in 2008. * Jeff Miller, former Republican U.S. representative from Florida who was chairman of the House Veterans Affairs Committee * Larry Kudlow, economist and media commentator 
"""
print(f'Prediction: {predict(example_text)}')



# import torch
# from transformers import BertForSequenceClassification, BertTokenizer

# # Load tokenized data
# inputs = torch.load('tokenized_inputs.pt')
# labels = torch.load('labels.pt')

# # Load the tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# print("S1")
# # Create a dataset
# class NewsDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels
    
#     def __getitem__(self, idx):
#         item = {key: val[idx] for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item
    
#     def __len__(self):
#         return len(self.labels)

# dataset = NewsDataset(inputs, labels)

# # Create a DataLoader
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
# print("S2")
# # Load the model
# model = BertForSequenceClassification.from_pretrained('./bert_fake_news_classifier_3')
# model.eval()
# print("S3")
# # Define the evaluation function
# def evaluate(model, dataloader):
#     model.eval()
#     total, correct = 0, 0
#     for batch in dataloader:
#         inputs = {key: val.to(model.device) for key, val in batch.items() if key != 'labels'}
#         labels = batch['labels'].to(model.device)
        
#         with torch.no_grad():
#             outputs = model(**inputs)
        
#         logits = outputs.logits
#         predictions = torch.argmax(logits, dim=-1)
#         correct += (predictions == labels).sum().item()
#         total += labels.size(0)
    
#     accuracy = correct / total
#     return accuracy

# # Move model to the appropriate device (CPU or GPU)
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model.to(device)
# print("S4")
# # Evaluate the model
# accuracy = evaluate(model, dataloader)
# print(f'Accuracy: {accuracy}')
# print("S5")
# # Prediction for a single input
# def predict(text):
#     # Tokenize the input text
#     inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
#     inputs = {key: val.to(model.device) for key, val in inputs.items()}
    
#     # Make a prediction
#     with torch.no_grad():
#         outputs = model(**inputs)
    
#     logits = outputs.logits
#     prediction = torch.argmax(logits, dim=-1)
#     return "Fake" if prediction.item() == 0 else "True"
# print("S6")
# # Example prediction
# example_text = "Henningsen on White House Press Dinner: The Fourth Estate is Nonexistent in Americaâ€™"

# print(f'Prediction: {predict(example_text)}')
