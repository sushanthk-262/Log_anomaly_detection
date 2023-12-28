#import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Load the Data
df = pd.read_csv('BGL_sequence.csv', sep=',', quotechar='"', names=["text", "label"])
df = df[1:]
# Feature extraction
X = list(df['text'])
y = list(df['label'])

# Get dummies(mapping)
y = pd.get_dummies(y, drop_first=True)['Normal']
y = y.astype(int)

#Make Anomalous as 1 and Normal as 0
for x in range(len(y)):
  if y[x] == 0:
    y[x] = 1
  else:
    y[x] = 0

# Train-test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Load the Model from transformers
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Get the Encodings 
train_encodings = tokenizer(X_train, truncation=True, padding=True)
test_encodings = tokenizer(X_test, truncation=True, padding=True)

# Get DataSets
# Create datasets
class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_ids = self.inputs['input_ids'][idx]
        attention_mask = self.inputs['attention_mask'][idx]
        label = torch.tensor(self.labels.iloc[idx])

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': label}

# Create an instance of the custom dataset
dataset_train = CustomDataset(train_encodings, y_train)
dataset_test = CustomDataset(test_encodings, y_test)

# Train the model
# Training Arguments
training_args = TrainingArguments(
    output_dir="./bert_base_model",
    evaluation_strategy="steps",
    eval_steps=100,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_steps=1000,
    save_total_limit=2,
    num_train_epochs=10,
    logging_dir="./logs",
)

# Trainer
trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=dataset_train,         # training dataset
    eval_dataset=dataset_test             # evaluation dataset
)

#Training
trainer.train()

# Save the Trained Model
model.save_pretrained("./fine_tuned_bert_model_for_BGL")
tokenizer.save_pretrained("./fine_tuned_bert_model_for_BGL")