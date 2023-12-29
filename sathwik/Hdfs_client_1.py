from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import flwr as fl
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import zipfile
import pandas as pd
import shutil
import os
from sklearn.model_selection import train_test_split

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor([self.labels[idx], 1 - self.labels[idx]]).float()  # Ensure labels are float3
        return item

    def __len__(self):
        return len(self.labels)

# Initialize your model and tokenizer
model_saved = BertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer_saved = BertTokenizer.from_pretrained("bert-base-uncased")

# Specify the folder name
model_folder = "fine_tuned_bert_model_for_HDFS"
pickle_file = "pickle_fine_tuned_bert_model_for_HDFS.pickle"

# Load your dataset
df = pd.read_csv('Hdfs_labelled_sequence.csv', sep=',', quotechar='"', names=["text", "label"])
df = df[1:200]

X = list(df['text'])
y = pd.get_dummies(df['label'], drop_first=True)['Normal'].values

y = y.astype(int)

print(y)

for x in range(len(y)):
  if y[x] == 0:
    y[x] = 1
  else:
    y[x] = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Tokenize the test data
train_encodings = tokenizer_saved(X_train, truncation=True, padding=True)
test_encodings = tokenizer_saved(X_test, truncation=True, padding=True) #return_tensors="pt"

# Create an instance of the custom dataset
dataset_train = CustomDataset(train_encodings, y_train)
test_dataset = CustomDataset(test_encodings, y_test)


# Save the model using torch.save
torch.save(model_saved.state_dict(), os.path.join(model_folder, "model_weights.pth"))

# Create a ZIP archive of the entire folder AFTER saving the model
shutil.make_archive(model_folder, 'zip', model_folder)

# Rename the archive to have a .pickle extension
# os.rename(f"{model_folder}.zip", pickle_file)

# Load your model from the zip file
with zipfile.ZipFile(f"{model_folder}.zip", "r") as zip_ref:
    with zip_ref.open("model_weights.pth", "r") as file:
        model_weights = torch.load(file)

# Initialize your model and tokenizer
model_saved = BertForSequenceClassification.from_pretrained("bert-base-uncased")
model_saved.load_state_dict(model_weights)
tokenizer_saved = BertTokenizer.from_pretrained("bert-base-uncased")

# Initialize your optimizer and scheduler
optimizer = AdamW(model_saved.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(test_dataset))

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

# Define trainer outside methods
trainer = Trainer(
    model=model_saved,
    args=training_args,
    train_dataset=dataset_train,  # assuming you have a test_dataset
    eval_dataset=test_dataset,
    data_collator=None
    # optimizer=optimizer,
    # scheduler=scheduler,
    # optimizer=AdamW(model_saved.parameters(), lr=5e-5),
    # scheduler=get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=0, num_training_steps=len(test_dataset)
    # ),
)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = model_saved
        self.tokenizer = tokenizer_saved
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=len(test_dataset)
        )

    def get_parameters(self, config):
        return [param.cpu().detach().numpy() for param in self.model.parameters()]

    def fit(self, parameters, config):
        # Load parameters into the model
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param, requires_grad=True)

        # Train on your local dataset
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=len(test_dataset)
        ) 

        for epoch in range(3):  # Adjust the number of epochs as needed
            # trainer.train()
            # scheduler.step()
            for batch in DataLoader(test_dataset, batch_size=2, shuffle=True):
                inputs = {key: batch[key] for key in batch if key != 'labels'}
                inputs['labels'] = batch['labels'].squeeze().to(torch.float32)  # Ensure labels are float32
                outputs = self.model(**inputs)
                # Ensure the shape of labels matches the shape of logits
                loss = F.binary_cross_entropy_with_logits(outputs.logits.squeeze(), inputs['labels'].squeeze())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                
                
        # Print fit history
        print("Fit history: {'loss':", loss.item(), "}")

        # Return updated parameters
        return [param.cpu().detach().numpy() for param in self.model.parameters()], len(test_dataset), {}

    def evaluate(self, parameters, config):
        # Load parameters into the model
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param, requires_grad=False)

        # # Fine-tune on your local dataset
        # optimizer = AdamW(self.model.parameters(), lr=5e-5)
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=0, num_training_steps=len(test_dataset)
        # )

        # for epoch in range(3):  # Adjust the number of epochs as needed
        #     trainer.train()
        #     scheduler.step()
        
        # Set the model to evaluation mode
        self.model.eval()
        
        # Evaluate on your local dataset
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch in DataLoader(test_dataset, batch_size=2, shuffle=False):
                inputs = {key: batch[key] for key in batch if key != 'labels'}
                inputs['labels'] = batch['labels'].squeeze().to(torch.float32)  # Ensure labels are float32
                outputs = self.model(**inputs)
                # Ensure the shape of labels matches the shape of logits
                loss = F.binary_cross_entropy_with_logits(outputs.logits.squeeze(), inputs['labels'].squeeze())
                total_loss += loss.item() * len(batch['labels'])
                total_samples += len(batch['labels'])
                

        
        average_loss = total_loss / total_samples
        
        # Set the model back to training mode
        self.model.train()
        
        # # Return the evaluation metrics
        # return [average_loss, total_samples]
        
        # Evaluate on your local dataset
        predictions = trainer.evaluate()
        metrics = {'eval_loss': predictions['eval_loss']}  # Access 'eval_loss' directly
        
        # Print evaluation metrics
        print("Evaluation metrics: {'eval_loss':", average_loss, "}")
        
        # Print global Eval accuracy
        print("Now the global Eval accuracy:", 1.0 - average_loss)

        # Return the evaluation metrics
        return [average_loss, total_samples, {'eval_loss': average_loss}] #, metrics['eval_loss'], len(test_dataset), metrics]
    
    def get_config(self):
        return {}

    def get_weights(self):
        return [param.cpu().detach().numpy() for param in self.model.parameters()]

if __name__ == '__main__':
    fl.client.start_numpy_client(
        server_address="127.0.0.1:5020",
        client=FlowerClient(),
    )









































# import flwr as fl
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# import pickle
# import pandas as pd

# # Load and compile Keras model from the pickle file
# model_pickle_path = "trained_model_1.pickle"
# with open(model_pickle_path, 'rb') as model_pickle:
#     model = pickle.load(model_pickle)

# # Load custom dataset for client 1
# custom_data = pd.read_csv("data_set.csv")
# x_train = custom_data.drop("label", axis=1).values
# y_train = custom_data["label"].values

# # Normalize the features
# x_train = x_train / 255.0

# # Define Flower client for client 1
# class FlowerClient1(fl.client.NumPyClient):
#     def get_parameters(self, config):
#         return model.get_weights()

#     def fit(self, parameters, config):
#         model.set_weights(parameters)
#         # Perform training on client 1 with its data
#         r = model.fit(x_train, y_train, epochs=1, verbose=0)
#         hist = r.history
#         print("Client 1 Fit history: ", hist)
#         return model.get_weights(), len(x_train), {}

#     def evaluate(self, parameters, config):
#         model.set_weights(parameters)
#         # Perform evaluation on client 1 with its data
#         loss, accuracy = model.evaluate(x_train, y_train, verbose=0)
#         print("Client 1 Eval accuracy: ", accuracy)
#         return loss, len(x_train), {"accuracy": accuracy}

# # Start Flower client 1
# fl.client.start_numpy_client(
#     server_address="127.0.0.1:5012",
#     client=FlowerClient1()
# )

# class FlowerClient(fl.client.NumPyClient):
#     def __init__(self):
#         self.model = model_saved
#         self.tokenizer = tokenizer_saved

#     def get_parameters(self):
#         return self.model.state_dict()

#     def fit(self, parameters, config):
#         self.model.load_state_dict(parameters)

#         # Train on your local dataset
#         trainer = Trainer(
#             model=self.model,
#             args=TrainingArguments(
#                 output_dir="./fed_hdfs_training",
#                 per_device_train_batch_size=2,
#                 num_train_epochs=5,
#                 evaluation_strategy="epoch",
#                 logging_steps=50,
#                 save_steps=500,
#                 save_total_limit=2,
#                 gradient_accumulation_steps=4,
#                 gradient_checkpointing=True,
#                 load_best_model_at_end=True,
#                 metric_for_best_model="wer",
#                 greater_is_better=False,
#             ),
#             train_dataset=test_dataset,
#             data_collator=None,
#         )
#         trainer.train()

#         return self.model.state_dict(), len(test_dataset), {}

#     def evaluate(self, parameters, config):
#         self.model.load_state_dict(parameters)

#         # Evaluate on your local dataset
#         predictions = trainer.predict(test_dataset)
#         metrics = {'eval_loss': predictions.metrics['eval_loss']}

#         return metrics['eval_loss'], len(test_dataset), metrics
