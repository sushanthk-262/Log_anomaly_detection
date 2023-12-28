#import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F


# Load the Data
df = pd.read_csv('Spirit_sequence.csv', sep=',', quotechar='"', names=["text", "label"])
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

# Load the saved Model
model_path = "./fine_tuned_bert_model_for_Spirit"
model_saved = BertForSequenceClassification.from_pretrained(model_path)
tokenizer_saved = BertTokenizer.from_pretrained(model_path)

#Prediction
test_encodings = tokenizer_saved(X_test, truncation=True, padding=True)
predictions = model_saved(**test_encodings)
logits = predictions.logits

# Apply softmax to get probabilities
probabilities = F.softmax(logits, dim=1)

# Get the predicted label (0 or 1) using argmax
predicted_labels = torch.argmax(probabilities, dim=1)
predicted_labels = predicted_labels.tolist()
print(predicted_labels)

# Evaluation
y_test_labels = y_test.tolist()
# Create confusion matrix
cm = confusion_matrix(y_test_labels, predicted_labels)
# Calculate metrics
accuracy = accuracy_score(y_test_labels, predicted_labels)
precision = precision_score(y_test_labels, predicted_labels)
recall = recall_score(y_test_labels, predicted_labels)
f1 = f1_score(y_test_labels, predicted_labels)
print("Confusion Matrix:")
print(cm)
print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)