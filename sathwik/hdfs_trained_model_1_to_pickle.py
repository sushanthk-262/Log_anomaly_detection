# import shutil
# import pickle
# import os

# # Specify the folder name
# model_folder = "HDFS_bert_model_1"
# pickle_file = "HDFS_trained_model_1.pickle"

# # Create a ZIP archive of the entire folder
# shutil.make_archive(model_folder, 'zip', model_folder)

# # Rename the archive to have a .pickle extension
# os.rename(f"{model_folder}.zip", pickle_file)

# print(f"Model saved as pickle file: {pickle_file}")






import shutil
import cloudpickle
import os
from transformers import BertForSequenceClassification, BertTokenizer

# Initialize your model and tokenizer
model_saved = BertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer_saved = BertTokenizer.from_pretrained("bert-base-uncased")

# Specify the folder name
model_folder = "HDFS_bert_model_1"
pickle_file = "HDFS_trained_model_1.pickle"

# Save the model using cloudpickle
with open(os.path.join(model_folder, "model_weights.pkl"), "wb") as file:
    cloudpickle.dump(model_saved.state_dict(), file)

# Create a ZIP archive of the entire folder
shutil.make_archive(model_folder, 'zip', model_folder)

# Rename the archive to have a .pickle extension
os.rename(f"{model_folder}.zip", pickle_file)

print(f"Model saved as pickle file: {pickle_file}")
