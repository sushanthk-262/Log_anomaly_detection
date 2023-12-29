import shutil
import pickle
import os

# Specify the folder name
model_folder = "HDFS_bert_model_2"
pickle_file = "HDFS_trained_model_2.pickle"

# Create a ZIP archive of the entire folder
shutil.make_archive(model_folder, 'zip', model_folder)

# Rename the archive to have a .pickle extension
os.rename(f"{model_folder}.zip", pickle_file)

print(f"Model saved as pickle file: {pickle_file}")
