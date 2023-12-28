import pandas as pd

# Read the CSV data into a pandas DataFrame
# Replace 'your_dataset.csv' with the actual filename or path
df = pd.read_csv('Thunderbird.log_structured.csv')

# Convert the 'Timestamp' column to datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')

# Define the interval size in seconds (1800 seconds = 30 minutes)
interval_size = 900

# Create a new column for the window index
df['WindowIndex'] = (df['Timestamp'].astype('int64') // 10**9 // interval_size).astype('int64')

# Group the data based on the 'WindowIndex'
grouped_data = df.groupby('WindowIndex')

# Initialize a list to store results
result_list = []

# Iterate through groups and append to the list
for group_name, group_data in grouped_data:
    event_ids = set(group_data['EventId'])
    label = 'Normal' if all(group_data['Label'] == '-') else 'Anomalous'
    result_list.append({'EventIds': event_ids, 'Label': label})

# Create a new DataFrame from the list of dictionaries
result_df = pd.DataFrame(result_list)

# Save the result DataFrame to a new CSV file
result_df.to_csv('Thunderbird_sequence.csv')