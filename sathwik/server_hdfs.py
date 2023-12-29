import flwr as fl
import torch
import sys
import numpy as np

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd,
        results,
        failures
    ):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

# Create strategy and run server
strategy = SaveModelStrategy()

# Start Flower server for three rounds of federated learning
fl.server.start_server(
        server_address = "0.0.0.0:5020", 
        config=fl.server.ServerConfig(num_rounds=5),
        grpc_max_message_length = 1024*1024*1024,
        strategy = strategy
)


















# import flwr as fl
# import pickle
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras

# # Load and compile Keras model from the pickle file
# model_pickle_path = "HDFS_trained_model_1.pickle"
# with open(model_pickle_path, 'rb') as model_pickle:
#     model = pickle.load(model_pickle)

# # Define Flower server
# class FlowerServer(fl.server.NumPyServer):
#     def __init__(self, model):
#         self.model = model

#     def get_parameters(self):
#         return self.model.get_weights()

#     def fit(self, parameters):
#         # The server does not perform training directly on its data
#         return parameters

#     def evaluate(self, parameters, config):
#         # The server does not perform evaluation directly
#         return 0.0, 0, {}

# # Start Flower server
# server = FlowerServer(model)
# fl.server.start_numpy_server(
#     server_address="0.0.0.0:5012",
#     server=server,
#     config=fl.server.ServerConfig(num_rounds=4)
# )






# import flwr as fl
# import torch  # Assuming you are using PyTorch for your model

# # Replace this with your actual evaluation data
# # Example: Assuming you have a PyTorch DataLoader for evaluation
# eval_data = your_evaluation_dataloader

# # Define the Flower server
# class FlowerServer:
#     def __init__(self):
#         # Replace this with your actual global model initialization
#         # Example: Assuming you have a PyTorch model
#         self.model = your_global_model

#     def get_parameters(self):
#         return self.model.get_weights()

#     def fit(self, parameters, config):
#         self.model.set_weights(parameters)
#         return self.model.get_weights(), len(parameters), {}  # Adjust as needed

#     def evaluate(self, parameters, config):
#         self.model.set_weights(parameters)
        
#         # Perform evaluation on the server using the global model
#         eval_loss, eval_accuracy = self.evaluate_on_server(eval_data)
        
#         return eval_loss, len(eval_data), {"accuracy": eval_accuracy}

#     def evaluate_on_server(self, eval_data):
#         # Replace this with your actual evaluation logic
#         # Example: evaluate the model on the provided eval_data

#         # Assuming you have a PyTorch model
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(device)
#         self.model.eval()

#         eval_loss = 0.0
#         correct_predictions = 0
#         total_samples = 0

#         with torch.no_grad():
#             for inputs, labels in eval_data:
#                 inputs, labels = inputs.to(device), labels.to(device)

#                 # Assuming your model has a forward method
#                 outputs = self.model(inputs)
#                 loss = your_loss_function(outputs, labels)

#                 eval_loss += loss.item()
#                 _, predicted = torch.max(outputs, 1)
#                 correct_predictions += (predicted == labels).sum().item()
#                 total_samples += labels.size(0)

#         eval_loss /= len(eval_data)
#         eval_accuracy = correct_predictions / total_samples

#         return eval_loss, eval_accuracy

# if __name__ == '__main__':
#     fl.server.start_server(
#         server_address="0.0.0.0:5020",  # Adjust the server address
#         config=fl.server.strategy.FedAvg(num_rounds=5),
#     )



# if __name__ == '__main__':
#     server = FlowerServer()
#     fl.server.start_server(
#         server_address="0.0.0.0:5020",  # Adjust the server address
#         config=fl.server.ServerConfig(num_rounds=4),
#         server=server,
#     )



# import flwr as fl
# import torch

# class FlowerServer(fl.server.Server):
#     def __init__(self, model):
#         self.model = model

#     def get_parameters(self):
#         return [param.cpu().numpy() for param in self.model.parameters()]

#     def fit(self, parameters, config):
#         # Load parameters into the model
#         for param, new_param in zip(self.model.parameters(), parameters):
#             param.data = torch.tensor(new_param)

#         # Train on the global dataset (not provided in the code, please replace it with your global dataset)

#         # Return updated parameters
#         return [param.cpu().numpy() for param in self.model.parameters()], len(global_dataset), {}

#     def evaluate(self, parameters, config):
#         # Load parameters into the model
#         for param, new_param in zip(self.model.parameters(), parameters):
#             param.data = torch.tensor(new_param)

#         # Evaluate on the global dataset (not provided in the code, please replace it with your global dataset)

#         # Return evaluation metrics
#         return eval_loss, len(global_dataset), {'eval_loss': eval_loss}

# if __name__ == "__main__":
#     model = BertForSequenceClassification.from_pretrained("bert-base-uncased")  # Initialize your global model
#     fl.server.start_server(config={"num_rounds": 3}, server_address="[::]:8080", server=FlowerServer(model))
