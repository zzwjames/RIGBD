import torch
import torch.nn as nn
import torch.optim as optim

# Define the Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 2 * input_size // 3),
            nn.ReLU(True),
            nn.Linear(2 * input_size // 3, input_size // 3),
            nn.ReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(input_size // 3, 2 * input_size // 3),
            nn.ReLU(True),
            nn.Linear(2 * input_size // 3, input_size),
            nn.Sigmoid() # Use Sigmoid if the input data is normalized between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class MLPAE(nn.Module):
    def __init__(self, ori_x, trigger, device, epochs):
        super(MLPAE, self).__init__()
        self.device = device
        self.model = Autoencoder(len(ori_x[0])).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.epochs = epochs
        self.ori_x = ori_x
        self.trigger = trigger

    def fit(self):
        for epoch in range(self.epochs):
            output = self.model(self.ori_x)
            loss = self.criterion(output, self.ori_x)
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # print(f'Epoch [{epoch+1}], Loss: {loss.item():.4f}')

    # def inference(self, input):
    #     self.model.eval()
    #     reconstruction_errors = []
    #     with torch.no_grad():
    #         for sample in input:
    #             reconstructed = self.model(sample)
    #             loss = self.criterion(reconstructed, sample)
    #             reconstruction_errors.append(loss.item())
    #     return reconstruction_errors
    def inference(self, input):
        self.model.eval()
        reconstruction_errors = []
        with torch.no_grad():
            for sample in input:
                reconstructed = self.model(sample)
                loss = self.criterion(reconstructed, sample)
                reconstruction_errors.append(loss)

        # Convert the list of tensors to a single tensor
        reconstruction_errors_tensor = torch.stack(reconstruction_errors)
        return reconstruction_errors_tensor






# # Parameters
# input_size = m # replace 'm' with the number of features in your dataset
# num_epochs = 50 # number of epochs for training
# learning_rate = 1e-3

# # Model, Loss Function and Optimizer
# model = Autoencoder(input_size)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # Sample Dataset (replace this with your actual dataset)
# # Assuming each row in your dataset is an individual sample
# dataset = torch.randn(n, m) # replace 'n' and 'm' with your dataset's dimensions

# # Training Loop
# for epoch in range(num_epochs):
#     for data in dataset:
#         # Forward pass
#         output = model(data)
#         loss = criterion(output, data)

#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     # Log the progress
#     if (epoch+1) % 10 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
# You can now use model to reconstruct data and evaluate its performance.
