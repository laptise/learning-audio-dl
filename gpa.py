import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

device = "cpu"

# Load data
data = pd.read_csv('/Users/yoonsookim/Library/CloudStorage/GoogleDrive-ml.laptise@gmail.com/マイドライブ/AI/gpascore.csv').dropna()
y_train = data['admit'].values
x_train = data[['gre', 'gpa', 'rank']].values

# Normalize input features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

# Convert data to PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

# Define the neural network model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)
        self.activation = nn.Tanh()
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.output_activation(self.fc3(x))
        return x

model = Model().to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Training the model
epochs = 10000
batch_size = 10000
for epoch in range(epochs):
    # Shuffle data
    indices = torch.randperm(x_train.size(0))
    x_train_shuffled = x_train[indices]
    y_train_shuffled = y_train[indices]

    for i in range(0, len(x_train_shuffled), batch_size):
        x_batch = x_train_shuffled[i:i+batch_size]
        y_batch = y_train_shuffled[i:i+batch_size]

        # Forward pass
        outputs = model(x_batch)

        # Compute loss
        loss = criterion(outputs, y_batch.unsqueeze(1))

        # Zero gradients, backward pass, update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 1000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print('Training finished.')
