import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate 1D training data
def generate_data(n_samples=200):
    x = np.random.uniform(-3, 3, n_samples)
    x = np.sort(x)  # Sort for better visualization
    y = np.zeros_like(x)
    
    for i in range(n_samples):
        if np.random.rand() > 0.5:
            y[i] = 0.5 * np.sin(x[i]) + 5 + 0.5 * np.random.randn()
        else:
            y[i] = 0.5 * np.sin(x[i]) - 5 + 0.5 * np.random.randn()
    
    return x, y

# MDN Model
class MDN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=15, n_gaussians=2):
        super(MDN, self).__init__()
        self.n_gaussians = n_gaussians
        
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.pi = nn.Linear(hidden_dim, n_gaussians)
        self.mu = nn.Linear(hidden_dim, n_gaussians)
        self.sigma = nn.Linear(hidden_dim, n_gaussians)
    
    def forward(self, x):
        hidden = self.hidden(x)
        pi = torch.softmax(self.pi(hidden), dim=1)
        mu = self.mu(hidden)
        sigma = torch.exp(self.sigma(hidden))
        return pi, mu, sigma

# Generate data
x_train, y_train = generate_data(500)
layout_dummy = np.random.randint(0, 2, 500)
component = [1]*500
data = np.column_stack((x_train, y_train))  # Combine x_train and y_train column-wise
data = np.column_stack((data, layout_dummy))  # Add layout column
data = np.column_stack((data, component))  # Add component column
df = pd.DataFrame(data, columns=["X_train", "y_train", "LAYOUT", "COMPONENTS"])
df.to_csv("data/ARCADE/test_data.csv", index=False)

X = torch.FloatTensor(x_train).reshape(-1, 1)
Y = torch.FloatTensor(y_train).reshape(-1, 1)

# Create data loader
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model and optimizer
model = MDN(input_dim=1, hidden_dim=15, n_gaussians=2)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Loss function
def mdn_loss(pi, mu, sigma, y):
    gaussian_prob = (1.0 / (sigma * np.sqrt(2*np.pi))) * \
                   torch.exp(-0.5 * ((y.unsqueeze(1) - mu) / sigma)**2)
    weighted_prob = pi * gaussian_prob
    return -torch.log(torch.sum(weighted_prob, dim=1) + 1e-6).mean()

# Training loop
n_epochs = 200
losses = []

for epoch in range(n_epochs):
    epoch_loss = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        pi, mu, sigma = model(batch_x)
        loss = mdn_loss(pi, mu, sigma, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    losses.append(epoch_loss / len(loader))
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/{n_epochs}, Loss: {losses[-1]:.4f}')

# Generate predictions for visualization
model.eval()
X_test, y_test = generate_data(200)
X_test = torch.FloatTensor(X_test).reshape(-1, 1)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)

with torch.no_grad():
    pi, mu, sigma = model(X_test)
    pi = pi.numpy()
    mu = mu.numpy()
    sigma = sigma.numpy()

# Calculate R2 score
from sklearn.metrics import r2_score
y_pred = np.sum(pi * mu, axis=1)
print(y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nR2 Score: {r2:.4f}")
X_test = X_test.numpy()

# Plotting
plt.figure(figsize=(15, 10))

# Plot 1: Training Data and Predictions
plt.subplot(2, 1, 1)

# Plot training data
plt.scatter(x_train, y_train, alpha=0.5, label='Training Data', color='blue', s=20)

# Plot predicted means and uncertainties for each Gaussian component
colors = ['red', 'green']
for i in range(model.n_gaussians):
    # Plot mean
    plt.plot(X_test, mu[:, i], color=colors[i], 
             label=f'Mean (Gaussian {i+1})', linewidth=2)
    
    # Plot uncertainty bounds (±2 sigma)
    plt.fill_between(X_test.flatten(), 
                    mu[:, i] - 2*sigma[:, i],
                    mu[:, i] + 2*sigma[:, i],
                    color=colors[i], alpha=0.2,
                    label=f'Uncertainty (Gaussian {i+1})')

plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.title('Training Data with Predicted Means and Uncertainties')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Mixing Coefficients
plt.subplot(2, 1, 2)
for i in range(model.n_gaussians):
    plt.plot(X_test, pi[:, i], color=colors[i],
             label=f'π{i+1}', linewidth=2)
plt.xlabel('Input (x)')
plt.ylabel('Mixing Coefficient (π)')
plt.title('Mixing Coefficients vs Input')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.savefig('mdn_results.png')

# Print example predictions for specific x values
test_points = [-1, 0, 1]
print("\nExample predictions at specific points:")
for x_val in test_points:
    x = torch.tensor([[x_val]], dtype=torch.float32)
    with torch.no_grad():
        pi, mu, sigma = model(x)
    print(f"\nx = {x_val}")
    print(f"Mixing coefficients (π): {pi.numpy().flatten()}")
    print(f"Means (μ): {mu.numpy().flatten()}")
    print(f"Standard deviations (σ): {sigma.numpy().flatten()}")