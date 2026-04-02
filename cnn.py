import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import uuid


X_raw = np.array(pd.read_csv('X_raw.csv'))
X_scaler = StandardScaler().fit(X_raw)
X_scaled = X_scaler.transform(X_raw)
X_scaled = X_scaled.reshape(9999, 1, 425)  # s x c x n (c = # channels, n = # features aka pressure locations, s = # samples)
I = torch.from_numpy(X_scaled).float()

Y_raw = np.array(pd.read_csv('Y_raw.csv'))
Y_raw = Y_raw.reshape(9999, 3)
Y_scaler = StandardScaler().fit(Y_raw)
Y_scaled = Y_scaler.transform(Y_raw)
Y_scaled = torch.from_numpy(Y_scaled).float()

class CNN(nn.Module):
    def __init__(self, input_size=425):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=11, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=7, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
        )
        
        dummy_input = torch.randn(1, 1, input_size)
        with torch.no_grad():
            dummy_out = self.features(dummy_input)
        flattened_size = dummy_out.view(1, -1).shape[1]
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128 , 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
     
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)       
        return x
    
cnn = CNN()
criterion = nn.MSELoss()

kf = KFold(n_splits=4, shuffle=True)
fold = 1
all_train_losses = []
all_val_losses = []
all_val_predictions = []
all_val_actuals = []

for train_idx, val_idx in kf.split(I):
    print(f"\n--- Fold {fold} ---")
    
    # Reset model and optimizer for each fold
    cnn = CNN()
    optimizer = optim.Adam(cnn.parameters(), lr=0.01)
    
    # Split data
    train_X, val_X = I[train_idx], I[val_idx]
    train_Y, val_Y = Y_scaled[train_idx], Y_scaled[val_idx]
    
    fold_train_losses = []
    fold_val_losses = []
    epochs = 250
    
    for epoch in range(epochs):
        # Training
        train_pred = cnn.forward(train_X)
        train_loss = criterion(train_pred, train_Y)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        fold_train_losses.append(train_loss.item())
        
        # Validation (no backprop)
        with torch.no_grad():
            val_pred = cnn.forward(val_X)
            val_loss = criterion(val_pred, val_Y)
        fold_val_losses.append(val_loss.item())
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train Loss {train_loss.item():.4f}, Val Loss {val_loss.item():.4f}")
    
    all_train_losses.append(fold_train_losses)
    all_val_losses.append(fold_val_losses)
    
    # Store final validation predictions from this fold
    all_val_predictions.append(val_pred.detach())
    all_val_actuals.append(val_Y.detach())
    
    fold += 1

# Concatenate predictions and actuals from all folds
all_predictions = torch.cat(all_val_predictions, dim=0)
all_actuals = torch.cat(all_val_actuals, dim=0)

# Define plotting functions
def plot_individual_folds(train_losses, val_losses, folds=4):
    """Plot training and validation loss for each fold separately."""
    plt.figure(figsize=(12, 6))
    for i in range(folds):
        plt.plot(train_losses[i], label=f'Fold {i+1} Train', linestyle='-', alpha=0.7)
        plt.plot(val_losses[i], label=f'Fold {i+1} Val', linestyle='--', alpha=0.7)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{folds}-Fold Cross-Validation: Training vs Validation Loss (Individual Folds)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cv_loss_individual_folds.png', dpi=150)
    plt.show()
    print("Plot saved as 'cv_loss_individual_folds.png'")

def plot_averaged_folds(train_losses, val_losses):
    """Plot averaged training and validation loss across folds with std dev bands."""
    mean_train = np.mean(train_losses, axis=0)
    mean_val = np.mean(val_losses, axis=0)
    std_train = np.std(train_losses, axis=0)
    std_val = np.std(val_losses, axis=0)
    
    plt.figure(figsize=(12, 6))
    plt.plot(mean_train, label='Train Loss', linewidth=2, color='blue')
    plt.plot(mean_val, label='Validation Loss', linewidth=2, color='orange')
    plt.fill_between(range(len(mean_train)), 
                     mean_train - std_train, 
                     mean_train + std_train, 
                     alpha=0.2, color='blue', label='Train Std Dev')
    plt.fill_between(range(len(mean_val)), 
                     mean_val - std_val, 
                     mean_val + std_val, 
                     alpha=0.2, color='orange', label='Val Std Dev')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Cross-Validation: Average Training vs Validation Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cv_loss_averaged_folds.png', dpi=150)
    plt.show()
    print("Plot saved as 'cv_loss_averaged_folds.png'")

def plot_parity(y_actual, y_predicted, title='Parity Plot: Predicted vs Actual'):
    """Plot parity plot comparing predicted vs actual values."""
    # Convert to numpy if needed
    if isinstance(y_actual, torch.Tensor):
        y_actual = y_actual.detach().cpu().numpy()
    if isinstance(y_predicted, torch.Tensor):
        y_predicted = y_predicted.detach().cpu().numpy()
    
    # Flatten if multi-dimensional
    y_actual = y_actual.flatten()
    y_predicted = y_predicted.flatten()
    
    # Get min and max for diagonal line
    min_val = min(y_actual.min(), y_predicted.min())
    max_val = max(y_actual.max(), y_predicted.max())
    
    # Calculate MSE
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_actual, y_predicted)
    
    # Generate random hash for filename
    random_hash = str(uuid.uuid4())[:8]
    filename = f'parity_plot_{random_hash}.png'
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y_actual, y_predicted, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    # Use scientific notation for MSE to handle both large and small magnitude outputs
    plt.title(f'{title}\nMSE = {mse:.4e}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Parity plot saved as '{filename}' (MSE = {mse:.4f})")
    return mse

# Call plotting functions
print("\n" + "="*50)
print("Plotting individual folds...")
plot_individual_folds(all_train_losses, all_val_losses)

print("\n" + "="*50)
print("Plotting averaged folds...")
plot_averaged_folds(all_train_losses, all_val_losses)

print("\n" + "="*50)
print("Generating parity plots...")

# Unscale predictions and actuals
all_actuals_unscaled = Y_scaler.inverse_transform(all_actuals.cpu().numpy())
all_predictions_unscaled = Y_scaler.inverse_transform(all_predictions.cpu().numpy())

for i in range(all_actuals_unscaled.shape[1]):
    plot_parity(all_actuals_unscaled[:, i], all_predictions_unscaled[:, i], title=f'Cross-Validation Parity Plot: Output {i+1}')