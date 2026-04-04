import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load the saved predictions and actuals
ann_actuals = np.load('ann_actuals.npy')
ann_predictions = np.load('ann_predictions.npy')
cnn_actuals = np.load('cnn_actuals.npy')
cnn_predictions = np.load('cnn_predictions.npy')

# Output names
output_names = ['Permeability', 'Mean Throat Size', 'Std Throat Size']

# Create ANN parity plots (3 vertical subplots)
fig_ann, axes_ann = plt.subplots(3, 1, figsize=(8, 12))

for i in range(3):
    ax = axes_ann[i]
    
    # Get data for this output
    ann_actual = ann_actuals[:, i]
    ann_pred = ann_predictions[:, i]
    
    # Plot data
    ax.scatter(ann_actual, ann_pred, alpha=1.0, s=50, c='#3498db', 
               edgecolors='#1a5276', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(ann_actual.min(), ann_pred.min())
    max_val = max(ann_actual.max(), ann_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate MSE
    ann_mse = mean_squared_error(ann_actual, ann_pred)
    
    # Labels and formatting
    ax.set_xlabel('Actual Values', fontsize=14)
    ax.set_ylabel('Predicted Values', fontsize=14)
    ax.set_title(f'{output_names[i]}\nMSE = {ann_mse:.4e}', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

fig_ann.suptitle('ANN Parity Plots', fontsize=18, fontweight='bold')
fig_ann.tight_layout()
fig_ann.savefig('parity_ann.png', dpi=150)
fig_ann.show()

# Create CNN parity plots (3 vertical subplots)
fig_cnn, axes_cnn = plt.subplots(3, 1, figsize=(8, 12))

for i in range(3):
    ax = axes_cnn[i]
    
    # Get data for this output
    cnn_actual = cnn_actuals[:, i]
    cnn_pred = cnn_predictions[:, i]
    
    # Plot data
    ax.scatter(cnn_actual, cnn_pred, alpha=1.0, s=50, c='#9b59b6', 
               edgecolors='#a93226', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(cnn_actual.min(), cnn_pred.min())
    max_val = max(cnn_actual.max(), cnn_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate MSE
    cnn_mse = mean_squared_error(cnn_actual, cnn_pred)
    
    # Labels and formatting
    ax.set_xlabel('Actual Values', fontsize=14)
    ax.set_ylabel('Predicted Values', fontsize=14)
    ax.set_title(f'{output_names[i]}\nMSE = {cnn_mse:.4e}', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

fig_cnn.suptitle('CNN Parity Plots', fontsize=18, fontweight='bold')
fig_cnn.tight_layout()
fig_cnn.savefig('parity_cnn.png', dpi=150)
fig_cnn.show()

print("ANN parity plots saved as 'parity_ann.png'")
print("CNN parity plots saved as 'parity_cnn.png'")
