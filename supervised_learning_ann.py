# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 13:00:58 2026

@author: gerar
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


### Parameters
test_size = 0.2
# depth = 2
width = 8
# activation function = ReLU
epochs = 1000
learning_rate = 0.01


### Parameters
T = np.genfromtxt("T.csv", delimiter=',')
Y = np.genfromtxt("Y_raw.csv", delimiter=',')

T_train, T_test, Y_train, Y_test = train_test_split(T, Y, test_size=0.2, random_state=0)

T_scaler = StandardScaler()
T_train = T_scaler.fit_transform(T_train)
T_test = T_scaler.transform(T_test)

Y_scaler = StandardScaler()
Y_train = Y_scaler.fit_transform(Y_train)
Y_test = Y_scaler.transform(Y_test)


num_features = len(T[0])
num_outputs = len(Y[0])



### Parameters

import torch
import torch.nn as nn
import torch.nn.functional as F


class NetworkCharacteristicPredictor(nn.Module):
    def __init__(self):
        super(NetworkCharacteristicPredictor, self).__init__()
        self.fc1 = nn.Linear(num_features, width)
        self.fc2 = nn.Linear(width, width)
        # self.fc3 = nn.Linear(width, width)
        # self.fc4 = nn.Linear(width, width)
        # self.fc5 = nn.Linear(width, width)
        self.output = nn.Linear(width, num_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        x = self.output(x)   # fixed - no activation on output
        return x
  


model = NetworkCharacteristicPredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
test_losses = []

for epoch in range(epochs):
    model.train()
    inputs = torch.tensor(T_train, dtype=torch.float32)
    labels = torch.tensor(Y_train, dtype=torch.float32)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        test_inputs = torch.tensor(T_test, dtype=torch.float32)
        test_labels = torch.tensor(Y_test, dtype=torch.float32)
        test_outputs = model(test_inputs)
        test_loss = criterion(test_outputs, test_labels)
        test_losses.append(test_loss.item())

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}]  Train Loss: {loss.item():.4f}  Test Loss: {test_loss.item():.4f}')


fig, ax = plt.subplots()
ax.plot(np.array(list(range(epochs))) + 1, train_losses)
ax.plot(np.array(list(range(epochs))) + 1, test_losses)
ax.set_yscale('log')
plt.show()


model.eval()
with torch.no_grad():
    test_inputs = torch.tensor(T_test, dtype=torch.float32)
    predictions = model(test_inputs).numpy()

predictions = Y_scaler.inverse_transform(predictions)
actuals = Y_scaler.inverse_transform(Y_test)

output_names = ['Permeability', 'Mean Throat Size', 'Std Throat Size']  # rename these

fig, ax = plt.subplots(1, 3, figsize=(10, 5))

colors = ['pink', 'mediumpurple', 'skyblue']

for i in range(3):
    ax[i].scatter(actuals[:, i], predictions[:, i], alpha=0.5, edgecolors='white', linewidths=0.3, c=colors[i])
    
    mn = min(actuals[:, i].min(), predictions[:, i].min())
    mx = max(actuals[:, i].max(), predictions[:, i].max())
    ax[i].plot([mn, mx], [mn, mx], 'r--')
    
    ax[i].set_xlabel('Actual')
    ax[i].set_ylabel('Predicted')
    ax[i].set_title(output_names[i])
    ax[i].legend()
    ax[i].grid(True)

plt.suptitle('Parity Plots', fontsize=14)
plt.tight_layout()
plt.show()


# Save predictions and actuals to .npy files
np.save('ann_predictions.npy', predictions)
np.save('ann_actuals.npy', actuals)
