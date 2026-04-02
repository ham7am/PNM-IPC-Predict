class CNN(nn.Module):
    def __init__(self, input_size=425):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=7, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        dummy_input = torch.randn(1, 1, input_size)
        with torch.no_grad():
            dummy_out = self.features(dummy_input)
        flattened_size = dummy_out.view(1, -1).shape[1]
        
        self.classifier = nn.Sequential(
            # nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(flattened_size, 64),
            nn.ReLU(),
            # nn.Dropout(0.2), # turn of x% of perceptrons - random which ones each time. creates an ensemble network and also prevents network from relying on individual patterns
            nn.Linear(64, 3)
        )