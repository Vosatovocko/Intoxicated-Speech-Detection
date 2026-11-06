import torch
from torch import nn

# drop out, batchnorm

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_units, output_size):
        super().__init__()
        self.binary_classify = nn.Sequential(
            #nn.Dropout(p=0.2, inplace=True),
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_size),
            nn.Sigmoid()
        )
        print(f"instantiate the model with {sum(p.numel() for p in self.parameters() if p.requires_grad)} params")
    def forward(self, input_size):
        return self.binary_classify(input_size)

class MulticlassNN(nn.Module):
    def __init__(self, input_size, hidden_units, output_size):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_units,
                      out_features=output_size)
        )
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x



class ImprovedNN(nn.Module):
    def __init__(self, input_size, hidden_units, output_size):
        super(ImprovedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_units)
        self.bn1 = nn.BatchNorm1d(hidden_units)
        self.tanh1 = nn.Tanh() # Tanh is for deeper nns
        self.dropout1 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.bn2 = nn.BatchNorm1d(hidden_units)
        self.tanh2 = nn.Tanh()
        self.dropout2 = nn.Dropout(p=0.3)

        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.bn3 = nn.BatchNorm1d(hidden_units)
        self.tanh3 = nn.Tanh()
        self.dropout3 = nn.Dropout(p=0.3)

        self.fc4 = nn.Linear(hidden_units, output_size)
        self.sigmoid = nn.Sigmoid()
        print(f"instantiate the model with {sum(p.numel() for p in self.parameters() if p.requires_grad)} params")

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.tanh1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.tanh2(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        out = self.bn3(out)
        out = self.tanh3(out)
        out = self.dropout3(out)

        out = self.fc4(out)
        out = self.sigmoid(out)
        return out
