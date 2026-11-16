import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


labels = pd.read_csv("labeled_anomalies.csv")
print("Labels loaded successfully!")


row = labels[labels.chan_id == "P-1"].iloc[0]


train_data = np.load(f"data/data/train/{row.chan_id}.npy")
test_data = np.load(f"data/data/test/{row.chan_id}.npy")

print(f"Channel ID: {row.chan_id}")
print(f"Train shape: {train_data.shape}")
print(f"Test shape: {test_data.shape}")


L = 250
def make_windows(x):
    X = np.stack([x[i:i + L] for i in range(len(x) - L)])
    return X, x[L:, 0]

X_train, y_train = make_windows(train_data)
X_test, y_test = make_windows(test_data)

print(f"X_train shape (Windows): {X_train.shape}")
print(f"y_train shape (Targets): {y_train.shape}")


plt.figure(figsize=(12, 4))
plt.plot(train_data[:, 0], label="Train Telemetry (P-1)", color="blue")
plt.title(f"Training Telemetry for Channel {row.chan_id}")
plt.xlabel("Time Steps")
plt.ylabel("Sensor Value")
plt.legend()
plt.show()


class TelemetryDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TelemetryDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)


class LSTMAnomalyModel(nn.Module):
    def __init__(self, n_features):
        super(LSTMAnomalyModel, self).__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=80, num_layers=2, dropout=0.3, batch_first=True)
        self.fc = nn.Linear(80, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out.squeeze()

n_features = X_train.shape[2]
model = LSTMAnomalyModel(n_features)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("LSTM Model successfully defined and ready for training!")



epochs = 5  
model.train()

print("Training started...")
for epoch in range(epochs):
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

print("Training completed successfully!")



model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    predictions = model(X_test_tensor).numpy()

errors = np.abs(y_test - predictions)


error_series = pd.Series(errors)
smoothed_errors = error_series.ewm(span=30).mean().values


threshold = smoothed_errors.mean() + 3 * smoothed_errors.std()
anomalies = smoothed_errors > threshold

print(f"Total test points: {len(test_data)}")
print(f"Total flagged anomalies: {np.sum(anomalies)}")