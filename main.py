import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Labels load karein
labels = pd.read_csv("labeled_anomalies.csv")
print("Labels loaded successfully!")

# 2. Ek channel select karein (jaise 'P-1')
row = labels[labels.chan_id == "P-1"].iloc[0]

# 3. Train aur test .npy files load karein
train_data = np.load(f"data/data/train/{row.chan_id}.npy")
test_data = np.load(f"data/data/test/{row.chan_id}.npy")

print(f"Channel ID: {row.chan_id}")
print(f"Train shape: {train_data.shape}")
print(f"Test shape: {test_data.shape}")

# 4. Telemetry aur commands ki sliding windows banana (Model ke liye last 250 steps dekh kar agla step predict karna)
L = 250
def make_windows(x):
    X = np.stack([x[i:i + L] for i in range(len(x) - L)])
    return X, x[L:, 0]

X_train, y_train = make_windows(train_data)
X_test, y_test = make_windows(test_data)

print(f"X_train shape (Windows): {X_train.shape}")
print(f"y_train shape (Targets): {y_train.shape}")

# 5. Data ko plot karna
plt.figure(figsize=(12, 4))
plt.plot(train_data[:, 0], label="Train Telemetry (P-1)", color="blue")
plt.title(f"Training Telemetry for Channel {row.chan_id}")
plt.xlabel("Time Steps")
plt.ylabel("Sensor Value")
plt.legend()
plt.show()

# 6. PyTorch Dataset aur Dataloader banana
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

# 7. LSTM Model Architecture (2 layers of 80 units with 0.3 dropout)
class LSTMAnomalyModel(nn.Module):
    def __init__(self, n_features):
        super(LSTMAnomalyModel, self).__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=80, num_layers=2, dropout=0.3, batch_first=True)
        self.fc = nn.Linear(80, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Aakhri time step ki prediction
        return out.squeeze()

n_features = X_train.shape[2]
model = LSTMAnomalyModel(n_features)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("LSTM Model successfully defined and ready for training!")


# --- Yeh code main.py ke aakhir mein add karein ---

# 8. Training Loop (Model ko train karna)
epochs = 5  # Aap shuru mein 5 epochs rakh lein taake jaldi check ho jaye
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


# 9. Test data par predictions lena aur errors calculate karna
model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    predictions = model(X_test_tensor).numpy()

# Real telemetry values (y_test) aur predictions ka absolute error
errors = np.abs(y_test - predictions)

# 10. Errors ko smooth karna (EWMA ke zariye taake chote mota shor alarm na bajaye)
error_series = pd.Series(errors)
smoothed_errors = error_series.ewm(span=30).mean().values

# 11. Simple Threshold (Mean + 3 * Standard Deviation)
threshold = smoothed_errors.mean() + 3 * smoothed_errors.std()
anomalies = smoothed_errors > threshold

print(f"Total test points: {len(test_data)}")
print(f"Total flagged anomalies: {np.sum(anomalies)}")