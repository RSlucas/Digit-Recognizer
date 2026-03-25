
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

dftrain = pd.read_csv('train.csv')
dftest = pd.read_csv('test.csv')

# %%
x = dftrain.drop(columns='label').values
y = dftrain['label'].values
x_final = dftest.values
x_final = x_final/225.0
x = x/225.0

# %%
x = x.reshape(-1, 1, 28, 28)
x_final = x_final.reshape(-1, 1, 28, 28)


# -1 = numărul de imagini (PyTorch îl calculează)
# 1 = canal (alb-negru)
# 28x28 = dimensiunea imaginii



# %%
x_final = torch.tensor(x_final, dtype=torch.float32)
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# float32 → pentru input
# long → pentru clasificare (CrossEntropyLoss)

class MNISTDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        # salvează datele în obiect

    def __len__(self):
        return len(self.x)
        # nr de exemple

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
        # returnează un exemplu (imagine + label)


#Dataset va lua batchuri automat

dataset = MNISTDataset(x, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

class CNNsimplu(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1) # 1 canal → 16 feature maps + 3x3 filtru
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2) # reduce dimensiunea la jumătate : 28x28 → 14x14 → 7x7!
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) # 10 clase (0->9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 1×28×28  → 16×14×14
        x = self.pool(F.relu(self.conv2(x)))   # 16×14×14 → 32×7×7
        x = x.view(x.size(0), -1)             # flatten  → 1568
        x = F.relu(self.fc1(x))               # 1568     → 128
        x = self.fc2(x)                        # 128      → 10
        return x

model = CNNsimplu()

# %%
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    total_loss, corect = 0, 0
    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        corect += (pred.argmax(1) == yb).sum().item()

    acc = corect / len(dataset)*100
    print(f"Epoch {epoch+1}/{EPOCHS} loss ={total_loss/len(loader)} acc = {acc:.2f}%")

model.eval()

with torch.no_grad():
    final_preds = model(x_final).argmax(1).numpy()

final = pd.DataFrame({
    'ImageID':range(1, len(x_final)+1),
    'Label': final_preds
})

final

final.to_csv('submission.csv', index=False)

