# %%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score

# %%
dftrain = pd.read_csv('train.csv')
dftest = pd.read_csv('test.csv')


# %%
class ArtDataset(Dataset):
    def __init__(self, df, transform = None, has_labels = True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.has_labels = has_labels
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['ImagePath']).convert('RGB')

        if self.transform:
            img = self.transform(img)
        
        if self.has_labels:
            return img, int(row['Label'])
        else:
            return img, row['SampleID']

# %%
IMG_SIZE = 224

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# %%
class Advanced_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        #CONV BLOCK 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        #2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        #3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        #4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))

        x = x.mean([2,3])

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x



# %%
val_size = int(0.1 * len(dftrain))
df_val = dftrain.sample(val_size, random_state=42)
df_train = dftrain.drop(df_val.index)

# %%
train_ds = ArtDataset(dftrain, train_transform)
val_ds = ArtDataset(df_val, test_transform)
test_ds = ArtDataset(dftest, test_transform)

# %%
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
test_loader = DataLoader(test_ds, batch_size=32)

# %%
model = Advanced_CNN()

# %%
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%
EPOCHS = 20
best_f1 = 0
best_state = None

# %%
for epoch in range(EPOCHS):

    model.train()
    total_loss = 0
    corect = 0 
    total =0

    for imgs, labels in train_loader:
        
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        corect += (preds.argmax(1) == labels).sum().item()
        total += len(labels)

    train_acc = corect / total * 100

    model.eval()                                                         
    all_preds =[]
    all_labels =[]
    val_loss = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            preds = model(imgs)
            val_loss += criterion(preds, labels).item()

            all_preds.extend(preds.argmax(1).numpy())
            all_labels.extend(labels.numpy())

    f1 = f1_score(all_labels, all_preds)

    print(f"Epoch {epoch+1}/ {EPOCHS}"
          f"train_acc= {train_acc:.2f}%"
          f"val_f1= {f1:.4f}")
    
    if(f1>best_f1):
        best_f1 = f1
        best_state = model.state_dict()

print("Best_F1", best_f1)

# %%
model.load_state_dict(best_state)
model.eval()

all_ids = []
all_preds_final = []

# %%
with torch.no_grad():
    for imgs, IDds in test_loader:
        preds = model(imgs).argmax(1).numpy()

        all_ids.extend(IDds.numpy())
        all_preds_final.extend(preds)


# %%
submission = pd.DataFrame({
    'SampleID': all_ids,
    'Label': all_preds_final
})

# %%
submission.to_csv('submission.csv', index=False)


