
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import numpy as np
from sklearn.model_selection import train_test_split

class SmartRallyNet(nn.Module):
    def __init__(self):
        super(SmartRallyNet, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64) 
        self.fc4 = nn.Linear(64, 18)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

x, y = [], []
with open("dataset/processed.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    for row in reader:
        if row:
            x.append([int(r) for r in row[:-1]])
            y.append(int(row[-1]))   

X = np.array(x)
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

model = SmartRallyNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())  

for epoch in range(2000):  
    outputs = model(X_train_tensor)
    loss = loss_fn(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        #trainign acc
        _, pred = torch.max(outputs, 1)
        training_acc = (pred == y_train_tensor).float().mean().item()
        # test 
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            _, test_pred = torch.max(test_outputs, 1)
            test_acc = (test_pred == y_test_tensor).float().mean().item()
        print(f"epoch {epoch+1}, train acc {training_acc:.3f}, test acc {test_acc:.3f} ")

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, test_pred = torch.max(test_outputs, 1)
    test_acc = (test_pred == y_test_tensor).float().mean().item()

print(f"final test acc {test_acc:.3f}")
torch.save(model.state_dict(), "model.pth")

def predict_sequence(model, sequence):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        output = model(input_tensor)
        return torch.argmax(output, dim=1).item()
