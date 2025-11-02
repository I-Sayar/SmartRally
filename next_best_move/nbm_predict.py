import torch
import torch.nn as nn
import torch.nn.functional as F

class SmartRallyNet(nn.Module):
    def __init__(self):
        super(SmartRallyNet, self).__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)  #[recommended move, reccomended aiming location]

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = SmartRallyNet()
model.load_state_dict(torch.load("model.pth"))
model.eval()

def predict(input_raw):
    input_tensor = torch.tensor(input_raw, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        recommended_move = output[0][0].item()
        landing_location = output[0][1].item()
    return recommended_move, landing_location

if __name__ == "__main__":
    test_input = [5, 5, 2, 8, 8]  #[player loc, opponent loc, hit loc, land loc, prev move]
    move, location = predict(test_input)
    print(f"Recommended move (index or value): {move}")
    print(f"Predicted birdie landing location: {location}")
