import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Model definition
class MultiInputModel(nn.Module):
    def __init__(self, num_metadata_features, num_classes, image_height=128, image_width=128):
        super(MultiInputModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1_input_size = 128 * (image_height // 8) * (image_width // 8)
        self.fc1 = nn.Linear(self.fc1_input_size, 128)
        self.fc_meta = nn.Linear(num_metadata_features, 64)
        self.fc_output = nn.Linear(128 + 64, num_classes)

    def forward(self, img, meta):
        x = self.pool(F.relu(self.conv1(img)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        meta_out = F.relu(self.fc_meta(meta))
        combined = torch.cat((x, meta_out), dim=1)
        return self.fc_output(combined)

# Load the model
num_metadata_features = 2
num_classes = 2
model = MultiInputModel(num_metadata_features, num_classes)

try:
    model.load_state_dict(torch.load('modelo_entrenadomok.pth', map_location='cpu'))
    model.eval()
except FileNotFoundError:
    print("Model file not found.")
except Exception as e:
    print(f"Error loading model: {str(e)}")

# Model state check
def check_model_state():
    state = model.state_dict()
    for layer, params in state.items():
        print(f"Layer: {layer}, Weights size: {params.size()}")
