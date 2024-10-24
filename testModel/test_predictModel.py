import pytest
#import requests
import torch.nn as nn
from model import load_model, MultiInputModel
from prediction import predict_with_metadata
import os

# Base URL of your API
BASE_URL = 'http://127.0.0.1:5000'

# Example image file paths
ROOT_IMAGE_PATH = 'test/test_images' # revisar si en windows hay que poner \\ en lugar de /
VALID_IMAGE_PATH =  os.path.join(ROOT_IMAGE_PATH, 'isic_0024307.jpg')
INVALID_IMAGE_PATH = 'readme.md'
BENIGNO = 0
MALIGNO = 1
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

        print(f"Image features shape: {x.shape}")
        # FIXME Error en la siguiente linea, no se puede concatenar x con meta
        meta_out = F.relu(self.fc_meta(meta))

        print(f"Metadata features shape: {meta_out.shape}")

        combined = torch.cat((x, meta_out), dim=1)  

        print(f"Combined features shape: {combined.shape}")

        output = self.fc_output(combined)
        return output


def util_test_image_prediction(file_path, data, expected_predicted_class):
    """Utility function to test image prediction."""
    with open(file_path, 'rb') as image_file:
        #files = {'image': ('image.jpg', image_file, 'multipart/form-data')}
        edad= data['edad']
        sexo = data['sexo']
        model =load_model()
        predicted_class, predicted_probabilities = predict_with_metadata(file_path,edad,sexo)
        #assert predicted_class
        #assert predicted_probabilities
        #assert predicted_class == expected_predicted_class, f"Expected predicted_class to be {expected_predicted_class}, but got {predicted_class} predicted probabilities are {predicted_probabilities}"
        assert predicted_class == expected_predicted_class
        #response = requests.post(f'{BASE_URL}/predict', files=files, data=data)
        #assert response.status_code == 200
        #json_data = response.json()
        #assert 'predicted_class' in json_data
        #assert 'predicted_probabilities' in json_data
        #assert json_data['predicted_class'] == expected_predicted_class, f"Expected predicted_class to be {expected_predicted_class}, but got {json_data['predicted_class']} predicted probabilities are {json_data['predicted_probabilities']}"

def test_valid_image_prediction_class0_nv_ori():
    """Test prediction with a valid image and metadata using the model VER ACA"""
    #data = {'edad': '50', 'sexo': 'male'}
    data = {'edad': 50, 'sexo': 'male'}
    util_test_image_prediction(VALID_IMAGE_PATH, data, BENIGNO)

def test_valid_image_prediction_class0_bkl_ori():
    """Test prediction with a valid image and metadata using the model."""
    data = {'edad': 70, 'sexo': 'female'}
    util_test_image_prediction(os.path.join(ROOT_IMAGE_PATH, 'isic_0027037.jpg'), data, BENIGNO)

def test_valid_image_prediction_class0_nv2_ori():
    """Test prediction with a valid image and metadata using the model."""
    data = {'edad': 55, 'sexo': 'female'}
    util_test_image_prediction(os.path.join(ROOT_IMAGE_PATH, 'isic_0030675.jpg'), data, BENIGNO)

def test_valid_image_prediction_class0_sek1_extra():
    """Test prediction with a valid image and metadata using the model."""
    data = {'edad': 90, 'sexo': 'female'}
    util_test_image_prediction(os.path.join(ROOT_IMAGE_PATH, 'pat_270_1382_561.jpg'), data, BENIGNO)

def test_valid_image_prediction_class0_sek2_extra():
    """Test prediction with a valid image and metadata using the model."""
    data = {'edad': 69, 'sexo': 'male'}
    util_test_image_prediction(os.path.join(ROOT_IMAGE_PATH, 'pat_701_1321_156.jpg'), data, BENIGNO)

def test_valid_image_prediction_class1_mel1_ori():
    """Test prediction with a valid image and metadata using the model."""
    data = {'edad': 60, 'sexo': 'male'}
    util_test_image_prediction(os.path.join(ROOT_IMAGE_PATH, 'isic_0024310.jpg'), data, MALIGNO)
    
def test_valid_image_prediction_class1_bcc_ori():
    """Test prediction with a valid image and metadata using the model."""
    data = {'edad': 55, 'sexo': 'female'}
    util_test_image_prediction(os.path.join(ROOT_IMAGE_PATH, 'isic_0029545.jpg'), data, MALIGNO)

def test_valid_image_prediction_class1_mel2_ori():
    """Test prediction with a valid image and metadata using the model."""
    data = {'edad': 85, 'sexo': 'female'}
    util_test_image_prediction(os.path.join(ROOT_IMAGE_PATH, 'isic_0033999.jpg'), data, MALIGNO)

def test_valid_image_prediction_class1_mel_extra():
    """Test prediction with a valid image and metadata using the model."""
    data = {'edad': 75, 'sexo': 'male'}
    util_test_image_prediction(os.path.join(ROOT_IMAGE_PATH, 'pat_754_1429_380.jpg'), data, MALIGNO)

def test_valid_image_prediction_class1_bcc_extra():
    """Test prediction with a valid image and metadata using the model."""
    data = {'edad': 70, 'sexo': 'male'}
    util_test_image_prediction(os.path.join(ROOT_IMAGE_PATH, 'pat_789_1483_793.jpg'), data, MALIGNO)



