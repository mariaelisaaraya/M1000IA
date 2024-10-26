import pytest
#import requests
import torch.nn as nn
from prediction_tf import predict_with_metadata
from model_tf import load_model
import os

# Base URL of your API
BASE_URL = 'http://127.0.0.1:5000'

# Example image file paths
ROOT_IMAGE_PATH = 'test/test_images' # revisar si en windows hay que poner \\ en lugar de /
VALID_IMAGE_PATH =  os.path.join(ROOT_IMAGE_PATH, 'isic_0024307.jpg')
INVALID_IMAGE_PATH = 'readme.md'
BENIGNO = 0
MALIGNO = 1

def util_test_image_prediction(file_path, data, expected_predicted_class):
    """Utility function to test image prediction."""
    with open(file_path, 'rb') as image_file:
        files = {'image': ('image.jpg', image_file, 'multipart/form-data')}
        edad= data['edad']
        sexo = data['sexo']
        model = load_model()
        predicted_class, predicted_probabilities = predict_with_metadata(file_path,edad,sexo)
        assert predicted_class == expected_predicted_class
 
def test_valid_image_prediction_class0_nv_ori():
    """Test prediction with a valid image and metadata using the model VER ACA"""
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
    """Test prediction with a valid image andmetadata using the model."""
    data = {'edad': 75, 'sexo': 'male'}
    util_test_image_prediction(os.path.join(ROOT_IMAGE_PATH, 'pat_754_1429_380.jpg'), data, MALIGNO)

def test_valid_image_prediction_class1_bcc_extra():
    """Test prediction with a valid image and metadata using the model."""
    data = {'edad': 70, 'sexo': 'male'}
    util_test_image_prediction(os.path.join(ROOT_IMAGE_PATH, 'pat_789_1483_793.jpg'), data, MALIGNO)



