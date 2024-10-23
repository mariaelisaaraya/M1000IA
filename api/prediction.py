import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model import model

# Image transformation
image_height, image_width = 128, 128
transform = transforms.Compose([
    transforms.Resize((image_height, image_width)),
    transforms.ToTensor(),
])

# Predict with image and metadata
def predict_with_metadata(image_path, age, sex):
    try:
        # Load and transform the image
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)

        # Metadata tensor
        sex_numeric = 0 if sex == 'male' else 1
        metadata_tensor = torch.tensor([age, sex_numeric]).float().unsqueeze(0)

        # Predict
        with torch.no_grad():
            output = model(image, metadata_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            return predicted_class, probabilities.squeeze().tolist()

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None, None
