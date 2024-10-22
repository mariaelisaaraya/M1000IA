import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torchvision import transforms
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

# Definir las dimensiones de las imágenes
image_height, image_width, channels = 128, 128, 3

# Función para cargar imágenes desde una carpeta
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('RGB')  
        img = img.resize((image_width, image_height))  
        img_array = np.array(img)  
        images.append(img_array)
    return np.array(images)

# Definir el modelo
class MultiInputModel(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super(MultiInputModel, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
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
        output = self.fc_output(combined)

        return output

# Cargar imágenes
images_folder = 'carpeta_nueva_img'
images = load_images_from_folder(images_folder)

# Cargar el archivo CSV de metadatos
metadata_file = 'metadatos_actualizados_sin_nv_reducidos.csv'
metadata_df = pd.read_csv(metadata_file)

# Procesar metadatos
metadata_df = metadata_df.dropna(subset=['age']).drop_duplicates()
bins = [0, 18, 35, 50, 65, 100]
labels = [0, 1, 2, 3, 4] 
metadata_df['age_group'] = pd.cut(metadata_df['age'], bins=bins, labels=labels)

# Normalizar metadatos
scaler = StandardScaler()
age_normalized = scaler.fit_transform(metadata_df[['age']])

# Usar sparse_output en lugar de sparse
encoder = OneHotEncoder(sparse_output=False)
sex_encoded = encoder.fit_transform(metadata_df[['sex']].apply(lambda x: x.str.lower()))

# Concatenar metadatos
metadata = np.concatenate([age_normalized, sex_encoded], axis=1)

# Etiquetas para clasificación
labels = pd.get_dummies(metadata_df['classification']).values

# Crear un Dataset de PyTorch
class CustomDataset(Dataset):
    def __init__(self, images, metadata, labels, transform=None):
        self.images = images
        self.metadata = metadata
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        meta = self.metadata[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, meta, label

# Transformaciones para las imágenes (Data Augmentation)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.Resize((image_height, image_width)),
    transforms.ToTensor(),
])

# Crear Dataset
dataset = CustomDataset(images, metadata, labels, transform)

# Crear un DataLoader para la validación cruzada
val_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Cargar el modelo guardado
model = torch.load('modelo_entrenadomok.pth', map_location=torch.device('cpu'))

# Definir la función de evaluación del modelo
def evaluate_model(model, data_loader):
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for images, metadata, labels in data_loader:
            outputs = model(images.float(), metadata.float())
            predicted = torch.argmax(outputs, dim=1)
            y_pred.extend(predicted.numpy())
            y_true.extend(torch.max(labels, 1)[1].numpy())

    return np.array(y_true), np.array(y_pred)

# Realizar validación cruzada
y_true, y_pred = evaluate_model(model, val_loader)

# Imprimir métricas de evaluación
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

# Calcular y mostrar la matriz de confusión
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# Identificar falsos positivos y falsos negativos
false_positives = (y_pred == 1) & (y_true == 0)
false_negatives = (y_pred == 0) & (y_true == 1)

print(f'Falsos positivos: {np.sum(false_positives)}')
print(f'Falsos negativos: {np.sum(false_negatives)}')
