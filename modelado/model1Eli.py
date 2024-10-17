import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import time  # Asegúrate de importar el módulo time para controlar tiempos
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image

# Configura las dimensiones de las imágenes
image_height, image_width, channels = 128, 128, 3  # Ajusta según tus necesidades

# Cargar imágenes
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('RGB')  # Abrir la imagen y convertirla a RGB
        img = img.resize((image_width, image_height))  # Redimensionar la imagen
        img_array = np.array(img)  # Convertir a array NumPy
        images.append(img_array)
    return np.array(images)

# Ajusta la ruta a la carpeta donde están tus imágenes
images_folder = '../imagenesConRecorte_5021img'
print("Cargando imágenes...")
images = load_images_from_folder(images_folder)
print("Imágenes cargadas. Total:", len(images))

# Cargar el archivo CSV de metadatos
metadata_file = '/home/a3lisa/Documentos/Prog/IA/M1000IA/16_10_metadatos_actualizados_sin_nv_reducidos.csv'
metadata_df = pd.read_csv(metadata_file)

# Verificar nulos y duplicados
metadata_df = metadata_df.dropna(subset=['age'])
metadata_df = metadata_df.drop_duplicates()

# Clasificar la edad en rangos
bins = [0, 18, 35, 50, 65, 100]
labels = [0, 1, 2, 3, 4]  # Etiquetas para cada rango
metadata_df['age_group'] = pd.cut(metadata_df['age'], bins=bins, labels=labels)

# Normalizar metadatos
scaler = StandardScaler()
age_normalized = scaler.fit_transform(metadata_df['age'].values.reshape(-1, 1))

# One-Hot Encoding para 'sex'
encoder = OneHotEncoder(sparse_output=False)  # Cambiado a sparse_output
sex_encoded = encoder.fit_transform(metadata_df['sex'].str.lower().values.reshape(-1, 1))

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

# Crear DataLoader
dataset = CustomDataset(images, metadata, labels, transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

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

# Inicializar el modelo
num_metadata_features = metadata.shape[1]
num_classes = labels.shape[1]
model = MultiInputModel(num_metadata_features, num_classes)

# Definir el optimizador y la función de pérdida
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

# Función de evaluación del modelo con ajuste de umbral
def evaluate_model(model, data_loader, threshold=0.5):
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for images, metadata, labels in data_loader:
            outputs = model(images.float(), metadata.float())
            probabilities = F.softmax(outputs, dim=1)  # Obtener probabilidades
            predicted = (probabilities >= threshold).float()  # Aplicar el umbral
            predicted = torch.argmax(predicted, dim=1)  # Obtener la clase predicha
            y_pred.extend(predicted.numpy())
            y_true.extend(torch.max(labels, 1)[1].numpy())

    return np.array(y_true), np.array(y_pred)

# Función de entrenamiento con ajuste de umbral
def train_model(model, train_loader, num_epochs=40, threshold=0.5):
    print(f"Iniciando entrenamiento por {num_epochs} épocas...")
    for epoch in range(num_epochs):
        print(f"Iniciando época {epoch + 1}...")
        model.train()
        epoch_loss = 0
        start_time = time.time()  # Captura el tiempo de inicio

        for batch_idx, (images, metadata, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images.float(), metadata.float())
            loss = criterion(outputs, torch.max(labels, 1)[1])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if batch_idx % 10 == 0:  # Imprimir cada 10 lotes
            elapsed_time = time.time() - start_time  # Tiempo transcurrido
            estimated_time_per_batch = elapsed_time / (batch_idx + 1)  # Tiempo promedio por lote
            estimated_time_remaining = estimated_time_per_batch * (len(train_loader) - batch_idx - 1)  # Tiempo restante estimado
            print(f"Lote {batch_idx}, Pérdida: {loss.item():.4f}, Tiempo estimado restante: {estimated_time_remaining:.2f} segundos")

        print(f'Época {epoch + 1} finalizada, Pérdida: {epoch_loss / len(train_loader):.4f}')

        # Evaluación en el conjunto de entrenamiento
        y_true, y_pred = evaluate_model(model, train_loader, threshold)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        print(f'Accuracia: {accuracy:.4f}, Precisión: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

        # Calcular y mostrar el tiempo total de la época
        total_epoch_time = time.time() - start_time
        print(f'Tiempo total de la época {epoch + 1}: {total_epoch_time:.2f} segundos')


# Realizar Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"Iniciando Fold {fold + 1}...")
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    # Entrenar el modelo en cada fold
    train_model(model, train_loader)
    print(f"Fold {fold + 1} completado.")

# Guardar el modelo entrenado
torch.save(model, 'modelo_entrenadomok.pth')

