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
images_folder = '../imagenes/imagenesConRecorte_5021img'
print("Cargando imágenes...")
images = load_images_from_folder(images_folder)
print("Imágenes cargadas. Total:", len(images))

# Cargar el archivo CSV de metadatos
metadata_file = '../metadatos_eda/16_10_metadatos_actualizados_sin_nv_reducidos.csv'
metadata_df = pd.read_csv(metadata_file)

# Verificar nulos y duplicados
metadata_df = metadata_df.dropna(subset=['age']).drop_duplicates()

# Clasificar la edad en rangos
bins = [0, 18, 35, 50, 65, 100]
labels = [0, 1, 2, 3, 4]  
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

# Transformaciones para las imágenes
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((image_height, image_width)),
    transforms.ToTensor(),
])

# Crear DataLoader con tamaño de batch reducido y sin uso de workers adicionales
batch_size = 16  # Tamaño reducido para mejor rendimiento en CPU
dataset = CustomDataset(images, metadata, labels, transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

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

# Función para evaluar el modelo
def evaluate_model(model, data_loader, threshold=0.5):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, metadata, labels in data_loader:
            outputs = model(images.float(), metadata.float())
            predicted = (outputs > threshold).float()  # Usar umbral fijo aquí
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Convertir las etiquetas y predicciones si es un problema multiclase
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1 and y_true.shape[1] > 1:  # Si es multilabel/multiclase
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

    return y_true, y_pred

# Función de entrenamiento
def train_model(model, train_loader, num_epochs=2, threshold=0.5):
    print(f"Iniciando entrenamiento por {num_epochs} épocas...")
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print(f"Iniciando época {epoch + 1}...")
        model.train()
        epoch_loss = 0
        start_time = time.time()  # Captura el tiempo de inicio al comienzo de cada época

        for batch_idx, (images, metadata, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images.float(), metadata.float())

            # Ajuste potencialmente problemático:
            # Aquí asumes que las etiquetas son adecuadas para CrossEntropyLoss.
            # Si tus etiquetas son multiclase, esto está bien, pero si son multilabel, CrossEntropyLoss no es el criterio correcto.
            loss = criterion(outputs, torch.max(labels, 1)[1])  # Verifica que las etiquetas estén en el formato correcto
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
        y_true, y_pred = evaluate_model(model, train_loader, threshold=threshold)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        print(f'Accuracia: {accuracy:.4f}, Precisión: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

        # Calcular y mostrar el tiempo total de la época
        total_epoch_time = time.time() - start_time
        print(f'Tiempo total de la época {epoch + 1}: {total_epoch_time:.2f} segundos')

# Realizar Cross-Validation - Folds 5!
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Para almacenar métricas
fold_accuracies = []
fold_precisions = []
fold_recalls = []
fold_f1_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"Iniciando Fold {fold + 1}...")

    # Crear subconjuntos de entrenamiento y validación
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    # Crear DataLoaders para el fold actual
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    # Inicializar el modelo para cada fold
    model = MultiInputModel(num_metadata_features=metadata.shape[1], num_classes=labels.shape[1])
    
    # Entrenar el modelo
    train_model(model, train_loader, num_epochs=10)  # Ajusta el número de épocas según sea necesario

    # Evaluar en el conjunto de validación
    y_true, y_pred = evaluate_model(model, val_loader)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    fold_accuracies.append(accuracy)
    fold_precisions.append(precision)
    fold_recalls.append(recall)
    fold_f1_scores.append(f1)

    print(f'Fold {fold + 1} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

# Promediar las métricas de los folds
print(f'Métricas promedio - Accuracy: {np.mean(fold_accuracies):.4f}, Precision: {np.mean(fold_precisions):.4f}, Recall: {np.mean(fold_recalls):.4f}, F1 Score: {np.mean(fold_f1_scores):.4f}')

# Mostrar matriz de confusión para el último fold - Aca se rompe, cambiar. 
plt.figure(figsize=(10, 7))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.categories_[0], yticklabels=encoder.categories_[0])
plt.title('Matriz de Confusión')
plt.xlabel('Predicciones')
plt.ylabel('Verdaderos')
plt.show()