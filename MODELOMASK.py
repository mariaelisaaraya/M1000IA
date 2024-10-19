import os
import numpy as np
import cv2
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Dimensiones de las imágenes
image_height, image_width = 128, 128

# Preprocesar imágenes con CLAHE para mejorar contraste
def apply_clahe(img_array):
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final_img

# Cargar imágenes y aplicar CLAHE
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((image_width, image_height))
        img_array = np.array(img)
        img_array = apply_clahe(img_array)  # Aplicar CLAHE
        images.append(img_array)
    return np.array(images)

# Ajusta la ruta a la carpeta donde están tus imágenes
images_folder = 'carpeta_nueva_img'  # Cambia esto a la ruta correcta
images = load_images_from_folder(images_folder)

# Cargar metadatos
metadata_file = 'metadatos_actualizados_sin_nv_reducidos.csv'  # Cambia esto a la ruta correcta
metadata_df = pd.read_csv(metadata_file)

# Preprocesamiento de metadatos (clasificación de edad, normalización, One-Hot Encoding)
metadata_df = metadata_df.dropna(subset=['age']).drop_duplicates()
bins = [0, 18, 35, 50, 65, 100]
labels = [0, 1, 2, 3, 4]
metadata_df['age_group'] = pd.cut(metadata_df['age'], bins=bins, labels=labels)

# Normalización edad y One-Hot Encoding para 'sex'
scaler = StandardScaler()
age_normalized = scaler.fit_transform(metadata_df['age'].values.reshape(-1, 1))

encoder = OneHotEncoder(sparse_output=False)
sex_encoded = encoder.fit_transform(metadata_df['sex'].str.lower().values.reshape(-1, 1))

# Concatenar metadatos
metadata = np.concatenate([age_normalized, sex_encoded], axis=1)

# Etiquetas para clasificación
labels = pd.get_dummies(metadata_df['classification']).values

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

# Transformación de las imágenes
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
])

dataset = CustomDataset(images, metadata, labels, transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Cargar Mask R-CNN preentrenado
mask_rcnn_model = maskrcnn_resnet50_fpn(weights='DEFAULT')

# Ajustar el número de clases
num_classes = 2  # Cambia según el número de clases en tu problema
in_features = mask_rcnn_model.roi_heads.box_predictor.cls_score.in_features
mask_rcnn_model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Ajustar el predictor de máscaras
in_features_mask = mask_rcnn_model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
mask_rcnn_model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

# Configurar el modelo en modo entrenamiento
mask_rcnn_model.train()

# Inicializar el optimizador
optimizer = torch.optim.Adam(mask_rcnn_model.parameters(), lr=0.001)

# Ejemplo de entrenamiento
for images, metadata, labels in train_loader:
    targets = []
    
    for label in labels:
        # Asegúrate de que los cuadros delimitadores tengan coordenadas válidas
        box = torch.rand(1, 4) * torch.tensor([image_width, image_height, image_width, image_height])  # Escalando a las dimensiones de la imagen
        
        # Cambiar la lógica de comparación usando paréntesis
        valid_box = box[(box[:, 2] > box[:, 0]) & (box[:, 3] > box[:, 1])]  # Asegurarse de que la altura y el ancho sean positivos
        
        if valid_box.numel() == 0:
            box = torch.tensor([[0, 0, 1, 1]])  # Si no hay coordenadas válidas, usa una caja mínima
        else:
            box = valid_box
        
        # Generar máscara aleatoria (esto es solo un ejemplo, deberías usar tus propias máscaras)
        mask = torch.rand(1, image_height, image_width) > 0.5  # Generar una máscara aleatoria
        targets.append({"boxes": box,
                        "labels": torch.tensor(label, dtype=torch.int64),
                        "masks": mask.float()})

    # Aplicar Mask R-CNN
    loss_dict = mask_rcnn_model(images, targets)
    losses = sum(loss for loss in loss_dict.values())

    # Optimización
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

# Configuración de K-Fold Cross Validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"Fold {fold + 1}")
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

    # Entrenar Mask R-CNN en cada fold
    for epoch in range(10):  # Ajusta el número de épocas según necesites
        mask_rcnn_model.train()
        epoch_loss = 0

        for images, metadata, labels in train_loader:
            targets = []
            
            for label in labels:
                box = torch.rand(1, 4) * torch.tensor([image_width, image_height, image_width, image_height])
                valid_box = box[(box[:, 2] > box[:, 0]) & (box[:, 3] > box[:, 1])]
                if valid_box.numel() == 0:
                    box = torch.tensor([[0, 0, 1, 1]])
                else:
                    box = valid_box
                
                mask = torch.rand(1, image_height, image_width) > 0.5  # Generar una máscara aleatoria
                
                targets.append({"boxes": box,
                                "labels": torch.tensor(label, dtype=torch.int64),
                                "masks": mask.float()})

            # Aplicar Mask R-CNN
            loss_dict = mask_rcnn_model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader)}')

    # Evaluación del modelo
    y_true, y_pred = [], []
    mask_rcnn_model.eval()
    with torch.no_grad():
        for images, metadata, labels in val_loader:
            output = mask_rcnn_model(images)
            pred_labels = [torch.argmax(out['scores']).item() for out in output]
            y_true.extend(labels.numpy())  # Asegúrate de que las etiquetas sean numpy
            y_pred.extend(pred_labels)

    accuracy = accuracy_score(y_true, y_pred)
    print(f'Fold {fold + 1} Accuracy: {accuracy:.4f}')

# Guardar el modelo entrenado
torch.save(mask_rcnn_model.state_dict(), 'mask_rcnn_model.pth')
