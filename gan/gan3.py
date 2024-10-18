import os
import numpy as np
import pandas as pd
from torchvision import transforms
from torchvision.utils import save_image
import torch
from PIL import Image
import glob
import random

# Define parámetros
channels = 3  # Imágenes RGB
image_height = 128  # Altura de la imagen
image_width = 128   # Ancho de la imagen
num_images = 1000  # Número de imágenes a generar
output_dir = 'generated_images_gan'
os.makedirs(output_dir, exist_ok=True)  # Crear directorio para guardar imágenes

# Función para cargar imágenes reales desde un directorio sin estructura de carpetas
def cargar_tus_imagenes_reales(data_dir):
    transform = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizar a [-1, 1]
    ])
    
    # Cargar imágenes desde el directorio
    image_paths = glob.glob(os.path.join(data_dir, '*.jpg'))  # Cambia a '*.png' si es necesario
    images = []
    
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')  # Asegurarse de que la imagen sea RGB
        image = transform(image)
        images.append(image)
    
    return torch.stack(images)  # Apilar imágenes en un tensor

import matplotlib.pyplot as plt

def mostrar_imagen(imagen):
    plt.imshow(imagen.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.show()


# Función para generar nuevas imágenes basadas en las imágenes reales cargadas
def generate_images_from_real(data_dir):
    real_images = cargar_tus_imagenes_reales(data_dir)

    # Transformaciones ajustadas
    augmentation_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=10),  # Menor rotación
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Ajustes menores
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.25, 0.25, 0.25), (0.25, 0.25, 0.25)),  # Normaliza a [-1, 1]
    ])

    for i in range(num_images):
        index = np.random.randint(0, real_images.size(0))
        new_image = real_images[index].clone()
        
        # Aplicar las transformaciones
        new_image = augmentation_transform(new_image)

        # Guardar la imagen generada
        save_image(new_image, os.path.join(output_dir, f'generated_{i}.jpg'), normalize=True)
        
        # Visualizar la imagen generada
        mostrar_imagen(new_image)



# Función para crear metadatos variados
def create_metadata(data_dir, num_images):
    # Cargar los metadatos existentes
    metadata_df = pd.read_csv(r'/home/a3lisa/Documentos/Prog/IA/M1000IA/metadatos_eda/16_10_metadatos_actualizados_sin_nv_reducidos.csv') #Se cambia por tu local
    print("Contenido de los metadatos:")
    print(metadata_df.head())  # Verificar el contenido del DataFrame

    # Filtrar solo las lesiones malignas
    malignant_metadata = metadata_df[metadata_df['classification'] == 'lesión maligna']
    print("Número de entradas malignas:", malignant_metadata.shape[0])  # Número de entradas malignas

    if malignant_metadata.shape[0] > 0:
        # Generar metadatos variados para las imágenes
        generated_metadata = []
        
        for _ in range(num_images):
            # Seleccionar una entrada aleatoria del metadata maligno
            entry = malignant_metadata.sample().iloc[0]
            
            # Crear variaciones en los metadatos
            lesion_id = f'ham_{np.random.randint(1000000)}'
            image_id = f'generated_{_}.jpg'
            dx = entry['dx']  # Mantener el mismo diagnóstico
            age = entry['age'] + random.randint(-5, 5)  # Variar la edad
            sex = random.choice(['male', 'female'])  # Alternar el sexo
            localization = entry['localization']  # Mantener la localización o variarla si deseas
            
            # Añadir el nuevo registro a la lista
            generated_metadata.append({
                'lesion_id': lesion_id,
                'image_id': image_id,
                'dx': dx,
                'age': age,
                'sex': sex,
                'localization': localization,
                'classification': '1',  # Solo lesiones malignas
                'dataset': 'synthetic'  # Etiqueta como sintético
            })
        
        # Crear un DataFrame de metadatos generados
        generated_metadata_df = pd.DataFrame(generated_metadata)

        # Guardar los metadatos generados en un archivo CSV
        generated_metadata_df.to_csv(os.path.join(output_dir, 'metadatagan.csv'), index=False)
        print("Metadatos generados y guardados con éxito.")
    else:
        raise ValueError("No malignant entries found in metadata.")

# Especifica el directorio con tus imágenes
data_dir = '/home/a3lisa/Documentos/Prog/IA/M1000IA/imagenes/imagenesConRecorte_5021img'  #Se cambia por tu local
generate_images_from_real(data_dir)
create_metadata(data_dir, num_images)
