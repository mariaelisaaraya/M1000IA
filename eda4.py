import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage import exposure
from skimage.filters import threshold_otsu
from sklearn.preprocessing import StandardScaler
import os

# Ruta a la carpeta que contiene las imágenes
folder_path = 'Skin Cancer'
output_folder = 'visualizaciones'  # Carpeta para guardar visualizaciones

# Crear la carpeta de salida si no existe
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Crear una lista para almacenar las características
features = []
labels = []  # Asumiendo que tienes etiquetas para cada imagen


def resize_image(image, new_size=(256, 256)):
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

def extract_features(image_path, filename):
    # Cargar la imagen y convertir a escala de grises
    image = cv2.imread(image_path)
    image = resize_image(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Extraer características de textura usando HOG
    hog_features, hog_image = hog(gray_image, visualize=True, block_norm='L2-Hys')
    features.append(hog_features)
    
    # Imprimir histogramas de textura
    contrast = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    dissimilarity = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    
    print(f"\nHistogramas de Textura para {filename}:")
    print(f"Contraste (Primeros 10 valores): {contrast[:10].flatten()}")
    print(f"Dissimilarity (Primeros 10 valores): {dissimilarity[:10].flatten()}")
    
    # Guardar imagen HOG
    hog_filename = os.path.join(output_folder, f"{filename.split('.')[0]}_hog.jpg")
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Imagen Original')
    
    plt.subplot(1, 2, 2)
    plt.imshow(hog_image, cmap='gray')
    plt.title('Imagen HOG')
    plt.savefig(hog_filename)
    plt.close()
    
    return gray_image

def segment_color(image, filename):
    # Redimensionar imagen
    image = resize_image(image)
    
    # Convertir la imagen a espacio de color HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Definir un rango de color para la segmentación
    lower_bound = np.array([0, 30, 30])
    upper_bound = np.array([20, 255, 255])
    
    # Aplicar la máscara para la segmentación
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    
    print(f"\nSegmentación Basada en Color para {filename}:")
    print(f"Máscara aplicada. Muestra áreas segmentadas en el color especificado.")
    
    # Guardar segmentación basada en color
    segmented_color_filename = os.path.join(output_folder, f"{filename.split('.')[0]}_seg_color.jpg")
    cv2.imwrite(segmented_color_filename, cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    
    return segmented_image

def segment_otsu(image, filename):
    # Redimensionar imagen
    image = resize_image(image)
    
    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar el umbral de Otsu
    thresh = threshold_otsu(gray_image)
    binary = gray_image > thresh
    
    print(f"\nSegmentación Semántica para {filename}:")
    print(f"Umbral de Otsu calculado: {thresh:.2f}")
    
    # Guardar segmentación semántica
    segmented_semantic_filename = os.path.join(output_folder, f"{filename.split('.')[0]}_seg_semantic.jpg")
    cv2.imwrite(segmented_semantic_filename, binary * 255)  # Convertir a formato de imagen

    return binary

# Procesar las imágenes en la carpeta
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(folder_path, filename)
        print(f"\nProcesando imagen: {filename}")
        
        # Extraer características y mostrar imagen HOG
        gray_image = extract_features(image_path, filename)
        
        # Realizar segmentación basada en color
        image = cv2.imread(image_path)
        segmented_color = segment_color(image, filename)
        
        # Realizar segmentación semántica
        segmented_semantic = segment_otsu(image, filename)
        
        # Mostrar y guardar visualizaciones combinadas
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(segmented_color, cv2.COLOR_BGR2RGB))
        plt.title('Segmentación Basada en Color')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(segmented_semantic, cmap='gray')
        plt.title('Segmentación Semántica')
        plt.axis('off')
        
        segmentations_filename = os.path.join(output_folder, f"{filename.split('.')[0]}_segmentations.jpg")
        plt.savefig(segmentations_filename)
        plt.close()

# Normalizar y mostrar características
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)
print("\nCaracterísticas normalizadas:\n", features_normalized)
