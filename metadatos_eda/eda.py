import cv2
import os
from matplotlib import pyplot as plt
import numpy as np

# Ruta a la carpeta que contiene las imágenes
folder_path = 'Skin Cancer\Skin Cancer'  # Reemplaza con la ruta a tu carpeta

# Inicializar listas para almacenar estadísticas globales
all_means = []
all_stddevs = []
image_count = 0

# Número de imágenes a mostrar
max_visualize = 5

# Función para procesar cada imagen
def process_image(image_path, visualize=False):
    # Cargar la imagen
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error al cargar la imagen: {image_path}")
        return None
    
    # Convertir la imagen a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Mostrar la imagen si se requiere
    if visualize:
        plt.imshow(image_rgb)
        plt.title(f"Imagen: {os.path.basename(image_path)}")
        plt.axis('off')
        plt.show()

    # Calcular estadísticas (media y desviación estándar de los colores)
    mean, stddev = cv2.meanStdDev(image)
    return mean, stddev

# Procesar todas las imágenes de la carpeta
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Asegúrate de tener solo imágenes
        image_path = os.path.join(folder_path, filename)
        
        # Visualizar las primeras 5 imágenes
        if image_count < max_visualize:
            mean, stddev = process_image(image_path, visualize=True)
        else:
            mean, stddev = process_image(image_path, visualize=False)
        
        # Acumular estadísticas
        if mean is not None and stddev is not None:
            all_means.append(mean.flatten())
            all_stddevs.append(stddev.flatten())
        
        image_count += 1

# Convertir listas a arrays de numpy para cálculos
all_means = np.array(all_means)
all_stddevs = np.array(all_stddevs)

# Calcular la media y desviación estándar general para todas las imágenes
global_mean = np.mean(all_means, axis=0)
global_stddev = np.mean(all_stddevs, axis=0)

# Mostrar las estadísticas generales
print("Media global de colores (B, G, R):", global_mean)
print("Desviación estándar global de colores (B, G, R):", global_stddev)

# Opcional: Puedes mostrar gráficos de las estadísticas si lo deseas.
plt.figure(figsize=(10,5))

# Histograma de las medias
plt.subplot(1, 2, 1)
plt.hist(all_means[:, 0], bins=30, color='blue', alpha=0.5, label='Canal Azul')
plt.hist(all_means[:, 1], bins=30, color='green', alpha=0.5, label='Canal Verde')
plt.hist(all_means[:, 2], bins=30, color='red', alpha=0.5, label='Canal Rojo')
plt.title("Distribución de medias de color")
plt.legend()

# Histograma de las desviaciones estándar
plt.subplot(1, 2, 2)
plt.hist(all_stddevs[:, 0], bins=30, color='blue', alpha=0.5, label='Canal Azul')
plt.hist(all_stddevs[:, 1], bins=30, color='green', alpha=0.5, label='Canal Verde')
plt.hist(all_stddevs[:, 2], bins=30, color='red', alpha=0.5, label='Canal Rojo')
plt.title("Distribución de desviaciones estándar de color")
plt.legend()

plt.tight_layout()
plt.show()
