import cv2
import os
from matplotlib import pyplot as plt
import numpy as np

# Ruta a la carpeta que contiene las imágenes
folder_path = 'Skin Cancer/Skin Cancer'  # Reemplaza con la ruta a tu carpeta

# Inicializar listas para almacenar estadísticas globales
all_means = []
all_stddevs = []
histograms = []
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

    # Calcular estadísticas de píxeles
    mean, stddev = cv2.meanStdDev(image_rgb)
    
    # Calcular histograma para cada canal (R, G, B)
    hist_r = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])
    
    return mean, stddev, (hist_r, hist_g, hist_b)

# Procesar todas las imágenes de la carpeta
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Asegúrate de tener solo imágenes
        image_path = os.path.join(folder_path, filename)
        
        # Visualizar las primeras 5 imágenes
        if image_count < max_visualize:
            mean, stddev, hist = process_image(image_path, visualize=True)
        else:
            mean, stddev, hist = process_image(image_path, visualize=False)
        
        # Acumular estadísticas si los resultados no son None
        if mean is not None and stddev is not None:
            all_means.append(mean.flatten())
            all_stddevs.append(stddev.flatten())
            histograms.append(hist)
        
        image_count += 1

# Convertir listas a arrays de numpy para cálculos
all_means = np.array(all_means)
all_stddevs = np.array(all_stddevs)

# Calcular la media y desviación estándar general para todas las imágenes
global_mean = np.mean(all_means, axis=0)
global_stddev = np.mean(all_stddevs, axis=0)

# Mostrar las estadísticas generales
print("Media global de colores (R, G, B):", global_mean)
print("Desviación estándar global de colores (R, G, B):", global_stddev)

# Visualizar los histogramas de una imagen
if histograms:
    hist_r, hist_g, hist_b = histograms[0]
    plt.plot(hist_r, color='r', label='Rojo')
    plt.plot(hist_g, color='g', label='Verde')
    plt.plot(hist_b, color='b', label='Azul')
    plt.title('Histograma de los valores de color (R, G, B)')
    plt.xlabel('Valor de píxeles')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.show()
