import cv2
import os
from matplotlib import pyplot as plt
import numpy as np

# Ruta a la carpeta que contiene las imágenes
folder_path = 'Skin Cancer/Skin Cancer'  # Reemplaza con la ruta a tu carpeta
output_folder = 'output_images'  # Carpeta para guardar imágenes procesadas

# Crear la carpeta de salida si no existe
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Inicializar listas para almacenar estadísticas globales
all_means = []
all_stddevs = []
histograms = []
aspect_ratios = []
dimensions = []
image_count = 0

# Número de imágenes a mostrar
max_visualize = 5

# Función para procesar cada imagen
def process_image(image_path):
    global image_count
    # Cargar la imagen
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error al cargar la imagen: {image_path}")
        return None
    
    # Convertir la imagen a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 1. Calcular estadísticas de píxeles
    mean, stddev = cv2.meanStdDev(image_rgb)
    
    # 2. Histograma de cada canal (R, G, B)
    hist_r = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])

    # 3. Detección de bordes usando el detector de Canny
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 100, 200)
    # Guardar la imagen de bordes
    edges_filename = os.path.join(output_folder, f"{os.path.basename(image_path).split('.')[0]}_edges.jpg")
    cv2.imwrite(edges_filename, edges)

    # 4. Análisis de formas y texturas usando Sobel y Laplaciano
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    # Guardar las imágenes de Sobel y Laplaciano
    sobel_x_filename = os.path.join(output_folder, f"{os.path.basename(image_path).split('.')[0]}_sobel_x.jpg")
    sobel_y_filename = os.path.join(output_folder, f"{os.path.basename(image_path).split('.')[0]}_sobel_y.jpg")
    laplacian_filename = os.path.join(output_folder, f"{os.path.basename(image_path).split('.')[0]}_laplacian.jpg")
    cv2.imwrite(sobel_x_filename, sobel_x)
    cv2.imwrite(sobel_y_filename, sobel_y)
    cv2.imwrite(laplacian_filename, laplacian)

    # 5. Relación de aspecto y dimensiones
    height, width, _ = image.shape
    aspect_ratio = width / height
    dimensions.append((width, height))
    aspect_ratios.append(aspect_ratio)

    # Acumular estadísticas si los resultados no son None
    if mean is not None and stddev is not None:
        all_means.append(mean.flatten())
        all_stddevs.append(stddev.flatten())
        histograms.append((hist_r, hist_g, hist_b))
        
    image_count += 1

# Procesar todas las imágenes de la carpeta
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Asegúrate de tener solo imágenes
        image_path = os.path.join(folder_path, filename)
        process_image(image_path)

# Convertir listas a arrays de numpy para cálculos
all_means = np.array(all_means)
all_stddevs = np.array(all_stddevs)

# Calcular la media y desviación estándar general para todas las imágenes
global_mean = np.mean(all_means, axis=0)
global_stddev = np.mean(all_stddevs, axis=0)
average_aspect_ratio = np.mean(aspect_ratios)
average_dimensions = np.mean(dimensions, axis=0)

# Mostrar las estadísticas generales
print("\n=== Estadísticas Generales ===")
print("Número total de imágenes procesadas:", image_count)
print("Media global de colores (R, G, B):", global_mean)
print("Desviación estándar global de colores (R, G, B):", global_stddev)
print("Relación de aspecto promedio:", average_aspect_ratio)
print("Dimensiones promedio (Ancho, Alto):", average_dimensions)

# Mostrar resultados de ejemplos de histogramas y bordes
if histograms:
    for i, (hist_r, hist_g, hist_b) in enumerate(histograms[:1]):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.plot(hist_r, color='r')
        plt.title('Histograma Rojo')
        plt.xlabel('Valor de píxeles')
        plt.ylabel('Frecuencia')

        plt.subplot(1, 3, 2)
        plt.plot(hist_g, color='g')
        plt.title('Histograma Verde')
        plt.xlabel('Valor de píxeles')
        plt.ylabel('Frecuencia')

        plt.subplot(1, 3, 3)
        plt.plot(hist_b, color='b')
        plt.title('Histograma Azul')
        plt.xlabel('Valor de píxeles')
        plt.ylabel('Frecuencia')

        plt.show()

    print(f"\nHistograma de colores visualizado para imagen ejemplo.")

print("\nResumen del análisis:")
print(f"Total de imágenes procesadas: {image_count}")
print(f"Relación de aspecto promedio: {average_aspect_ratio:.2f}")
print(f"Dimensiones promedio: {average_dimensions}")
