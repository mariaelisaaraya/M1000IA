import pandas as pd
import os
import shutil

# Rutas de archivo
archivo_metadatos = 'metadatos_actualizados_sin_nv_reducidos.csv'  # Cambia esta ruta según sea necesario
carpeta_imagenes = ''  # Cambia esta ruta según sea necesario
carpeta_nueva = ''  # Cambia esta ruta según sea necesario

# Crear la nueva carpeta si no existe
os.makedirs(carpeta_nueva, exist_ok=True)

# Cargar los metadatos desde el CSV
metadatos = pd.read_csv(archivo_metadatos)

# Obtener la lista de los 'image_id' que están en el archivo de metadatos
image_ids_con_metadatos = set(metadatos['image_id'])

# Recorrer la carpeta de imágenes y copiar las que están en los metadatos
for imagen in os.listdir(carpeta_imagenes):
    # Verificar si la imagen está en los metadatos
    if imagen in image_ids_con_metadatos:
        # Construir la ruta completa de la imagen
        ruta_imagen = os.path.join(carpeta_imagenes, imagen)
        # Copiar la imagen a la nueva carpeta
        shutil.copy(ruta_imagen, carpeta_nueva)
        print(f"Copiada: {ruta_imagen} a {carpeta_nueva}")

print("Proceso de copiado de imágenes con metadatos completado.")
