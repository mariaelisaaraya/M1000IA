import os
import shutil
import pandas as pd

def copiar_imagenes(df_path, directorio_imagenes, directorio_destino):
    # Leer el archivo CSV
    df = pd.read_csv(df_path)

    # Crear el directorio de destino si no existe
    os.makedirs(directorio_destino, exist_ok=True)

    # Iterar a través del DataFrame
    for _, fila in df.iterrows():
        image_id = fila['image_id']  # Obtener el ID de la imagen
        nombre_archivo = f"{image_id}.jpg"  # Construir el nombre del archivo jpg
        ruta_origen = os.path.join(directorio_imagenes, nombre_archivo)
        ruta_destino = os.path.join(directorio_destino, nombre_archivo)

        # Verificar si el archivo existe antes de copiar
        if os.path.exists(ruta_origen):
            shutil.copy(ruta_origen, ruta_destino)
            print(f"Copiado: {nombre_archivo} a {directorio_destino}")
        else:
            print(f"No se encontró el archivo: {nombre_archivo}")

# Uso de la función para copiar las imagenes
copiar_imagenes('Mas2IMGs\LesionesCon3IMG.csv', 'Skin Cancer', 'imagenesFiltradas\lesion3IMG')
copiar_imagenes('Mas2IMGs\LesionesCon4IMG.csv', 'Skin Cancer', 'imagenesFiltradas\lesion4IMG')
copiar_imagenes('Mas2IMGs\LesionesCon5IMG.csv', 'Skin Cancer', 'imagenesFiltradas\lesion5IMG')
copiar_imagenes('Mas2IMGs\LesionesCon6IMG.csv', 'Skin Cancer', 'imagenesFiltradas\lesion6IMG')

