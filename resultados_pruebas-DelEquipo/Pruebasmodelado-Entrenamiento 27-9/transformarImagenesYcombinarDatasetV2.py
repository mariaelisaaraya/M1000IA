###este script trabaja con el dataset convinado
#### 1. **Transformar imagenes a datos numéricos y combinarlas con los  metadatos en un unico dataset**:
###al final me quedo solo con las etiquetas
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import h5py
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input


# leer archivo con metadatos


df = pd.read_csv('dataset_unido3.csv')

#df['image_name'] = df['image_id'].apply(lambda x: x + '.jpg') 
df['image_name'] = df['image_id']
# Función para transformar la imagen en un tensor
def load_image(image_name, image_dir):
    image_path = os.path.join(image_dir, image_name)
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [112, 112])  # Ajustar el tamaño si es necesario
    img = img / 255.0  # Normalizar
    return img.numpy()  # Convertir a numpy array para guardar

# Lista de imágenes transformadas
image_dir = 'Skin Cancer'  # Directorio donde se encuentran las imágenes
#image_dir = 'imagenesPrueba'  # Directorio donde se encuentran las imágenes
# aplicar la función load_image a cada imagen en el directorio para transformar cada imagen en un array numérico
imagenes_transformandas = [load_image(image_name, image_dir) for image_name in df['image_name']]

# Convertir la lista de imágenes en un array de numpy
imagenes_array = np.stack(imagenes_transformandas)


# Convertir etiquetas a formato numérico si es necesario (ejemplo con codificación one-hot)
etiquetas = pd.get_dummies(df['dx']).values  # Convertir etiquetas a one-hot encoding
et=pd.get_dummies(df['dx'])

#imprimir las columnas generadas
print("Columnas generadas por pd.get_dummies:")
nombres_clases = list(et.columns)
print(nombres_clases)

# Guardar las imágenes y etiquetas en un archivo HDF5
#with h5py.File('dataset_lesiones.h5', 'w') as hdf: 
with h5py.File('datasetAumentado_lesiones.h5', 'w') as hdf:
    hdf.create_dataset('imagenes', data=imagenes_array)
    hdf.create_dataset('dx_labels', data=etiquetas)



