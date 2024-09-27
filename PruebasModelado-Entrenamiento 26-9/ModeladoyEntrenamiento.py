import os
import pandas as pd
import numpy as np
import tensorflow as tf
import h5py
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

#import seaborn as sns
import matplotlib.pyplot as plt

# Función para transformar la imagen en un tensor
def load_image(image_name, image_dir):
    image_path = os.path.join(image_dir, image_name)
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [112, 112])  # Ajustar el tamaño si es necesario
    img = img / 255.0  # Normalizar
    return img.numpy()  # Convertir a numpy array para guardar

# Cargar el dataset desde HDF5

with h5py.File('datasetCompleto_lesiones.h5', 'r') as hdf:
    imagenes = hdf['imagenes'][:]
    etiquetas_dx = hdf['dx_labels'][:]
    #etiquetas_dx_Type = hdf['dx_type_labels'][:]
    #metadatos = hdf['metadatos'][:]

# A Dividir el dataset en entrenamiento y test (80% entrenamiento, 20% test)
imagenes_train, imagenes_test, etiquetas_train, etiquetas_test = train_test_split(
    #imagenes, etiquetas_dx, test_size=0.2, random_state=42, stratify=etiquetas_dx  # Stratify para mantener el balance
    imagenes, etiquetas_dx, test_size=0.2, random_state=42
)

"""
   # Definir un modelo sencillo de CNN usando chatgpt de ayuda
model = Sequential([
       Conv2D(32, (3, 3), activation='relu', input_shape=(112, 112, 3)),
       MaxPooling2D(pool_size=(2, 2)),
       Flatten(),
       Dense(128, activation='relu'),
       Dense(6, activation='softmax')  
   ])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

"""
# Definición del modelo CNN propuesto en el notebook dado por los profes
model_cnn = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(112, 112, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

   # Compilar el modelo del notebook
model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_cnn.summary()

   # Entrenar el modelo sencillo
'''
model.fit(imagenes_train, etiquetas_train, epochs=20, batch_size=20)
test_loss, test_acc = model.evaluate(imagenes_test, etiquetas_test)
'''
#history_cnn = model_cnn.fit(imagenes_train, etiquetas_train, epochs=15, batch_size=20, validation_data=(imagenes_test, etiquetas_test))
#history_cnn = model_cnn.fit(imagenes_train, etiquetas_train, epochs=20, batch_size=20, validation_data=(imagenes_test, etiquetas_test))
# agregarmos parametro para detener entrenamiento cuando empiece a sobreajustra
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
history_cnn = model_cnn.fit(imagenes_train, etiquetas_train, epochs=15, batch_size=20, validation_data=(imagenes_test, etiquetas_test),callbacks=[early_stopping])

test_loss, test_acc = model_cnn.evaluate(imagenes_test, etiquetas_test)

print("Accuracy del modelo CNN en el conjunto de prueba:", test_acc)

# Evaluación del modelo

# 
# Gráficas de entrenamiento y validación
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_cnn.history['accuracy'], label='Precisión en entrenamiento')
plt.plot(history_cnn.history['val_accuracy'], label='Precisión en validación')
plt.title('Curva de Precisión')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_cnn.history['loss'], label='Pérdida en entrenamiento')
plt.plot(history_cnn.history['val_loss'], label='Pérdida en validación')
plt.title('Curva de Pérdida')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.show()


'''
### Predecir 

# Cargar la imagen
imagen = 'ISIC_0024389.jpg' ### esta imagen tiene como dx 'nv'
#img = image.load_img(ruta_imagen, target_size=(112, 112))
image_dir='imagenesParaValidar'

imagen_Cargada= load_image(imagen, image_dir)
imagen_transformada = resize(imagen_Cargada, (112, 112), anti_aliasing=True)

# Convertir la imagen a un array de NumPy
img_array = image.img_to_array(imagen_transformada)

# Redimensionar la imagen (opcional si ya está en el tamaño correcto)
img_array = np.expand_dims(img_array, axis=0)

# Preprocesar la imagen de la misma manera que se hizo durante el entrenamiento (si es necesario)
# Esto puede incluir normalización o escalado. Aquí se usa preprocess_input, pero ajusta según tu caso
# img_array = preprocess_input(img_array)

# Realizar la predicción
prediccion = model.predict(img_array)

# Obtener la clase predicha
clase_predicha = np.argmax(prediccion, axis=1)
# Mapear la clase predicha a la etiqueta original usando las columnas
print(f"Clase predicha: {clase_predicha[0]}")
etiqueta_original = nombres_clases[clase_predicha[0]]
print(f"Etiqueta original para la clase predicha {clase_predicha}: {etiqueta_original}")

'''