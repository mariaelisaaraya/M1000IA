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
from sklearn.utils.class_weight import compute_class_weight

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

# Cargar el dataset desde HDF5 con las imagenes agregadas para balancear

with h5py.File('datasetAumentado_lesiones.h5', 'r') as hdf:
    imagenes = hdf['imagenes'][:]
    etiquetas_dx = hdf['dx_labels'][:]
    #etiquetas_dx_Type = hdf['dx_type_labels'][:]
    #metadatos = hdf['metadatos'][:]


# A Dividir el dataset en entrenamiento y test (80% entrenamiento, 20% test)
imagenes_train, imagenes_test, etiquetas_train, etiquetas_test = train_test_split(
    imagenes, etiquetas_dx, test_size=0.2, random_state=42, stratify=etiquetas_dx 
)




# Definición del modelo CNN propuesto en el notebook dado por los profes
'''
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
'''
from tensorflow.keras.regularizers import l2
'''
model_cnn = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(112, 112, 3), kernel_regularizer=l2(0.01)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(7, activation='softmax')
])
'''
# modelo con mas capas de dropout
model_cnn = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(112, 112, 3), kernel_regularizer=l2(0.01)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),  # Dropout después de la primera capa de MaxPooling
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),  # Dropout después de la segunda capa de MaxPooling
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),   # Dropout ya existente antes de la capa de salida
    Dense(7, activation='softmax')
])

   # Compilar el modelo del notebook
model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_cnn.summary()


# agregarmos parametro para detener entrenamiento cuando empiece a sobreajustra
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

from sklearn.utils import class_weight

# Calcular los pesos de las clases
y_train = np.ravel(etiquetas_train)
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))


history_cnn = model_cnn.fit(
    imagenes_train, 
    etiquetas_train,
    class_weight=class_weights, 
    epochs=20, 
    batch_size=20, 
    validation_data=(imagenes_test, etiquetas_test),
    callbacks=[early_stopping]
)

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



### Predecir 

from sklearn.metrics import classification_report, confusion_matrix


# Suponiendo que ya tienes 'model_cnn' entrenado y las imágenes de prueba y etiquetas
predictions = model_cnn.predict(imagenes_test)

# Convertir predicciones a clases
predicted_classes = np.argmax(predictions, axis=1)

# Convertir etiquetas de prueba a formato unidimensional
true_classes = np.argmax(etiquetas_test, axis=1)

# Calcular y mostrar el informe de clasificación
print(classification_report(true_classes, predicted_classes))

# Calcular y mostrar la matriz de confusión
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print("Matriz de Confusión:")
print(conf_matrix)