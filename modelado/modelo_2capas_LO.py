import os
import pandas as pd
import numpy as np
import tensorflow as tf
import h5py
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from tensorflow.keras import layers, models
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras import regularizers

# Cargar el archivo
file_path = 'PruebasLucia/5021_dataset_con_metadatos.h5'
with h5py.File(file_path, 'r') as f:
    images = np.array(f['images'])  
    sex = np.array(f['sex'])        
    age = np.array(f['age'])        
    labels = np.array(f['classification'])  

# Normalizar el metadato edad
scaler = StandardScaler()
age_normalized = scaler.fit_transform(age.reshape(-1, 1))

# One-Hot Encoding para sexo
encoder = OneHotEncoder(sparse_output=False)
sex_encoded = encoder.fit_transform(sex.reshape(-1, 1))

# Concatenar los metadatos
metadata = np.concatenate([age_normalized, sex_encoded], axis=1)

# Dividir entre entrenamiento (70%) y test+validación (30%)
X_train, X_test, metadata_train, metadata_test, y_train, y_test = train_test_split(
    images, metadata, labels, test_size=0.30, random_state=42)

# Luego dividir la parte test en validación (10%) y prueba (20%)
X_val, X_test, metadata_val, metadata_test, y_val, y_test = train_test_split(
    X_test, metadata_test, y_test, test_size=2/3, random_state=42)

# Definir las dimensiones de las imágenes
image_height, image_width, channels = images.shape[1], images.shape[2], images.shape[3]

# Definir la entrada de imágenes y la CNN
input_img = layers.Input(shape=(image_height, image_width, channels))
x = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001))(input_img)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.3)(x)
x = layers.Flatten()(x)

# Definir la entrada de metadatos
input_metadata = layers.Input(shape=(metadata_train.shape[1],))
metadata_dense = layers.Dense(64, activation='relu')(input_metadata)

# Combinar las salidas de la CNN y la red densa
combined = layers.concatenate([x, metadata_dense])

# Añadir capas densas finales para la clasificación
z = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(combined)
x = layers.Dropout(0.5)(x)
output = layers.Dense(2, activation='softmax')(z)

# Crear el modelo
model_cnn_metadatos2 = models.Model(inputs=[input_img, input_metadata], outputs=output)

# Compilar el modelo con categorical_crossentropy
model_cnn_metadatos2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Definir el callback personalizado
class PrintEpoch(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f'Epoch {epoch + 1} ended. Loss: {logs["loss"]:.4f}, Accuracy: {logs["accuracy"]:.4f}')

# Crear una instancia del callback personalizado
print_epoch_callback = PrintEpoch()

# Entrenar el modelo
history = model_cnn_metadatos2.fit(
    [X_train, metadata_train], y_train,
    validation_data=([X_val, metadata_val], y_val),
    epochs=20, batch_size=32,
    callbacks=[print_epoch_callback]
)

print('Entrenamiento finalizado')   

# Evaluar en el conjunto de prueba para generar compile_metrics
test_loss, test_acc = model_cnn_metadatos2.evaluate([X_test, metadata_test], y_test)
print(f'Accuracy en test: {test_acc:.2f}')

# Predecir en el conjunto de prueba para obtener métricas adicionales
y_pred = model_cnn_metadatos2.predict([X_test, metadata_test])
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Imprimir matriz de confusión
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
print("Matriz de Confusión:\n", conf_matrix)

# Imprimir reporte de clasificación
class_report = classification_report(y_true_classes, y_pred_classes)
print("Reporte de Clasificación:\n", class_report)

# Guardar el historial de entrenamiento
import json
with open('training_history.json', 'w') as f:
    json.dump(history.history, f)

# Guardar el modelo completo en formato SavedModel y .h5
model_cnn_metadatos2.save('modelo_2_capas_LO_tf.keras')
model_cnn_metadatos2.save('modelo_2_capas_LO_tf.h5')



# Compilar el modelo con las métricas
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Evaluar el modelo con los datos reales
test_loss, test_accuracy = model.evaluate([X_test, metadata_test], y_test)

print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")