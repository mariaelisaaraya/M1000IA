import os
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight

# Define rutas
image_folder_path = 'carpeta_nueva_img'
metadata_file_path = 'metadatos_actualizados_sin_nv_reducidos.csv'

# Verifica si la carpeta de imágenes existe
if not os.path.exists(image_folder_path):
    print(f"Error: La carpeta de imágenes '{image_folder_path}' no se encuentra.")
else:
    print(f"Carpeta de imágenes '{image_folder_path}' cargada correctamente.")

# Carga los metadatos
try:
    metadata = pd.read_csv(metadata_file_path)
    print(f"Archivo de metadatos '{metadata_file_path}' cargado correctamente.")
    print(f"Número de registros en el archivo de metadatos: {len(metadata)}")
    print(f"Nombres de las etiquetas en la columna 'classification': {metadata['classification'].unique()}")
except FileNotFoundError:
    print(f"Error: El archivo de metadatos '{metadata_file_path}' no se encuentra.")
    metadata = None

# Inicializa listas para imágenes y etiquetas
images = []
labels = []
sex_list = []
age_list = []

# Define el tamaño de la imagen
height, width = 128, 128

# Carga las imágenes y etiquetas
if metadata is not None:
    for _, row in metadata.iterrows():
        img_path = os.path.join(image_folder_path, row['image_id'])
        if os.path.exists(img_path):
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(height, width))
            img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            images.append(img)
            labels.append(row['classification'])
            sex_list.append(row['sex'])
            age_list.append(row['age'])
        else:
            print(f"Advertencia: La imagen '{img_path}' no se encuentra en la carpeta.")

# Convertir a arrays de NumPy
images = np.array(images)
labels = np.array(labels)
sex = np.array(sex_list)
age = np.array(age_list)

print(f"Número total de imágenes cargadas: {len(images)}")
print(f"Número total de etiquetas cargadas: {len(labels)}")

# Normaliza el metadato edad
scaler = StandardScaler()
age_normalized = scaler.fit_transform(age.reshape(-1, 1).astype(float))
joblib.dump(scaler, 'standard_scaler.pkl')

# One-Hot Encoding para sex
encoder = OneHotEncoder(sparse_output=False)
sex_encoded = encoder.fit_transform(sex.reshape(-1, 1))

# Convierte etiquetas a formato numérico
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Concatena los metadatos
metadata = np.concatenate([age_normalized, sex_encoded], axis=1)

# 70% para entrenamiento, 15% para validación, 15% para test
X_train, X_temp, metadata_train, metadata_temp, y_train, y_temp = train_test_split(
    images, metadata, labels_encoded, test_size=0.30, random_state=42)

X_val, X_test, metadata_val, metadata_test, y_val, y_test = train_test_split(
    X_temp, metadata_temp, y_temp, test_size=0.5, random_state=42)

# Calcula los pesos de clase para balancear
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

# nuevo tamaño del batch
batch_size = 16  # Cambiado a 16, antes 32

# Definir el generador de aumento de datos con ajustes
data_gen_maligna = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],
    channel_shift_range=30.0
)

# Definir el generador de datos
def data_generator(images, metadata, labels, batch_size):
    num_samples = len(images)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_images = images[offset:offset + batch_size]
            batch_metadata = metadata[offset:offset + batch_size]
            batch_labels = labels[offset:offset + batch_size]

            augmented_images = []

            for img, label in zip(batch_images, batch_labels):
                if label_encoder.inverse_transform([label])[0] == 'lesión maligna':
                    img = data_gen_maligna.random_transform(img)
                augmented_images.append(img)

            augmented_images = np.array(augmented_images)
            yield (augmented_images, batch_metadata), batch_labels

output_signature = (
    (tf.TensorSpec(shape=(None, height, width, 3), dtype=tf.float32),
     tf.TensorSpec(shape=(None, metadata.shape[1]), dtype=tf.float32)),
    tf.TensorSpec(shape=(None,), dtype=tf.int32)
)

train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(X_train, metadata_train, y_train, batch_size=batch_size),
    output_signature=output_signature
)

val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(X_val, metadata_val, y_val, batch_size=batch_size),
    output_signature=output_signature
)

# Definir la red CNN con más capas convolucionales, tambien las ajuste y puse nuevas capas
input_img = layers.Input(shape=(height, width, 3))
x = layers.Conv2D(8, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_img)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)

input_metadata = layers.Input(shape=(metadata.shape[1],))
metadata_dense = layers.Dense(64, activation='relu')(input_metadata)

combined = layers.concatenate([x, metadata_dense])
z = layers.Dense(64, activation='relu')(combined)
z = layers.Dropout(0.5)(z)
output = layers.Dense(len(np.unique(labels_encoded)), activation='softmax')(z)

model = models.Model(inputs=[input_img, input_metadata], outputs=output)

# Configura el optimizador con una tasa de aprendizaje inicial
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Configura EarlyStopping y ReduceLROnPlateau
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    verbose=1,
    mode='max',
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1,
    mode='min'
)

# Entrena el modelo con los pesos de clase
history = model.fit(
    train_dataset,
    steps_per_epoch=len(X_train) // batch_size,
    validation_data=val_dataset,
    validation_steps=len(X_val) // batch_size,
    epochs=50,
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weights
)

# Definir el generador de datos para el conjunto de prueba
test_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(X_test, metadata_test, y_test, batch_size=batch_size),
    output_signature=output_signature
)

# Evalua el modelo con pasos definidos
y_pred = model.predict(test_dataset, steps=len(X_test) // batch_size)
y_pred_classes = np.argmax(y_pred, axis=1)

# Ajustar las longitudes de y_test y y_pred_classes
if len(y_test) != len(y_pred_classes):
    print(f"Ajustando las longitudes de y_test ({len(y_test)}) y y_pred_classes ({len(y_pred_classes)})")
    min_len = min(len(y_test), len(y_pred_classes))
    y_test = y_test[:min_len]
    y_pred_classes = y_pred_classes[:min_len]

# Mostrar el informe de clasificación y la matriz de confusión
print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

conf_matrix = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicho')
plt.title('Matriz de Confusión')
plt.show()

# Guarda el modelo
model.save('modelo.h5')


# Calcula la matriz de confusión y el reporte de clasificación
conf_matrix = confusion_matrix(y_test, y_pred_classes)
print("Matriz de Confusión:\n", conf_matrix)
print("Reporte de clasificación:\n", classification_report(y_test, y_pred_classes))

# Visualiza la matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.ylabel('Etiqueta verdadera')
plt.xlabel('Etiqueta predicha')
plt.title('Matriz de Confusión')
plt.show()

# Imprime el informe de clasificación
print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))



# Guarda el modelo
model.save('modelo_final.h5')
print("Modelo guardado como 'modelo_final.h5'")


# Guarda los codificadores
joblib.dump(scaler, 'standard_scaler.pkl')
joblib.dump(encoder, 'encoder.pkl')
joblib.dump(encoder, 'onehot_encoder.pkl')
# Guarda los pesos del modelo
model.save_weights('modelodai_pesos.weights.h5')

#  guardar el modelo completo en version keras x los warning
model.save('dermai_model.keras')

# Grafica el historial de entrenamiento
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
plt.title('Precisión del Modelo')
plt.ylabel('Precisión')
plt.xlabel('Épocas')
plt.legend()
plt.show()



