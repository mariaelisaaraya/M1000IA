
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from model_tf import load_model, check_model_state

# Cargar el modelo de TensorFlow
model = load_model()

# Compilar el modelo después de cargarlo para que las métricas estén disponibles
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#FIXME: sacar la tercera variable

# Realizar una predicción inicial de prueba para activar las métricas
dummy_image = np.zeros((1, 112, 112, 3))  # Imagen de prueba
dummy_metadata = np.zeros((1,3))  # Metadatos de prueba con 3 características
model.predict([dummy_image, dummy_metadata])

# Dimensiones de la imagen
image_height, image_width = 112, 112

# Transformación de la imagen
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((image_height, image_width))
    img = np.array(img) / 255.0  # Normalizar
    img = np.expand_dims(img, axis=0)  # Añadir batch dimension
    return img

# Predict con imagen y metadatos
def predict_with_metadata(image_path, age, sex):
    try:
        check_model_state()
        # model = load_model()
        # Preprocesar la imagen
        image = preprocess_image(image_path)
        
        # Convertir el sexo a numérico
        sex_numeric = 0 if sex == 'male' else 1
        metadata_tensor = np.array([[age, sex_numeric, 0]])
        
        # Añadir un tercer metadato ficticio si es necesario, por ejemplo, 0
        # third_feature = 0  # Esto es solo un valor de ejemplo

        # Asegurarte de que los metadatos tienen forma (None, 3)
        #metadata_tensor = np.array([[age, sex_numeric, third_feature]])  # Cambiar a (1, 3)
        
        # Hacer la predicción con el modelo de TensorFlow
        predictions = model.predict([image, metadata_tensor])
        predicted_class = np.argmax(predictions, axis=1)[0]
        probabilities = predictions[0].tolist()
        return predicted_class, probabilities

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None, None
