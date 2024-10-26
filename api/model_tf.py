
import tensorflow as tf

def load_model():
    # Cargar el modelo de TensorFlow guardado en formato .h5 o SavedModel
    model = tf.keras.models.load_model('modelo_2_capas_LO_tf.h5')  # Cambia la ruta si usas SavedModel
    return model

def check_model_state():
    model = load_model()
    model.summary()  # Imprimir la arquitectura del modelo
    return model
