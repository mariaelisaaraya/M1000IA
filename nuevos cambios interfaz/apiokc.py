from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from werkzeug.utils import secure_filename
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address



# Inicializar Flask
app = Flask(__name__)

# Configurar CORS
CORS(app)

# Inicializar Flask-Limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["60 per minute"]  # Límite global por minuto
)

# Define la ruta donde se almacenarán las imágenes subidas
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')  # Ejemplo de ruta
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Configura la carpeta de subida
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Función para verificar el formato de la imagen
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Inicializar el modelo
num_metadata_features = 2
num_classes = 2  # Define el número de clases aquí

# Configura las dimensiones de las imágenes
image_height, image_width, channels = 128, 128, 3  

class MultiInputModel(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super(MultiInputModel, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.5)

        # Asegúrate de calcular correctamente el tamaño de entrada para fc1
        self.fc1_input_size = 128 * (image_height // 8) * (image_width // 8)
        self.fc1 = nn.Linear(self.fc1_input_size, 128)
        self.fc_meta = nn.Linear(num_metadata_features, 64)
        self.fc_output = nn.Linear(128 + 64, num_classes)

    def forward(self, img, meta):
        # Verifica la forma de la imagen de entrada
        print(f"Input img shape: {img.shape}")

        x = self.pool(F.relu(self.conv1(img)))
        print(f"After conv1 shape: {x.shape}")

        x = self.pool(F.relu(self.conv2(x)))
        print(f"After conv2 shape: {x.shape}")

        x = self.pool(F.relu(self.conv3(x)))
        print(f"After conv3 shape: {x.shape}")

        x = x.view(x.size(0), -1)  # Asegúrate de que esto esté funcionando como esperas
        print(f"x shape before fc1: {x.shape}")

    # Guardar la forma de x en un archivo de texto
        try:
            with open('shapes_log.txt', 'w') as f:  # Abrir el archivo en modo write para sobreescribir
                f.write(f"x shape before fc1: {x.shape}\n")  # Escribir la forma en el archivo

            # Guardar la forma de meta
            if meta is None:
                raise ValueError("La entrada 'meta' no debe ser None")
            f.write(f"meta shape: {meta.shape}\n")  # Verifica la forma de meta

            # Procesar meta
            meta_out = F.relu(self.fc_meta(meta))
            f.write(f"meta_out shape: {meta_out.shape}\n")  # Verifica la forma de meta_out

            # Concatenar x y meta_out
            combined = torch.cat((x, meta_out), dim=1)
            f.write(f"combined shape: {combined.shape}\n")  # Verifica la forma combinada

            # Salida final
            output = self.fc_output(combined)
            return output  # Asegúrate de retornar aquí

        except Exception as e:
            print(f"Error al escribir en el archivo: {e}")  # Manejo de errores
        return None  # Retorna None en caso de error





model = MultiInputModel(num_metadata_features, num_classes)  # Asegúrate de pasar num_classes


 


# Cargar el modelo entrenado
try:
    model = torch.load('modelo_entrenadomok.pth')
    model.eval()
    print("Modelo cargado exitosamente.")  # Mensaje de éxito al cargar el modelo

    # Imprimir el número de clases del modelo
    num_classes = model.fc_output.out_features  # Obtener el número de clases desde la capa de salida
    print(f"El modelo tiene {num_classes} clases.")

except FileNotFoundError:
    print("Error: El archivo 'modelo_entrenadomok.pth' no se encuentra. Asegúrate de que la ruta sea correcta.")
except RuntimeError as e:
    print(f"Error de ejecución al cargar el modelo: {str(e)}. Asegúrate de que el modelo y los pesos sean compatibles.")
except Exception as e:
    print(f"Error desconocido al cargar el modelo: {str(e)}")

# Función para verificar el estado del modelo
#def check_model_state():
    #state = model.state_dict()
    #for layer, params in state.items():
        #print(f"Layer: {layer}, Weights size: {params.size()}")  
        # # Mostrar el tamaño de los pesos en lugar de imprimir todo


# Transformaciones para la imagen
transform = transforms.Compose([
    transforms.Resize((image_height, image_width, )),
    transforms.ToTensor(),
])

# Función de predicción con imagen y metadatos
def predict_with_metadata(image_path, age, sex):
    try:
        # Cargar y transformar la imagen
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)  # Forma: (1, 3, 128, 128)

        # Crear el tensor de metadatos con 'age' y 'sex'
        if sex not in ['male', 'female']:
            raise ValueError("Sexo debe ser 'male' o 'female'.")

        sex_numeric = 0 if sex == 'male' else 1
        metadata = [age, sex_numeric]
        metadata_tensor = torch.tensor(metadata).float().unsqueeze(0)  # Forma: (1, 2)

        # Realizar la predicción sin cálculo de gradiente
        with torch.no_grad():
            output = model(image, metadata_tensor)
            
            if output is not None:
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                predicted_probabilities = probabilities.squeeze().tolist()
            else:
                print("Error: output es None.")
                return None, None  # Retorna None en caso de error

        return predicted_class, predicted_probabilities

    except Exception as e:
        print(f'Error en la predicción: {str(e)}')
        raise


# Función para verificar el estado del modelo
def check_model_state():
    state = model.state_dict()
    for layer, params in state.items():
        print(f"Layer: {layer}, Weights: {params}")

predictions_store = {}

@app.route('/result', methods=['GET'])
def result_endpoint():
    try:
        prediction_id = request.args.get('id')
        if prediction_id in predictions_store:
            return jsonify({'prediction': predictions_store[prediction_id]}), 200
        else:
            return jsonify({'error': 'Prediction not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.after_request
def disable_cache(response):
    response.headers['Cache-Control'] = 'no-store'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/predict', methods=['POST'])
@limiter.limit("15 per minute")  # Límite específico para este endpoint
def predict_endpoint():
    try:
        # Verificación de estado del modelo antes de la predicción
        print("Verificando estado del modelo antes de la predicción:")
        check_model_state()

        # Obtener los metadatos de edad y sexo
        age = request.form.get('edad')
        sex = request.form.get('sexo')

        # Verificar si los metadatos están presentes
        if not age or not sex:
            return jsonify({'error': 'Age and Sex are required'}), 400

        # Verificar si hay una imagen en la solicitud
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        imagen = request.files['image']

        # Verificar si el archivo tiene un nombre seguro y es de un tipo permitido
        if imagen.filename == '' or not allowed_file(imagen.filename):
            return jsonify({'error': 'Invalid image format or no image sent'}), 400

        # Guardar la imagen de manera segura en el servidor
        filename = secure_filename(imagen.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        imagen.save(image_path)

        # Convertir la edad a número entero
        age = int(age)

        # Llamar a la función de predicción
        predicted_class, predicted_probabilities = predict_with_metadata(image_path, age, sex)

        # Almacenar el resultado en el diccionario
        prediction_id = str(len(predictions_store))
        predictions_store[prediction_id] = {
            'class': predicted_class,
            'probabilities': predicted_probabilities
        }

        # Verificación del estado del modelo después de la predicción
        print("Verificando estado del modelo después de la predicción:")
        check_model_state()

        # Responder con la clase predicha y las probabilidades
        return jsonify({
            'message': 'Prediction successful',
            'predicted_class': predicted_class,
            'predicted_probabilities': predicted_probabilities
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'API is running'})

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Ruta no encontrada'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Error interno del servidor'}), 500

if __name__ == '__main__':
    app.run(debug=True) 