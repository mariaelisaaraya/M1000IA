from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_cors import cross_origin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from io import BytesIO
import uuid
import base64
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# Inicializar Flask-Limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["60 per minute"]  # Límite global por minuto
)

# Tipos de archivo permitidos
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Función para verificar el formato de la imagen
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Normalización
normalize_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalización basada en ImageNet
])

# Cargar y normalizar la imagen
def load_and_preprocess_image(image_stream):
    image = Image.open(image_stream).convert('RGB')  # Asegúrate de convertir la imagen a RGB
    image = normalize_transform(image)
    return image

# Definir el modelo
class MultiInputModel(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super(MultiInputModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.5)

        self.fc1_input_size = 32768  # Cambia esto a 32768
        self.fc1 = nn.Linear(self.fc1_input_size, 128)
        self.fc_meta = nn.Linear(num_metadata_features, 64)  
        self.fc_output = nn.Linear(32832, num_classes)  # Ajustado para 32832



    def forward(self, img, meta):
        x = self.pool(F.relu(self.conv1(img)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
    
        x = x.view(x.size(0), -1)  # Aplanar el tensor
        print(x.shape)  # Agrega esta línea para ver el tamaño

        meta_out = F.relu(self.fc_meta(meta))  # Salida de los metadatos
        print(meta_out.shape)  # Verifica el tamaño de la salida de los metadatos


        # Concatenar
        combined = torch.cat((x, meta_out), dim=1)

        # Salida
        output = self.fc_output(combined)
        return output

# Inicializar el modelo
num_metadata_features = 2
num_classes = 2

# Inicializar un nuevo modelo (para pruebas)
model = MultiInputModel(num_metadata_features, num_classes)
torch.save(model, 'modelo_entrenadomok.pth')  # Cambia la ruta según sea necesario
model.eval()

# Función de predicción con imagen y metadatos
# Función de predicción con imagen y metadatos
def predict_with_metadata(image_stream, age, sex):
    # Verificar si el modelo está cargado
    if model is None:
        print('Modelo no cargado o entrenado')
        return None, None  # También podrías devolver un mensaje de error

    try:
        # Transforma la imagen y el metadato para la predicción
        image_tensor = load_and_preprocess_image(image_stream).unsqueeze(0)  # (1, 3, 128, 128)
        sex_numeric = 0 if sex == 'male' else 1
        metadata = [age, sex_numeric]
        metadata_tensor = torch.tensor(metadata).float().unsqueeze(0)  # (1, 2)

        # Predicción
        with torch.no_grad():
            output = model(image_tensor, metadata_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            predicted_probabilities = probabilities.squeeze().tolist()

        return predicted_class, predicted_probabilities

    except Exception as e:
        print(f'Error en la predicción: {str(e)}')
        return None, None

predictions_store = {} #'1': b'some bytes data'.decode('utf-8'),  # Decode bytes to string

# Ruta para verificar si el modelo está cargado
@app.route('/status', methods=['GET'])
def model_status():
    if model is None:
        return jsonify({'status': 'error', 'message': 'Modelo no cargado o entrenado'}), 500
    else:
        return jsonify({'status': 'ok', 'message': 'Modelo cargado correctamente'}), 200

@app.route('/api/welcome', methods=['GET'])
def welcome():
    return jsonify({"message": "¡Bienvenido a la API!"}), 200


@app.route('/result/<prediction_id>', methods=['GET'])
def result_endpoint(prediction_id):
    if prediction_id in predictions_store:
        prediction_value = predictions_store[prediction_id]
        
        # Verificar si el valor es de tipo bytes
        if isinstance(prediction_value, bytes):
            # Convertir a base64
            prediction_value = base64.b64encode(prediction_value).decode('utf-8')
        
        return jsonify({'prediction': prediction_value}), 200
    else:
        return jsonify({'error': 'Prediction not found'}), 404
    
print(predictions_store)  # Asegúrate de que contiene el ID correcto y la imagen



import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        age = request.form.get('edad')
        sex = request.form.get('sexo')

        if not age or not sex:
            return jsonify({'error': 'Age and Sex are required'}), 400

        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        imagen = request.files['image']

        if imagen.filename == '' or not allowed_file(imagen.filename):
            return jsonify({'error': 'Invalid image format or no image sent'}), 400

        # Lee el flujo de imagen
        image_stream = BytesIO(imagen.read())

        try:
            age = int(age)
            if age < 0 or age > 120:
                return jsonify({'error': 'Age must be between 0 and 120'}), 400
        except ValueError:
            return jsonify({'error': 'Age must be an integer'}), 400

        predicted_class, predicted_probabilities = predict_with_metadata(image_stream, age, sex)
        # Imprimir la clase predicha en la terminal
        print(f'Predicted Class: {predicted_class}, Age: {age}, Sex: {sex}')

        prediction_id = str(uuid.uuid4())
        predictions_store[prediction_id] = {
            'class': predicted_class,
            'probabilities': predicted_probabilities,
            'image_stream': image_stream.getvalue() , # guarda los datos en bytes
            'edad': age,
            'sexo': sex
        }

        return jsonify({
            'message': 'Prediction successful',
            'predicted_class': predicted_class,
            'predicted_probabilities': predicted_probabilities,
            'id': prediction_id,
            'edad': age,
            'sexo': sex
        }), 200

    except Exception as e:
        logging.error(f'Error in prediction: {str(e)}')
        return jsonify({'error': 'Internal server error'}), 500
    
@app.route('/predict/<prediction_id>', methods=['GET'])
def get_prediction(prediction_id):
    # Busca el prediction_id en tu almacenamiento de predicciones
    prediction = predictions_store.get(prediction_id)
    if prediction:
        return jsonify({
            'class': prediction['class'],
            'probabilities': prediction['probabilities'],
            'edad': prediction['edad'],
            'sexo': prediction['sexo']
        }), 200
    else:
        return jsonify({'error': 'Predicción no encontrada'}), 404


@app.route('/download/<string:prediction_id>', methods=['GET'])
@cross_origin()
def download_image(prediction_id):
    if prediction_id in predictions_store:
        image_data = predictions_store[prediction_id].get('image_stream')
        if image_data:
            return send_file(BytesIO(image_data), download_name='downloaded_image.png', as_attachment=True)

        else:
            return jsonify({'error': 'Image not found'}), 404
    else:
        return jsonify({'error': 'Prediction not found'}), 404


print(predictions_store.keys())  # Muestra todas las claves disponibles

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