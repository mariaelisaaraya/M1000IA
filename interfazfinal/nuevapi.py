from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from io import BytesIO
import uuid
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
model.eval()

# Función de predicción con imagen y metadatos
def predict_with_metadata(image_stream, age, sex):
    try:
        # Transformar la imagen de PIL a un tensor
        image_tensor = load_and_preprocess_image(image_stream).unsqueeze(0)  # (1, 3, 128, 128)
        sex_numeric = 0 if sex == 'male' else 1
        metadata = [age, sex_numeric]
        metadata_tensor = torch.tensor(metadata).float().unsqueeze(0)  # (1, 2)

        # Realizar la predicción con el modelo
        with torch.no_grad():
            output = model(image_tensor, metadata_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            predicted_probabilities = probabilities.squeeze().tolist()

        return predicted_class, predicted_probabilities

    except Exception as e:
        print(f'Error en la predicción: {str(e)}')
        return None, None

predictions_store = {}


@app.route('/api/welcome', methods=['GET'])
def welcome():
    return jsonify({"message": "¡Bienvenido a la API!"}), 200


@app.route('/result', methods=['GET'])
def result_endpoint():
    prediction_id = request.args.get('id')
    if prediction_id in predictions_store:
        return jsonify({'prediction': predictions_store[prediction_id]}), 200
    else:
        return jsonify({'error': 'Prediction not found'}), 404

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

@app.route('/predict', methods=['POST'])
@limiter.limit("15 per minute")
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

        # Aquí hacemos uso de BytesIO correctamente
        image_stream = imagen.stream

        # Validar edad
        try:
            age = int(age)
            if age < 0 or age > 120:
                return jsonify({'error': 'Age must be between 0 and 120'}), 400
        except ValueError:
            return jsonify({'error': 'Age must be an integer'}), 400

        predicted_class, predicted_probabilities = predict_with_metadata(image_stream, age, sex)

        prediction_id = str(uuid.uuid4())
        predictions_store[prediction_id] = {
            'class': predicted_class,
            'probabilities': predicted_probabilities
        }

        return jsonify({
        'message': 'Prediction successful',
        'predicted_class': predicted_class,
        'predicted_probabilities': predicted_probabilities,
        'id': prediction_id,  # Agrega una coma aquí
      # Asegúrate de que image_data contenga los datos de la imagen
        }), 200

    except Exception as e:
        logging.error(f'Error in prediction: {str(e)}')
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/download/<string:prediction_id>', methods=['GET'])
def download_image(prediction_id):
    if prediction_id in predictions_store:
        # Obtén el contenido de la imagen desde predictions_store
        image_data = predictions_store[prediction_id]['image_stream']
        print(f"Enviando imagen para prediction_id: {prediction_id}")
        if image_data:
            # Crea un flujo de BytesIO desde el contenido
            return send_file(BytesIO(image_data), attachment_filename='downloaded_image.png', as_attachment=True)
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