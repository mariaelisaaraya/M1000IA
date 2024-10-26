from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from prediction_tf import predict_with_metadata

app = Flask(__name__)

# Extensiones permitidas
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

# Ruta para cargar las imágenes
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Asegurarse de que la carpeta de uploads existe
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Función para verificar si el archivo tiene una extensión permitida
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#TODO ver si aceptar otras extensiones y en ese caso convertir a jpg

# Función para verificar el estado del modelo
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'API is running'}), 200

# Endpoint de predicción
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        # Obtener los metadatos desde la solicitud
        age = request.form.get('edad')
        sex = request.form.get('sexo')

        if not age or not sex:
            return jsonify({'error': 'Age and Sex are required'}), 400

        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        # Procesar y guardar la imagen
        image = request.files['image']

        # Verificar si la imagen tiene un formato permitido
        if image.filename == '' or not allowed_file(image.filename):
            return jsonify({'error': 'Invalid image format or no image sent'}), 400

        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)

        # Convertir la edad a entero
        age = int(age)

        # Hacer la predicción
        predicted_class, predicted_probabilities = predict_with_metadata(image_path, age, sex)

        if predicted_class is None or predicted_probabilities is None:
            return jsonify({'error': 'Prediction failed'}), 500

        # Retornar la predicción
        return jsonify({
            'message': 'Prediction successful',
            'predicted_class': int(predicted_class),  # Asegurar que sea tipo int de Python
            'predicted_probabilities': [float(prob) for prob in predicted_probabilities]  # Convertir a float
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
