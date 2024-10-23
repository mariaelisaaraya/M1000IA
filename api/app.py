from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from model import model, check_model_state
from prediction import predict_with_metadata
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Define the upload folder and allowed file extensions
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Function to check if a file is an allowed type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Prediction results store
predictions_store = {}

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'API is running'}), 200

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        # Verify model state before prediction
        check_model_state()

        # Get metadata from request
        age = request.form.get('edad')
        sex = request.form.get('sexo')

        if not age or not sex:
            return jsonify({'error': 'Age and Sex are required'}), 400

        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        # Process and save the image
        image = request.files['image']
        if image.filename == '' or not allowed_file(image.filename):
            return jsonify({'error': 'Invalid image format or no image sent'}), 400

        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)

        age = int(age)
        predicted_class, predicted_probabilities = predict_with_metadata(image_path, age, sex)

        if predicted_class is None or predicted_probabilities is None:
            return jsonify({'error': 'Prediction failed'}), 500

        # Store prediction result
        prediction_id = str(len(predictions_store))
        predictions_store[prediction_id] = {
            'class': predicted_class,
            'probabilities': predicted_probabilities
        }

        return jsonify({
            'message': 'Prediction successful',
            'predicted_class': predicted_class,
            'predicted_probabilities': predicted_probabilities
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Ruta no encontrada'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Error interno del servidor'}), 500

if __name__ == '__main__':
    app.run(debug=True)
