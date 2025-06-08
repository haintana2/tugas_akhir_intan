# Flask application with YOLO for leaf detection and LSTM for classification
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Load LSTM classification model
classification_model = load_model(r"D:\5-5-2025\app\models\intanlstm_new.h5")
CLASSES = ['KentangEarlyBlight', 'KentangLateBlight', 'KentangSehat', 'TomatSehat', 'TomatSpiderMite', 'TomatWhiteFly']

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

def classify_leaf(image):
    """Classify an image using the LSTM model"""
    image = Image.fromarray(image).resize((224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    predictions = classification_model.predict(image.reshape(1, 1, 224, 224, 3))
    class_idx = np.argmax(predictions)
    return CLASSES[class_idx], float(predictions[0][class_idx])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Load and convert image
        image = Image.open(file).convert('RGB')
        frame = np.array(image)  # Convert to numpy array (RGB format)
        
        # Classify image
        class_name, confidence = classify_leaf(frame)
        return jsonify({"class": class_name, "confidence": confidence})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
