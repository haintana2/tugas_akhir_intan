# Flask application with YOLO for leaf detection and LSTM for classification
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Load models
yolo_model = YOLO(r"C:\Users\INTAN\Downloads\bismillah\models\intanyolo.pt")  # YOLO model for leaf detection
classification_model = load_model("models/intanlstm.h5")                      # LSTM classification model
CLASSES = ['KentangEarlyBlight', 'KentangLateBlight', 'KentangSehat', 'TomatSehat', 'TomatSpiderMite']
CONFIDENCE_THRESHOLD = 0.5  # YOLO confidence threshold

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

def classify_leaf(image):
    """Classify a cropped leaf image using the LSTM model"""
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

        # YOLO detection
        results = yolo_model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, imgsz=320)
        
        # Process YOLO results
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confidences):
                if conf < CONFIDENCE_THRESHOLD:  # Skip low-confidence detections
                    continue

                x1, y1, x2, y2 = map(int, box)
                cropped = frame[y1:y2, x1:x2]  # Crop detected leaf
                
                if cropped.size > 0:  # Ensure valid crop
                    class_name, confidence = classify_leaf(cropped)
                    return jsonify({"class": class_name, "confidence": confidence})
        
        # No leaf detected
        return jsonify({"error": "No leaf detected"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
