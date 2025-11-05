from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import numpy as np
import os # <-- os module import kiya gaya
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

app = Flask(__name__)

# ✅ Allow all origins and all routes for CORS
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# --- 🎯 Path Correction Logic Start ---
# 1. app.py ka directory path nikalte hain (backend/)
current_dir = os.path.dirname(os.path.abspath(__file__)) 
# 2. Model file ka poora (absolute) path banate hain
model_path = os.path.join(current_dir, "mask_detector.h5")

try:
    # 3. Dynamic path ka upyog karke model load karte hain
    model = load_model(model_path) 
    print("✅ Model loaded successfully from:", model_path)
except Exception as e:
    print("❌ Error loading model:", e)
    # Agar model load nahi ho pata hai, to 'model' variable ko None set karte hain
    model = None 
# --- 🎯 Path Correction Logic End ---

labels = ['Mask', 'No Mask']

@app.route('/')
def home():
    if model is None:
        return jsonify({"message": "Face Mask Detection API is running, but model failed to load!"}), 503
    return jsonify({"message": "Face Mask Detection API is running successfully!"})

@app.route('/predict', methods=['POST'])
def predict():
    # Model check
    if model is None:
        return jsonify({"error": "Model is not loaded. Cannot perform prediction."}), 503

    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400

        image_data = data['image']
        # Remove metadata (if present)
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)

        # Process image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((224, 224))

        # Prepare for prediction
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0) / 255.0

        # Prediction
        pred = model.predict(image, verbose=0) # verbose=0 added for clean logs
        label = labels[np.argmax(pred)]
        confidence = float(np.max(pred)) * 100

        return jsonify({
            "label": label,
            "confidence": f"{confidence:.2f}%"
        })

    except Exception as e:
        # Agar koi aur prediction error aati hai (jaise galat image format), to 500 return karega
        print("Error in prediction:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
