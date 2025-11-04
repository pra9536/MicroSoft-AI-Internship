from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

app = Flask(__name__)

# ✅ Allow all origins and all routes for CORS
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

try:
    model = load_model("mask_detector.h5")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)

labels = ['Mask', 'No Mask']

@app.route('/')
def home():
    return jsonify({"message": "Face Mask Detection API is running successfully!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400

        image_data = data['image']
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)

        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((224, 224))

        image = img_to_array(image)
        image = np.expand_dims(image, axis=0) / 255.0

        pred = model.predict(image)
        label = labels[np.argmax(pred)]
        confidence = float(np.max(pred)) * 100

        return jsonify({
            "label": label,
            "confidence": f"{confidence:.2f}%"
        })

    except Exception as e:
        print("Error in prediction:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
