from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model once at startup
try:
    print("🟢 Loading mask_detector.h5 model...")
    model = load_model("mask_detector.h5")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

# Labels
labels = ['Mask', 'No Mask']

@app.route('/')
def home():
    return jsonify({"message": "Face Mask Detection API is running successfully!"})

@app.route('/predict', methods=['POST'])
def predict():
    print("\n📩 /predict endpoint hit")
    try:
        data = request.get_json()
        print("🟢 Received data keys:", list(data.keys()) if data else None)

        if not data or 'image' not in data:
            print("⚠️ No 'image' key found in request JSON")
            return jsonify({"error": "No image data provided"}), 400

        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        print("🔹 Image data length:", len(image_data))

        # Decode base64 to image bytes
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            print("❌ Base64 decode failed:", e)
            return jsonify({"error": "Invalid base64 image"}), 400

        # Convert bytes to image
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            print("❌ PIL failed to open image:", e)
            return jsonify({"error": "Invalid image format"}), 400

        image = image.resize((224, 224))
        print("🟢 Image resized successfully")

        # Prepare image for model
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0) / 255.0

        if model is None:
            print("❌ Model not loaded, aborting prediction")
            return jsonify({"error": "Model not loaded"}), 500

        # Prediction
        print("🔹 Predicting...")
        pred = model.predict(image)
        print("🔹 Raw prediction output:", pred)

        label = labels[np.argmax(pred)]
        confidence = float(np.max(pred)) * 100

        print(f"✅ Prediction complete — Label: {label}, Confidence: {confidence:.2f}%")

        return jsonify({
            "label": label,
            "confidence": f"{confidence:.2f}%"
        })

    except Exception as e:
        print("❌ Error in prediction:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # host='0.0.0.0' so it's accessible externally (e.g., on Render)
    app.run(host='0.0.0.0', port=5000, debug=True)
