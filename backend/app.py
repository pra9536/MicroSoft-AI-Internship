from flask import Flask, request, jsonify
from flask_cors import CORS
import base64, io
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

app = Flask(__name__)
CORS(app)

model = load_model("mask_detector.h5")
labels = ['Mask', 'No Mask']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image']
    image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0) / 255.0

    pred = model.predict(image)
    label = labels[np.argmax(pred)]
    return jsonify({"label": label})

if __name__ == "__main__":
    app.run(debug=True)
