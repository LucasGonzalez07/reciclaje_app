# app.py — versión corregida y completa
import os
from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

# Cargar modelo
model = tf.keras.models.load_model("model/reciclaje_model.h5")
class_names = ["Organic", "Recyclable"]  # O=0, R=1

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No se subió ningún archivo'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = Image.open(filepath).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    os.remove(filepath)

    return jsonify({
        'class': predicted_class,
        'confidence': round(confidence * 100, 2)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)