# app.py
import os
from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

# --- Ruta segura al modelo (funciona en cualquier entorno) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "reciclaje_model.h5")

# Verificar que el modelo existe
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ El modelo no se encontró en: {MODEL_PATH}")

# Cargar modelo
model = tf.keras.models.load_model(MODEL_PATH)
class_names = ["Organic", "Recyclable"]  # O=0, R=1

# --- Configurar Flask ---
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Rutas ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validar archivo
        if 'file' not in request.files:
            return jsonify({'error': 'No se subió ningún archivo'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Archivo sin nombre'}), 400

        # Guardar temporalmente
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocesar imagen
        img = Image.open(filepath).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predecir (sin verbose para no saturar logs)
        predictions = model.predict(img_array, verbose=0)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        # Limpiar archivo temporal
        os.remove(filepath)

        # Devolver resultado
        return jsonify({
            'class': predicted_class,
            'confidence': round(confidence * 100, 2)
        })

    except Exception as e:
        # Este mensaje aparecerá en los logs de Render
        print(f"🚨 ERROR EN PREDICCIÓN: {str(e)}")
        return jsonify({'error': 'Error al procesar la imagen. Inténtalo de nuevo.'}), 500

# --- Iniciar app ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # debug=False para producción (Render)
    app.run(host='0.0.0.0', port=port, debug=False)