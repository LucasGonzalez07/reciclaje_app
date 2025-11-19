# train.py
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuración
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = "data"  # debe contener O/ y R/
MODEL_SAVE_PATH = "model/reciclaje_model.h5"

# Crear carpeta del modelo
os.makedirs("model", exist_ok=True)

# Verificar que las carpetas O y R existen
if not (os.path.exists(os.path.join(DATA_DIR, "O")) and os.path.exists(os.path.join(DATA_DIR, "R"))):
    raise FileNotFoundError("❌ La carpeta 'data' debe contener subcarpetas 'O' y 'R'.")

# Preparar generadores
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print("Mapeo de clases:", train_gen.class_indices)  # Debe ser {'O': 0, 'R': 1}

# Construir modelo
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(2, activation='softmax')  # 2 clases: O y R
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar
print("Iniciando entrenamiento...")
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    verbose=1
)

# Guardar
model.save(MODEL_SAVE_PATH)
print(f"\n✅ Modelo guardado en: {MODEL_SAVE_PATH}")
print(f"Clases: 0 = Organic, 1 = Recyclable")