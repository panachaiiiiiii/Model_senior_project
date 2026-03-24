from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import keras
import numpy as np
from PIL import Image

# =========================
# CLASS NAMES (แต่ละ model แยกได้)
# =========================
MODEL_CONFIG = {
    "Densenet121": {
        "path": "DenseNet121_Model.keras",
        "class_names": ["Nevus", "Cancer", "Benign", "Precancer"],
        "img_size": (224, 224)
    },
    "MobileNetV2": {
        "path": "MobileNetV2_Model.keras",
        "class_names": ["Nevus", "Cancer", "Benign", "Precancer"],
        "img_size": (224, 224)
    }
}

# =========================
# PATCH Dense (fix quantization_config)
# =========================
from keras.layers import Dense

_original_init = Dense.__init__
def _patched_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    return _original_init(self, *args, **kwargs)

Dense.__init__ = _patched_init

# =========================
# Dummy loss
# =========================
def loss_fn(y_true, y_pred):
    return keras.losses.categorical_crossentropy(y_true, y_pred)

# =========================
# APP
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 🔥 LOAD ALL MODELS
# =========================
models = {}

for name, config in MODEL_CONFIG.items():
    models[name] = keras.models.load_model(
        config["path"],
        compile=False,
        safe_mode=False,
        custom_objects={"loss_fn": loss_fn}
    )

print(f"Loaded models: {list(models.keys())}")

# =========================
# PREPROCESS
# =========================
def preprocess(image: Image.Image, size) -> np.ndarray:
    image = image.resize(size)
    image = image.convert("RGB")
    img_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# =========================
# API
# =========================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form(...)
):
    # 🔹 check model exists
    if model_name not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not found"
        )

    model = models[model_name]
    config = MODEL_CONFIG[model_name]

    # 🔹 preprocess
    img = Image.open(file.file).convert("RGB")
    x = preprocess(img, config["img_size"])

    # 🔹 predict
    prediction = model.predict(x, verbose=0)

    return {
        "model": model_name,
        "result": {
            config["class_names"][i]: round(float(prob) * 100, 2)
            for i, prob in enumerate(prediction[0])
        }
    }
    
    
@app.get("/models")
def get_models():
    return {
        "models": [
            {
                "name": name,
                "class_names": config["class_names"],
                "img_size": config["img_size"]
            }
            for name, config in MODEL_CONFIG.items()
        ]
    }