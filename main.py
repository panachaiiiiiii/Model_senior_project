from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import keras
import numpy as np
from segment import segment_image
from PIL import Image
from io import BytesIO
import tensorflow as tf
import pickle
import os
import gdown

# =========================
# DOWNLOAD FOLDER (ครั้งเดียว)
# =========================
FOLDER_URL = "https://drive.google.com/drive/folders/1mNHHtEHOrYBiZ77POK-N4UjgotMvHTmb"

def download_models():
    if not os.path.exists("./model"):
        print("Downloading models from Google Drive...")
        gdown.download_folder(
            url=FOLDER_URL,
            output="./model",
            quiet=False
        )
    else:
        print("Models already exist, skip download.")

download_models()

# =========================
# CONFIG
# =========================
MODEL_CONFIG = {
    "Densenet121": {
        "path": "./model/DenseNet121.keras",
        "class_names": ["โรคผิวหนังระยะก่อนเป็นมะเร็ง", "กลุ่มโรคมะเร็งผิวหนัง", "เนื้องอกผิวหนังชนิดไม่ร้ายแรง"],
        "img_size": (224, 224)
    },
    "MobileNetV2": {
        "path": "./model/MobileNetV2.keras",
        "class_names": ["โรคผิวหนังระยะก่อนเป็นมะเร็ง", "กลุ่มโรคมะเร็งผิวหนัง", "เนื้องอกผิวหนังชนิดไม่ร้ายแรง"],
        "img_size": (224, 224)
    },
    "ResNet50": {
        "path": "./model/ResNet50.keras",
        "class_names": ["โรคผิวหนังระยะก่อนเป็นมะเร็ง", "กลุ่มโรคมะเร็งผิวหนัง", "เนื้องอกผิวหนังชนิดไม่ร้ายแรง"],
        "img_size": (224, 224)
    },
    "Densenet121_segmented": {
        "path": "./model/Segmented_DenseNet121.keras",
        "class_names": ["โรคผิวหนังระยะก่อนเป็นมะเร็ง", "กลุ่มโรคมะเร็งผิวหนัง", "เนื้องอกผิวหนังชนิดไม่ร้ายแรง"],
        "img_size": (224, 224)
    },
}

# =========================
# PATCH Dense
# =========================
from keras.layers import Dense

_original_init = Dense.__init__
def _patched_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    return _original_init(self, *args, **kwargs)

Dense.__init__ = _patched_init

# =========================
# LOSS
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
# LOAD MODELS
# =========================
models = {}

# hybrid keras
hybrid_base = tf.keras.models.load_model(
    "./model/hybrid/Hybrid_DenseNet121.keras",
    compile=False
)

feature_extractor = tf.keras.Model(
    inputs=hybrid_base.input,
    outputs=hybrid_base.get_layer("dense").output
)

# pkl
with open("./model/hybrid/Hybrid_PAD_DenseNet121_xgb.pkl", "rb") as f:
    pkl_model = pickle.load(f)

# normal models
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
def preprocess(image: Image.Image, size):
    image = image.resize(size)
    image = image.convert("RGB")
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# =========================
# ENSEMBLE
# =========================
def predict_ensemble_from_image(img: Image.Image):
    selected_models = ["Densenet121", "MobileNetV2", "ResNet50"]

    probs_list = []
    for m in selected_models:
        x = preprocess(img, MODEL_CONFIG[m]["img_size"])
        pred = models[m].predict(x, verbose=0)[0]
        probs_list.append(pred)

    ensemble_probs = np.mean(probs_list, axis=0)
    class_names = MODEL_CONFIG["Densenet121"]["class_names"]

    return {
        "model": "ensemble_model",
        "models_used": selected_models,
        "result": {
            class_names[i]: round(float(prob) * 100, 2)
            for i, prob in enumerate(ensemble_probs)
        }
    }

# =========================
# HYBRID
# =========================
def predict_hybrid(img: Image.Image):
    x = preprocess(img, (224, 224))
    features = feature_extractor.predict(x, verbose=0)
    probs = pkl_model.predict_proba(features)[0]

    class_names = MODEL_CONFIG["Densenet121"]["class_names"]

    return {
        "model": "hybrid",
        "result": {
            class_names[i]: round(float(prob) * 100, 2)
            for i, prob in enumerate(probs)
        }
    }

# =========================
# API
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...), model_name: str = Form(...)):
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")

    if model_name == "ensemble_model":
        return predict_ensemble_from_image(img)

    if model_name == "hybrid":
        return predict_hybrid(img)

    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not found")

    config = MODEL_CONFIG[model_name]

    if model_name == "Densenet121_segmented":
        img_array = segment_image(img, img_size=config["img_size"])
        x = np.expand_dims(img_array.astype(np.float32) / 255.0, axis=0)
    else:
        x = preprocess(img, config["img_size"])

    prediction = models[model_name].predict(x, verbose=0)

    return {
        "model": model_name,
        "result": {
            config["class_names"][i]: round(float(prob) * 100, 2)
            for i, prob in enumerate(prediction[0])
        }
    }

# =========================
# MODELS LIST
# =========================
@app.get("/models")
def get_models():
    model_list = [
        {
            "name": name,
            "class_names": config["class_names"],
            "img_size": config["img_size"]
        }
        for name, config in MODEL_CONFIG.items()
    ]

    model_list.append({"name": "ensemble_model", "img_size": (224, 224)})
    model_list.append({"name": "hybrid", "img_size": (224, 224)})

    return {"models": model_list}