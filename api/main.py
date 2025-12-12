import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import io

# โหลดโมเดลที่เพิ่งเทรน
model = tf.keras.models.load_model("models/fashion_mnist.keras")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

app = FastAPI(title="Fashion MNIST API", version="1.0")

@app.get("/")
async def root():
    return {"message": "Fashion MNIST API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("L").resize((28, 28))
    img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0
    pred = model.predict(img_array, verbose=0)[0]
    return {
        "prediction": class_names[np.argmax(pred)],
        "confidence": float(np.max(pred))
    }
