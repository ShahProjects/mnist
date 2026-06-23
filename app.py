from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

model = load_model("mnist_ann.h5")

@app.get("/")
def home():
    return {"message": "MNIST API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = Image.open(io.BytesIO(await file.read())).convert("L")

    image = image.resize((28, 28))

    img = np.array(image)

    img = img.astype("float32") / 255

    img = img.reshape(1, 784)

    prediction = model.predict(img)

    digit = int(np.argmax(prediction))

    return {"prediction": digit}