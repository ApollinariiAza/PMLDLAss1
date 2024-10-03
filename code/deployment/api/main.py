from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the model
model = joblib.load('/app/models/wine_model.pkl')

# Initialize FastAPI app
app = FastAPI()

class WineData(BaseModel):
    features: list

@app.post("/predict/")
async def predict(data: WineData):
    # Convert features to numpy array
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
