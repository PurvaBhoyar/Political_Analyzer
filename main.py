from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
# Initialize prediction pipeline
predictor = pipeline("text-classification", model="./models/final_model")

class PromiseInput(BaseModel):
    text: str

@app.post("/predict")
async def predict_outcome(input_data: PromiseInput):
    prediction = predictor(input_data.text)[0]
    label_map = {"LABEL_0": "Unlikely", "LABEL_1": "Partial", "LABEL_2": "Highly Likely"}
    return {
        "promise": input_data.text,
        "forecast": label_map[prediction['label']],
        "confidence": round(prediction['score'], 4)
    }