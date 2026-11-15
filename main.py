from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import json

# 1. Initialize the FastAPI app
app = FastAPI(
    title="Hospital Occupancy Early-Warning API",
    description="Predicts the change (delta) in hospital specialty occupancy for the next quarter.",
    version="1.0.0"
)

# 2. Define the input data schema using Pydantic
class HospitalRecord(BaseModel):
    loc_code: str = Field(..., example="A111H")
    spec_name: str = Field(..., example="Accident & Emergency")
    quarter_number: int = Field(..., example=4)
    aasb: int = Field(..., example=42)
    tobd: int = Field(..., example=42)
    asb: float = Field(..., example=0.45)
    aob: float = Field(..., example=0.45)
    p_occ: float = Field(..., example=100.0)
    eps: float = Field(..., example=20.0)
    los: float = Field(..., example=49.0)
    avlos: float = Field(..., example=2.45)
    emergency_activity_missing: int = Field(..., example=0)
    previous_p_occ: float = Field(None, example=98.5)
    previous_asb: float = Field(None, example=0.45)
    previous_eps: float = Field(None, example=18.0)

# Global variables to hold the model and metadata
pipeline = None
metadata = None

# 3. Load the model and metadata when the API starts
@app.on_event("startup")
def load_artifacts():
    global pipeline, metadata
    try:
        # This path assumes main.py is inside the 'api' folder and artifacts are in '../artifacts'
        pipeline = joblib.load("../artifacts/hospital_occupancy_pipeline.joblib")
        with open("../artifacts/model_metadata.json", "r") as f:
            metadata = json.load(f)
        print("Model and metadata loaded successfully!")
    except Exception as e:
        print(f"Error loading artifacts: {e}")

# 4. Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Hospital Operations Intelligence API"}

# 5. Predict endpoint
@app.post("/predict")
def predict_occupancy(record: HospitalRecord):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model pipeline is not loaded.")
    
    try:
        # Convert input to DataFrame (using model_dump for newer Pydantic versions)
        input_dict = record.model_dump() if hasattr(record, 'model_dump') else record.dict()
        input_data = pd.DataFrame([input_dict])
        
        # Ensure correct column order as expected by the pipeline
        feature_cols = metadata["feature_columns"]
        input_data = input_data[feature_cols]
        
        # Predict the delta
        pred_delta = pipeline.predict(input_data)[0]
        
        # Calculate final predicted occupancy
        current_occ = record.p_occ
        final_prediction = min(max(current_occ + pred_delta, 0), 100) # Clip between 0 and 100
        
        return {
            "current_occupancy": current_occ,
            "predicted_delta": round(pred_delta, 3),
            "predicted_next_occupancy": round(final_prediction, 3)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))