from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Water Quality Calculator")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load the model and scaler
try:
    model = joblib.load('water_quality_model.pkl')
    scaler = joblib.load('scaler.pkl')
    logger.info("Model and scaler loaded successfully")
    logger.info(f"Model type: {type(model)}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

class WaterQualityInput(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

    class Config:
        from_attributes = True

def create_feature_interactions(data_df):
    """Create interaction features for the input data"""
    interactions = {
        'ph_hardness': data_df['ph'] * data_df['Hardness'],
        'sulfate_conductivity': data_df['Sulfate'] * data_df['Conductivity'],
        'organic_trihalomethanes': data_df['Organic_carbon'] * data_df['Trihalomethanes'],
        'solids_turbidity': data_df['Solids'] * data_df['Turbidity'],
        'ph_squared': data_df['ph'] ** 2,
        'Sulfate_squared': data_df['Sulfate'] ** 2,
        'Chloramines_squared': data_df['Chloramines'] ** 2
    }
    return pd.concat([data_df, pd.DataFrame([interactions])], axis=1)

def assess_water_safety(input_dict):
    """Assess basic water safety based on WHO guidelines"""
    safety_issues = []
    
    if input_dict['ph'] < 6.5 or input_dict['ph'] > 8.5:
        safety_issues.append("pH outside safe range (6.5-8.5)")
    if input_dict['Turbidity'] > 5:
        safety_issues.append("Turbidity too high (>5 NTU)")
    if input_dict['Chloramines'] > 4:
        safety_issues.append("Chloramines too high (>4 mg/L)")
    if input_dict['Sulfate'] > 250:
        safety_issues.append("Sulfate too high (>250 mg/L)")
    if input_dict['Solids'] > 500:
        safety_issues.append("Total dissolved solids too high (>500 mg/L)")
    
    return safety_issues

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(data: WaterQualityInput):
    try:
        # Convert input to DataFrame
        input_dict = data.model_dump()
        logger.info(f"Received input: {input_dict}")
        
        # Check basic water safety
        safety_issues = assess_water_safety(input_dict)
        logger.info(f"Safety issues found: {safety_issues}")
        
        features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                   'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
        
        # Validate input ranges
        if not (0 <= input_dict['ph'] <= 14):
            raise ValueError("pH must be between 0 and 14")
        if any(value < 0 for value in input_dict.values()):
            raise ValueError("All measurements must be non-negative")
            
        input_data = pd.DataFrame([input_dict], columns=features)
        logger.info(f"Input DataFrame shape: {input_data.shape}")
        
        # Create interaction features
        input_data = create_feature_interactions(input_data)
        logger.info(f"DataFrame shape after interactions: {input_data.shape}")
        
        # Scale features
        input_scaled = scaler.transform(input_data)
        logger.info(f"Scaled input shape: {input_scaled.shape}")
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        logger.info(f"Raw prediction: {prediction}")
        logger.info(f"Prediction probabilities: {prediction_proba}")
        
        # If there are safety issues, override prediction
        if safety_issues:
            prediction = 0  # Not potable
            prediction_proba = [0.9, 0.1]  # High confidence in not potable
            logger.info("Prediction overridden due to safety issues")
        
        # Prepare response
        response = {
            'prediction': 'Potable' if prediction == 1 else 'Not Potable',
            'confidence': float(prediction_proba[1] if prediction == 1 else prediction_proba[0]),
            'safety_issues': safety_issues
        }
        logger.info(f"Final response: {response}")
        return response
            
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # Process the image and analyze using the model
        pollution_percentage = WaterQualityAnalyzer.predict(contents)  # Adjust as needed
        return JSONResponse(content={"pollutionPercentage": pollution_percentage})
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise HTTPException(status_code=500, detail="Error analyzing image")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5008)
