"""
Pydantic schemas for the Heart Disease Prediction API.
Uses Field aliases to accept JSON keys with spaces.
"""
from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    """
    Request schema for heart disease prediction.
    Field aliases allow JSON keys with spaces to match the dataset column names.
    """
    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex: 0=female, 1=male")
    chest_pain_type: int = Field(
        ..., ge=1, le=4, alias="chest pain type",
        description="Chest pain type: 1=typical angina, 2=atypical angina, 3=non-anginal, 4=asymptomatic"
    )
    resting_bp_s: int = Field(
        ..., gt=0, alias="resting bp s",
        description="Resting blood pressure (mm Hg)"
    )
    cholesterol: int = Field(
        ..., gt=0, description="Serum cholesterol (mg/dl)"
    )
    fasting_blood_sugar: int = Field(
        ..., ge=0, le=1, alias="fasting blood sugar",
        description="Fasting blood sugar > 120 mg/dl: 0=false, 1=true"
    )
    resting_ecg: int = Field(
        ..., ge=0, le=2, alias="resting ecg",
        description="Resting ECG: 0=normal, 1=ST-T abnormality, 2=LVH"
    )
    max_heart_rate: int = Field(
        ..., gt=0, alias="max heart rate",
        description="Maximum heart rate achieved"
    )
    exercise_angina: int = Field(
        ..., ge=0, le=1, alias="exercise angina",
        description="Exercise-induced angina: 0=no, 1=yes"
    )
    oldpeak: float = Field(
        ..., description="ST depression induced by exercise relative to rest (can be negative)"
    )
    st_slope: int = Field(
        ..., ge=0, le=3, alias="ST slope",
        description="ST slope: 0=unknown, 1=upsloping, 2=flat, 3=downsloping"
    )
    
    class Config:
        populate_by_name = True  # Allow both alias and field name
        json_schema_extra = {
            "example": {
                "age": 55,
                "sex": 1,
                "chest pain type": 2,
                "resting bp s": 130,
                "cholesterol": 250,
                "fasting blood sugar": 0,
                "resting ecg": 0,
                "max heart rate": 150,
                "exercise angina": 0,
                "oldpeak": 1.5,
                "ST slope": 2
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for heart disease prediction."""
    prediction: int = Field(..., description="Predicted class: 0=Normal, 1=Heart Disease")
    probability: float = Field(..., description="Probability of heart disease (0-1)")
    threshold: float = Field(..., description="Classification threshold used")


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: str = Field(..., description="Service status")
