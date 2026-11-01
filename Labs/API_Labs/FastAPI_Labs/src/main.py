from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data
from typing import Literal

app = FastAPI()

class WineData(BaseModel):
    """
    Pydantic BaseModel representing wine data.

    Attributes:
        alcohol (float): The alcohol content of the wine.
        malic_acid (float): The malic acid content of the wine.
        ash (float): The ash content of the wine.
        alcalinity_of_ash (float): The alcalinity of the ash in the wine.
        magnesium (float): The magnesium content of the wine.
        total_phenols (float): The total phenols content of the wine.
        flavanoids (float): The flavanoids content of the wine.
        nonflavanoid_phenols (float): The nonflavanoid phenols content of the wine.
        proanthocyanins (float): The proanthocyanins content of the wine.
        color_intensity (float): The color intensity of the wine.
        hue (float): The hue of the wine.
        od280_od315_of_diluted_wines (float): The OD280/OD315 ratio of the diluted wines.
        proline (float): The proline content of the wine.
    """
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

class WineResponse(BaseModel):
    response: Literal["Barolo", "Grignolino", "Barbera"]



"""Modern web apps use a technique named routing. This helps the user remember the URLs. 
For instance, instead of having /booking.php they see /booking/. Instead of /account.asp?id=1234/ 
they’d see /account/1234/."""

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    """Concurrent (multiple tasks can run simultaneously)"""
    return {"status": "healthy"}

@app.post("/predict", response_model=WineResponse)
async def predict_wine(wine_data: WineData):
    """
    Predict the wine quality based on provided features.
    This endpoint accepts wine data and returns the predicted quality.
    Args:
        wine_data (WineData): A WineData object containing:
            - alcohol (float): The alcohol content of the wine.
            - malic_acid (float): The malic acid content of the wine.
            - ash (float): The ash content of the wine.
            - alcalinity_of_ash (float): The alcalinity of the ash in the wine.
            - magnesium (float): The magnesium content of the wine.
            - total_phenols (float): The total phenols content of the wine.
            - flavanoids (float): The flavanoids content of the wine.
            - nonflavanoid_phenols (float): The nonflavanoid phenols content of the wine.
            - proanthocyanins (float): The proanthocyanins content of the wine.
            - color_intensity (float): The color intensity of the wine.
            - hue (float): The hue of the wine.
            - od280_od315_of_diluted_wines (float): The OD280/OD315 ratio of the diluted wines.
            - proline (float): The proline content of the wine.
    Returns:
        WineResponse: A response object containing:
            - response (str): The predicted wine quality.
    Raises:
        HTTPException: Returns a 500 status code with error details if prediction fails.
    Example:
        POST /predict
        {
            "alcohol": 12.0,
            "malic_acid": 1.0,
            "ash": 2.0,
            "alcalinity_of_ash": 15.0,
            "magnesium": 70.0,
            "total_phenols": 2.6,
            "flavanoids": 1.3,
            "nonflavanoid_phenols": 0.4,
            "proanthocyanins": 1.5,
            "color_intensity": 4.5,
            "hue": 12.0,
            "od280_od315_of_diluted_wines": 3.5,
            "proline": 350.0
        }
        Response:
        {
            "response": "Barolo"
        }
    """
    try:
        prediction = predict_data(wine_data)
        return WineResponse(response=prediction)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


    
