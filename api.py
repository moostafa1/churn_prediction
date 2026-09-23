from typing import Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import ConfigDict, create_model
from predict_churn import load_model, get_feature_order, get_user_prediction

app = FastAPI(
    title="XGBoost Churn Predictor API",
    description=(
        "API endpoint to predict customer churn probability using a trained"
        " model."
    ),
    version="1.0",
)

loaded_model = load_model()
feature_columns_ordered = get_feature_order(loaded_model) if loaded_model else []
ChurnPredictionRequest = create_model(
    "ChurnPredictionRequest",
    __config__=ConfigDict(
        json_schema_extra={
            "description": "Fill in every feature before sending the request."
        }
    ),
    **{
        feature_name: (Optional[Any], None)
        for feature_name in feature_columns_ordered
    },
)

@app.get("/")
def home():
  return {
      "message": (
          "Welcome to the Churn Prediction API! Send a POST request to"
          " /predict"
      )
  }


@app.post("/predict")
def predict_churn(user_data: ChurnPredictionRequest):
    """Takes model features and returns the predicted churn probability and class."""
    if loaded_model is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded. Please check the model file path.",
        )

    try:
        missing_features = [
            name for name, value in user_data.model_dump().items() if value is None
        ]
        if missing_features:
            raise HTTPException(
                status_code=422,
                detail=f"Missing values for: {', '.join(missing_features)}",
            )

        churn_probability, churn_class = get_user_prediction(
            loaded_model, user_data.model_dump(), feature_columns_ordered
        )

        return {
            "churn_probability": round(float(churn_probability), 4),
            "churn_class": int(churn_class),
            "prediction_label": (
                "Will Churn" if churn_class == 1 else "Will Not Churn"
            ),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error processing prediction: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    from config import RUNNING_MODE
    if RUNNING_MODE == "Development":
        uvicorn.run("api:app", host="127.0.0.1", port=8000, log_level="info", reload=True)
    elif RUNNING_MODE == "Production":
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")