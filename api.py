import logging
import re
from enum import Enum
from typing import Optional

import predict_churn as prediction
from fastapi import FastAPI, HTTPException
from pydantic import ConfigDict, Field, FiniteFloat, StrictInt, create_model

logger = logging.getLogger(__name__)

NUMERIC_INPUT_FIELDS = (
    "age",
    "index_linking_percentage",
    "policy_term_months",
    "tenure_months",
    "annualized_premium",
    "benefits_count",
    "avg_benefit_loading_pct",
    "total_sum_insured",
    "premiums_count",
    "paid_ratio",
    "sum_payment_inconsistency",
    "total_paid_amount",
    "total_due_amount",
    "avg_due_days",
    "due_count",
    "alterations_count",
    "risk_alterations_count",
    "retention_alterations_count",
    "fund_switch_count",
    "days_since_last_switch",
    "total_withdrawal_amount",
)
BINARY_INPUT_FIELDS = (
    "is_pep",
    "is_sensitive",
    "is_unit_linked",
    "defensive_switch_flag",
)
DERIVED_FIELDS = (
    "unpaid_debt_ratio",
    "grace_period_exhaustion",
    "payment_instability_score",
    "overdue_ratio",
    "withdrawal_to_premium_ratio",
    "tenure_completion_pct",
    "is_under_2yr_uls",
    "defensive_switch_rate",
    "alteration_retention_ratio",
)

app = FastAPI(
    title="Churn Prediction API",
    description=(
        "Predict policy status and return SHAP-based explanations from the "
        "trained LightGBM model."
    ),
    version="2.0",
)

loaded_model = getattr(prediction, "final_model", None)
feature_columns_ordered = getattr(prediction, "feature_columns_order", [])
mappings = getattr(prediction, "mappings", None)
explainer = getattr(prediction, "explainer", None)

categorical_choices = {}
if isinstance(mappings, dict):
    categorical_choices.update(mappings.get("one_hot_mappings", {}))
    if "job_mapping" in mappings:
        categorical_choices["job"] = list(mappings["job_mapping"])

request_fields = {
    name: (FiniteFloat, Field(..., description="Finite numeric model input."))
    for name in NUMERIC_INPUT_FIELDS
}
request_fields.update({
    name: (
        StrictInt,
        Field(..., ge=0, le=1, description="Enter 0 or 1."),
    )
    for name in BINARY_INPUT_FIELDS
})
for field_name, choices in categorical_choices.items():
    enum_name = re.sub(r"\W|^(?=\d)", "_", field_name) + "Options"
    choice_enum = Enum(
        enum_name,
        {f"option_{index}": value for index, value in enumerate(choices)},
        type=str,
    )
    request_fields[field_name] = (
        choice_enum,
        Field(..., description="Choose one of the listed training categories."),
    )
request_fields["policy_id"] = (
    Optional[str],
    Field(default=None, description="Optional identifier included in the explanation."),
)

ChurnPredictionRequest = create_model(
    "ChurnPredictionRequest",
    __config__=ConfigDict(extra="forbid", use_enum_values=True),
    **request_fields,
)


def require_artifacts():
    if (
        loaded_model is None
        or not feature_columns_ordered
        or not isinstance(mappings, dict)
        or explainer is None
    ):
        raise HTTPException(
            status_code=503,
            detail="Prediction artifacts are unavailable. Check the model artifact files.",
        )

@app.get("/")
def home():
  return {
      "message": (
          "Welcome to the Churn Prediction API! Send a POST request to"
          " /predict"
      )
  }


@app.get("/features")
def get_features():
    """Return the raw input fields, valid categories, and server-derived features."""
    require_artifacts()

    fields = []
    for name in NUMERIC_INPUT_FIELDS:
        fields.append({
            "name": name,
            "type": "number",
            "required": True,
            "allowed_values": None,
        })
    for name in BINARY_INPUT_FIELDS:
        fields.append({
            "name": name,
            "type": "integer",
            "required": True,
            "allowed_values": [0, 1],
        })
    for name, choices in categorical_choices.items():
        fields.append({
            "name": name,
            "type": "string",
            "required": True,
            "allowed_values": list(choices),
        })
    fields.append({
        "name": "policy_id",
        "type": "string",
        "required": False,
        "allowed_values": None,
    })

    return {
        "fields": fields,
        "model_features": list(feature_columns_ordered),
        "derived_features": list(DERIVED_FIELDS),
        "predicted_statuses": list(mappings["final_status_mapping"].values()),
    }


@app.post("/predict")
def predict_churn(user_data: ChurnPredictionRequest):
    """Predict a policy status and return its explanation and recommended action."""
    require_artifacts()
    try:
        result = prediction.full_prediction_pipeline(
            user_data.model_dump(),
            loaded_model,
            feature_columns_ordered,
            mappings,
            explainer,
        )
        result["prediction"].pop("_processed_df", None)
        return result
    except HTTPException:
        raise
    except Exception:
        logger.exception("Churn prediction pipeline failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed while processing the supplied features.",
        )

if __name__ == "__main__":
    import uvicorn
    from config import RUNNING_MODE
    if RUNNING_MODE == "Development":
        uvicorn.run("api:app", host="127.0.0.1", port=8000, log_level="info", reload=True)
    elif RUNNING_MODE == "Production":
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")