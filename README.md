# Churn Prediction API

This project serves a trained multi-class LightGBM model for insurance policy
status prediction. The API accepts raw policy and customer features, applies
the inference preprocessing used by the model, predicts a status, and returns
a SHAP-based explanation with an operational follow-up suggestion.

## Prediction behavior

The model predicts one of four statuses:

| Code | Status |
| ---: | --- |
| 0 | Not Churned |
| 1 | Lapsed |
| 2 | Surrender |
| 3 | Cancelled |

The API returns the class with the highest probability, its confidence, and
probabilities for all four classes. This is a policy-status classification
task, not a binary churn/no-churn response.

For each request, `predict_churn.py` performs these inference steps:

1. Removes identifier and target/leakage columns if present.
2. Derives ratios and indicators such as unpaid debt ratio, overdue ratio,
	 tenure completion, and defensive switching rate.
3. Drops redundant raw columns that are not model inputs.
4. Maps `job` to its saved target-encoding value and one-hot encodes the
	 low-cardinality categorical fields.
5. Reindexes the processed row to the exact order in `feature_columns.pkl`,
	 filling any absent model columns with zero.
6. Predicts the status and computes SHAP values for the predicted class.

The SHAP result reports positive and negative contributors to the selected
class. The response keys `retention_drivers_stay` and
`churn_risk_drivers_churn` are the current implementation's labels for those
positive and negative contributors; interpret the SHAP sign relative to the
predicted class, not as a universal churn/retention direction.

## API

Run the API from the project directory so the relative artifact path
`./churn_model_artifacts` resolves correctly:

```powershell
..\.venv\Scripts\Activate.ps1
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

### `GET /features`

Returns the raw request fields, valid categorical values, derived feature
names, predicted status labels, and the exact ordered model columns from
`feature_columns.pkl`. Raw request fields are what clients submit; model
columns are the transformed inputs assembled internally by preprocessing.

### `POST /predict`

Accepts a JSON object containing all required numeric, binary, and categorical
raw inputs. `policy_id` is optional and is returned for traceability. Unknown
properties and invalid categorical values are rejected by request validation.
Binary fields must be integers `0` or `1`; numeric fields must be finite
numbers.

Required numeric fields:

`age`, `index_linking_percentage`, `policy_term_months`, `tenure_months`,
`annualized_premium`, `benefits_count`, `avg_benefit_loading_pct`,
`total_sum_insured`, `premiums_count`, `paid_ratio`,
`sum_payment_inconsistency`, `total_paid_amount`, `total_due_amount`,
`avg_due_days`, `due_count`, `alterations_count`, `risk_alterations_count`,
`retention_alterations_count`, `fund_switch_count`,
`days_since_last_switch`, `total_withdrawal_amount`.

Required binary fields, each accepting `0` or `1`:

`is_pep`, `is_sensitive`, `is_unit_linked`, `defensive_switch_flag`.

Required categorical fields and their saved category values:

| Field | Allowed values |
| --- | --- |
| `gender` | `Female`, `Male` |
| `product_family` | `Dreams`, `Wedding`, `Retirement`, `Single Premium`, `Graduation`, `Hajj`, `Non Standard` |
| `payment_method` | `Premium Notice`, `ACH`, `Account Transfer`, `Direct Debit`, `Credit Card` |
| `index_linking_type` | `Linked`, `No Indexation`, `Fixed` |
| `job` | Use `GET /features` for the complete list of 60 saved values. |

These categorical choices come from `mappings.pkl`. `feature_columns.pkl`
defines the model's expected transformed feature names and order. Neither file
is interchangeable with the raw request schema.

Example request (category choices must match the loaded mappings):

```json
{
	"policy_id": "POL-10001",
	"age": 42,
	"is_pep": 0,
	"is_sensitive": 0,
	"index_linking_percentage": 0,
	"policy_term_months": 240,
	"tenure_months": 36,
	"annualized_premium": 1200,
	"is_unit_linked": 1,
	"benefits_count": 2,
	"avg_benefit_loading_pct": 0,
	"total_sum_insured": 50000,
	"premiums_count": 36,
	"paid_ratio": 1,
	"sum_payment_inconsistency": 0,
	"total_paid_amount": 3600,
	"total_due_amount": 0,
	"avg_due_days": 0,
	"due_count": 0,
	"alterations_count": 0,
	"risk_alterations_count": 0,
	"retention_alterations_count": 0,
	"fund_switch_count": 0,
	"days_since_last_switch": 0,
	"defensive_switch_flag": 0,
	"total_withdrawal_amount": 0,
	"gender": "Female",
	"product_family": "Dreams",
	"payment_method": "Direct Debit",
	"index_linking_type": "Fixed",
	"job": "Teacher"
}
```

The response has two top-level objects:

- `prediction`: policy ID, predicted status and code, confidence, and
	per-status class probabilities.
- `explanation`: policy ID and predicted status, ranked positive/negative
	SHAP contributors, and `actionable_routing` guidance.

The processed pandas row is used internally for prediction and SHAP, and is
intentionally omitted from the JSON response.

## Model artifacts

The `churn_model_artifacts/` directory must contain:

- `lgbm_churn_model.pkl`: trained LightGBM classifier.
- `feature_columns.pkl`: ordered transformed model feature names.
- `mappings.pkl`: categorical encodings and final status-code mapping.
- `shap_explainer.pkl`: SHAP explainer used for per-request explanations.

These artifacts must be produced together from a compatible model-training
run. The API loads them when `predict_churn.py` is imported. If an artifact
cannot be loaded, prediction is unavailable and the API reports a service
error.

## Notes

- The API enforces category membership, but does not enforce business ranges
	for numeric inputs because training ranges are not included in the artifacts.
- `policy_id` is used for response traceability; it is not a predictive input.
- Model predictions and SHAP explanations are decision-support outputs and
	should be reviewed in context before operational action.
