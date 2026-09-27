import numpy as np
import pandas as pd
import shap
import joblib
import pickle
import os

# Define the directory where artifacts are saved
ARTIFACTS_DIR = "./churn_model_artifacts"

# Load model artifacts dynamically
try:
    final_model = joblib.load(os.path.join(ARTIFACTS_DIR, "lgbm_churn_model.pkl"))
    feature_columns_order = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_columns.pkl"))
    mappings = joblib.load(os.path.join(ARTIFACTS_DIR, "mappings.pkl"))
    with open(os.path.join(ARTIFACTS_DIR, "shap_explainer.pkl"), 'rb') as f:
        explainer = pickle.load(f)
    print("All model artifacts loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading model artifacts: {e}. Please ensure '{ARTIFACTS_DIR}' exists and contains all required files.")
except Exception as e:
    print(f"An unexpected error occurred during artifact loading: {e}")


# ==============================================================================
# 1. HELPER: DYNAMIC FEATURE TRANSFORMATIONS & ENCODING
# ==============================================================================
def preprocess_user_payload(user_data_dict, feature_columns_order, mappings):
    """
    Computes synthesized domain ratios, applies categorical encodings,
    and enforces column order matching training alignment.
    """
    df = pd.DataFrame([user_data_dict])
    eps = 1e-6

    # Define lists of columns based on training pipeline
    leakage_and_id_cols = [
        'customer_id',
        'policy_id',
        'churned',
        'policy_status',
        'final_status'
    ]
    low_cardinality_cols = list(mappings['one_hot_mappings'].keys())
    redundant_features_to_drop = [
        'sum_due_days',
        'avg_due_days',
        'due_count',
        'paid_count'
    ]

    # --- Step 1: Initial Drop of known leakage/ID columns ---
    for col in leakage_and_id_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # --- Step 2: Synthesize Domain Features (replication of create_domain_features) ---
    if 'total_due_amount' in df.columns and 'total_paid_amount' in df.columns:
        df['unpaid_debt_ratio'] = df['total_due_amount'] / (df['total_paid_amount'] + eps)
    if 'avg_due_days' in df.columns:
        df['grace_period_exhaustion'] = df['avg_due_days'] / 60.0
    if 'sum_payment_inconsistency' in df.columns and 'premiums_count' in df.columns:
        df['payment_instability_score'] = df['sum_payment_inconsistency'] / (df['premiums_count'] + eps)
    if 'due_count' in df.columns and 'premiums_count' in df.columns:
        df['overdue_ratio'] = df['due_count'] / (df['premiums_count'] + eps)
    if 'total_withdrawal_amount' in df.columns and 'total_paid_amount' in df.columns:
        df['withdrawal_to_premium_ratio'] = df['total_withdrawal_amount'] / (df['total_paid_amount'] + eps)
    if 'tenure_months' in df.columns and 'policy_term_months' in df.columns:
        df['tenure_completion_pct'] = df['tenure_months'] / (df['policy_term_months'] + eps)
    if 'tenure_months' in df.columns and 'is_unit_linked' in df.columns:
        df['is_under_2yr_uls'] = ((df['tenure_months'] < 24) & (df['is_unit_linked'] == 1)).astype(int)
    if 'defensive_switch_flag' in df.columns and 'fund_switch_count' in df.columns:
        df['defensive_switch_rate'] = df['defensive_switch_flag'] / (df['fund_switch_count'] + eps)
    if 'retention_alterations_count' in df.columns and 'alterations_count' in df.columns:
        df['alteration_retention_ratio'] = df['retention_alterations_count'] / (df['alterations_count'] + eps)


    # --- Step 3: Drop Redundant Raw Features ---
    for col in redundant_features_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])

    # --- Step 4: Target Encode High-Cardinality Categorical ('job') ---
    if 'job' in df.columns and 'job_mapping' in mappings:
        job_val = df['job'].values[0]
        job_map = mappings['job_mapping']
        default_job_val = np.mean(list(job_map.values())) if job_map else 0.0
        df['job_encoded'] = job_map.get(job_val, default_job_val)
        df = df.drop(columns=['job'])

    # --- Step 5: One-Hot Encode Low-Cardinality Categoricals ---
    for col in low_cardinality_cols:
        if col in df.columns:
            temp_s = df[col].astype('category').cat.set_categories(mappings['one_hot_mappings'][col])
            dummies = pd.get_dummies(temp_s, prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    # Convert boolean columns created by get_dummies to int32
    df = df.astype({col: 'int32' for col in df.select_dtypes(include='bool').columns})

    # --- Step 6: Enforce Exact Feature Column Order & Fill Missing Columns with 0 ---
    df = df.reindex(columns=feature_columns_order, fill_value=0)

    return df

# ==============================================================================
# 2. PRIMARY PREDICTION FUNCTION
# ==============================================================================
def get_user_prediction(model, user_data_dict, feature_columns_order, mappings):
    """
    Takes a trained model and a dictionary of user input,
    and returns the predicted churn probability and class details.
    """
    X_processed = preprocess_user_payload(user_data_dict, feature_columns_order, mappings)

    # Predict probabilities across classes [0: Not Churned, 1: Lapsed, 2: Surrender, 3: Cancelled]
    probs = model.predict_proba(X_processed)[0]
    pred_class_code = int(np.argmax(probs))

    status_map = mappings['final_status_mapping']
    predicted_status = status_map.get(pred_class_code, "Unknown")

    return {
        "policy_id": user_data_dict.get("policy_id", "N/A"),
        "predicted_status": predicted_status,
        "status_code": pred_class_code,
        "confidence": float(probs[pred_class_code]),
        "class_probabilities": {
            status_map[i]: float(probs[i]) for i in range(len(probs))
        },
        "_processed_df": X_processed  # Passed internally to SHAP function
    }

# ==============================================================================
# 3. SHAP EXPLANATION & ACTION ROUTING FUNCTION
# ==============================================================================
def get_user_shap_factors(explainer, prediction_result, mappings, top_n=3):
    """
    Computes top retention (pull) and churn risk (push) SHAP factors for a user
    and returns actionable operational routing.
    """
    X_processed = prediction_result["_processed_df"]
    pred_code = prediction_result["status_code"]
    predicted_status_name = prediction_result["predicted_status"]
    policy_id = prediction_result["policy_id"]

    # Compute SHAP values
    shap_explanation = explainer(X_processed)

    if len(shap_explanation.values.shape) == 3:
        vals = shap_explanation.values[0, :, pred_code]
    else:
        vals = shap_explanation.values[0]

    feature_names = X_processed.columns
    raw_values = X_processed.iloc[0].values

    df_shap = pd.DataFrame({
        'Feature': feature_names,
        'Value': raw_values,
        'SHAP_Value': vals
    })

    df_shap = df_shap.sort_values(by='SHAP_Value', ascending=False).reset_index(drop=True)

    top_positive_drivers = df_shap[df_shap['SHAP_Value'] > 0].head(top_n)
    top_negative_drivers = df_shap[df_shap['SHAP_Value'] < 0].tail(top_n).sort_values(by='SHAP_Value', ascending=True)

    actionable_routing = {
        "prediction_insight": f"The policy for ID {policy_id} is predicted to be '{predicted_status_name}'.",
        "next_steps_suggestion": "",
        "flagged_for_review": False
    }

    if predicted_status_name == "Not Churned":
        actionable_routing["next_steps_suggestion"] = "No immediate action required, policy appears stable."
    elif predicted_status_name == "Lapsed":
        actionable_routing["next_steps_suggestion"] = "Engage with customer regarding overdue payments and policy reinstatement options. Highlight benefits that drive retention."
        actionable_routing["flagged_for_review"] = True
    elif predicted_status_name == "Surrender":
        actionable_routing["next_steps_suggestion"] = "Contact customer to understand reasons for surrender. Offer alternatives like premium reduction or policy alterations. Focus on high-value retention drivers."
        actionable_routing["flagged_for_review"] = True
    elif predicted_status_name == "Cancelled":
        actionable_routing["next_steps_suggestion"] = "Urgent review. Policy has been terminated without payout. Investigate for potential fraud or extreme customer dissatisfaction. Consider outreach for feedback."
        actionable_routing["flagged_for_review"] = True

    if 'total_withdrawal_amount' in df_shap['Feature'].values:
        withdrawal_shap_impact = df_shap[df_shap['Feature'] == 'total_withdrawal_amount']['SHAP_Value'].iloc[0]
        withdrawal_value = df_shap[df_shap['Feature'] == 'total_withdrawal_amount']['Value'].iloc[0]
        if withdrawal_value > 0 and withdrawal_shap_impact > 0.1:
             actionable_routing["next_steps_suggestion"] += " High withdrawal activity detected and contributing significantly to current status; assess financial distress."

    retention_drivers = []
    for _, row in top_positive_drivers.iterrows():
        retention_drivers.append({
            "feature": row['Feature'],
            "value": float(row['Value']),
            "shap_impact": float(row['SHAP_Value'])
        })

    churn_risk_drivers = []
    for _, row in top_negative_drivers.iterrows():
        churn_risk_drivers.append({
            "feature": row['Feature'],
            "value": float(row['Value']),
            "shap_impact": float(row['SHAP_Value'])
        })

    return {
        "policy_id": policy_id,
        "predicted_status": predicted_status_name,
        "retention_drivers_stay": retention_drivers,
        "churn_risk_drivers_churn": churn_risk_drivers,
        "actionable_routing": actionable_routing
    }


def full_prediction_pipeline(sample_user_data, model, feature_cols_order, maps, explainer_obj):
    """
    Tests the full prediction and explanation pipeline with sample user data.
    Returns a dictionary containing prediction and SHAP explanation results.
    """
    # Get prediction
    prediction_result = get_user_prediction(model, sample_user_data, feature_cols_order, maps)

    # Get SHAP explanation factors and action routing
    shap_explanation_result = get_user_shap_factors(explainer_obj, prediction_result, maps)

    return {
        "prediction": prediction_result,
        "explanation": shap_explanation_result
    }

