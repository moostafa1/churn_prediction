# AI Churn Prediction Model (v2.0) - Technical & Architectural Summary

## 1. Business Context & Problem Statement
The objective of this project is to build an end-to-end Machine Learning pipeline to predict policyholder churn within a **60-to-90-day prediction horizon**. By identifying customers at risk of exiting early, operational teams (Collections, Retention, Complaints, and Portfolio Management) can proactively deploy targeted retention interventions.

### Key Exit Definitions
- **Surrender:** Voluntary exit where payout occurs (ULS policies with $\ge 24$ months tenure and paid installments).
- **Cancellation:** Policy termination without payout (ULS under 2 years or pure protection policies).
- **Lapsation:** Automatic coverage termination driven by unpaid premiums depleting the account value.

---

## 2. ML System Architecture & Execution Plan

### A. Feature Engineering Categories
1. **Demographic & Policy Data:** Age band, marital status, policy type, tenure, number of active policies, auto-pay status.
2. **Billing & Premium Metrics:** Payment frequency, late payment count, price increase counts, percentage premium changes year-over-year.
3. **Account & Claim Behavioral Signals:** 12-month claim history, payout ratios, settlement times, withdrawal frequency/velocity.
4. **Interaction & Feedback Metrics:** Customer contact counts, CRM/Jira ticket logs, complaint flags, resolution times.

### B. Machine Learning Modeling Strategy
- **Task:** Binary / Multi-class Classification & Survival Analysis.
- **Primary Metrics:** Precision-Recall AUC (PR-AUC), Recall@K (Top 10% Risk Tier).
- **Model Baseline:** Gradient Boosted Trees (XGBoost / LightGBM) + NLP vectorizers (TF-IDF / Sentence Transformers) for unstructured complaint notes.

---

## 3. Inference API Specifications

### Request Schema (`POST /predict`)
```json
{
  "age_band": "35-44",
  "marital_status": "Married",
  "customer_tenure_months": 24,
  "multi_policy_flag": 1,
  "num_policies": 2,
  "policy_type": "Auto",
  "renewal_month": "8",
  "current_premium": 1250.50,
  "premium_last_year": 1150.00,
  "premium_change_pct": 8.74,
  "num_price_increases_last_3y": 1,
  "coverage_amount": 50000.0,
  "premium_to_coverage_ratio": 0.025,
  "payment_frequency": "Monthly",
  "autopay_enabled": 0,
  "late_payment_count_12m": 1,
  "missed_payment_flag": 0,
  "num_claims_12m": 0,
  "num_approved_claims_12m": 0,
  "num_rejected_claims_12m": 0,
  "num_pending_claims_12m": 0,
  "avg_claim_amount": 0.0,
  "total_claim_amount_12m": 0.0,
  "total_payout_amount_12m": 0.0,
  "payout_ratio_12m": 0.0,
  "avg_settlement_time_days": 0.0,
  "days_since_last_claim": 365,
  "num_contacts_12m": 2,
  "complaint_flag": 0,
  "complaint_resolution_days": 0.0,
  "coverage_downgrade_flag": 0
}
```

### Response Schema
```json
{
  "churn_probability": 0.7022,
  "churn_class": 1,
  "prediction_label": "Will Churn"
}
```

---

## 4. Intervention & Routing Matrix

| Risk Level | Key Trigger Factors | Operational Action / Destination |
| :--- | :--- | :--- |
| **High ($\ge 0.70$)** | Late payments, auto-pay disabled, recent premium increases | Route to **Collections & Retention Queue (Jira)** for immediate outreach |
| **Medium ($0.40 - 0.69$)** | Low performance/yield concerns, account balance drawdown | Route to **Portfolio Management Team** for fund switching/restructuring |
| **Low ($< 0.40$)** | Active policyholder, stable payment pattern | Maintain regular promotional & digital self-service messaging |