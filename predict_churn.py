import joblib
import pandas as pd
import numpy as np
from config import MODEL_PATH

# Load the model
def load_model(model_path=MODEL_PATH):
    """Load the trained model from the specified path."""
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Warning: Could not load model from {model_path}. Error: {e}")
        return None


def get_feature_order(model):
    """Attempts to extract feature names from the model pipeline if saved."""
    try:
        # Works if the pipeline or final estimator has feature_names_in_ attribute
        if hasattr(model, "feature_names_in_"):
            return model.feature_names_in_
        # If it's a scikit-learn pipeline, check the last step (e.g., the classifier)
        if hasattr(model, "steps"):
            last_step = model.steps[-1][1]
            if hasattr(last_step, "feature_names_in_"):
                return last_step.feature_names_in_
    except Exception:
        pass
        return []


def get_feature_categories(model):
    preprocessor = model.named_steps["preprocessor"]
    categories = {}

    for _, transformer, columns in preprocessor.transformers_:
        encoder = transformer

        # Handle a Pipeline containing OneHotEncoder
        if hasattr(transformer, "named_steps"):
            for step in transformer.named_steps.values():
                if hasattr(step, "categories_"):
                    encoder = step
                    break

        if hasattr(encoder, "categories_"):
            for column, values in zip(columns, encoder.categories_):
                categories[column] = values.tolist()

    return categories


def get_user_prediction(model, user_data_dict, feature_columns_order):
    """
    Takes a trained model and a dictionary of user input,
    and returns the predicted churn probability.

    Args:
        model: The trained scikit-learn pipeline model.
        user_data_dict (dict): A dictionary where keys are feature names
                                and values are the user's input for those features.
                                It's crucial that this dictionary contains
                                all features present in the training data (X),
                                with correct keys and appropriate data types.
        feature_columns_order (pd.Index): A pandas Index object representing
                                          the exact order of features used during training.

    Returns:
        float: The predicted churn probability for the user.
    """
    # Create a DataFrame from the single user input dictionary
    user_input_df = pd.DataFrame([user_data_dict])

    # Reindex to match the original training features' column order.
    # Missing columns will be filled with NaN. It's assumed the user provides complete input.
    user_input_df = user_input_df.reindex(columns=feature_columns_order, fill_value=np.nan)
    
    # The model pipeline (loaded_model) handles preprocessing internally
    churn_probability = model.predict_proba(user_input_df)[:, 1][0]
    churn_class = 1 if churn_probability > 0.5 else 0
    
    return churn_probability, churn_class