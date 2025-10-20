"""
FUREcast - Model Prediction Interface

This module provides a clean interface for making predictions with the trained GBR model.
Handles model loading, feature alignment, scaling, and prediction formatting.

Usage:
    from predict import load_model, predict_return, get_feature_importance
    
    model_bundle = load_model()
    prediction = predict_return(model_bundle, features_df)
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings

# Setup paths
MODULE_DIR = Path(__file__).parent
PRED_MODEL_DIR = MODULE_DIR.parent
MODELS_DIR = PRED_MODEL_DIR / "models"


class ModelBundle:
    """Container for all model artifacts."""
    
    def __init__(self, model, scaler, feature_names, metrics, metadata):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.metrics = metrics
        self.metadata = metadata
    
    def __repr__(self):
        return (f"ModelBundle(model={type(self.model).__name__}, "
                f"n_features={len(self.feature_names)}, "
                f"test_r2={self.metrics['test']['r2']:.4f})")


def load_model() -> ModelBundle:
    """
    Load trained GBR model and all artifacts.
    
    Returns:
        ModelBundle: Container with model, scaler, feature names, and metadata
    
    Raises:
        FileNotFoundError: If model files are not found
    """
    try:
        model = joblib.load(MODELS_DIR / "gbr_model.pkl")
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")
        
        with open(MODELS_DIR / "metrics.json", 'r') as f:
            metrics = json.load(f)
        
        with open(MODELS_DIR / "model_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        return ModelBundle(model, scaler, feature_names, metrics, metadata)
    
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Model artifacts not found in {MODELS_DIR}. "
            "Please train the model first using train_gbr_model.py"
        ) from e


def align_features(features: pd.DataFrame, expected_features: List[str]) -> pd.DataFrame:
    """
    Align input features to match training feature order.
    
    Args:
        features: DataFrame with feature columns
        expected_features: List of feature names from training
    
    Returns:
        DataFrame with columns reordered to match training
    
    Raises:
        ValueError: If required features are missing
    """
    missing_features = set(expected_features) - set(features.columns)
    
    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}\n"
            f"Expected {len(expected_features)} features, got {len(features.columns)}"
        )
    
    # Select and reorder columns
    return features[expected_features]


def predict_return(model_bundle: ModelBundle, 
                  features: pd.DataFrame,
                  return_confidence: bool = True) -> Dict[str, Any]:
    """
    Predict next-day return using trained model.
    
    Args:
        model_bundle: Loaded model bundle
        features: DataFrame with feature columns (single row or multiple rows)
        return_confidence: Whether to compute confidence score
    
    Returns:
        Dictionary with prediction results:
        - predicted_return: float or array
        - direction: 'up', 'down', or 'neutral'
        - confidence: float (0-1)
        - raw_prediction: unscaled model output
    
    Raises:
        ValueError: If features are invalid
    """
    # Align features
    features_aligned = align_features(features, model_bundle.feature_names)
    
    # Scale features
    features_scaled = model_bundle.scaler.transform(features_aligned)
    
    # Predict
    prediction = model_bundle.model.predict(features_scaled)
    
    # Handle single vs multiple predictions
    if len(prediction) == 1:
        pred_return = float(prediction[0])
        
        # Determine direction
        if pred_return > 0.002:
            direction = 'up'
        elif pred_return < -0.002:
            direction = 'down'
        else:
            direction = 'neutral'
        
        # Simple confidence based on magnitude (can be improved)
        confidence = min(abs(pred_return) * 20, 0.95)  # Scale to 0-0.95 range
        confidence = max(confidence, 0.5)  # Minimum 0.5
        
        return {
            'predicted_return': pred_return,
            'direction': direction,
            'confidence': confidence,
            'raw_prediction': pred_return
        }
    
    else:
        # Multiple predictions
        directions = ['up' if p > 0.002 else 'down' if p < -0.002 else 'neutral' 
                     for p in prediction]
        confidences = [min(abs(p) * 20, 0.95) for p in prediction]
        confidences = [max(c, 0.5) for c in confidences]
        
        return {
            'predicted_return': prediction.tolist(),
            'direction': directions,
            'confidence': confidences,
            'raw_prediction': prediction.tolist()
        }


def get_feature_importance(model_bundle: ModelBundle, 
                          top_n: int = 10) -> List[Dict[str, Any]]:
    """
    Get top N most important features from the model.
    
    Args:
        model_bundle: Loaded model bundle
        top_n: Number of top features to return
    
    Returns:
        List of dicts with 'name' and 'importance' keys
    """
    feature_importance = pd.DataFrame({
        'name': model_bundle.feature_names,
        'importance': model_bundle.model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return feature_importance.head(top_n).to_dict('records')


def predict_with_explanation(model_bundle: ModelBundle,
                             features: pd.DataFrame,
                             top_features: int = 5) -> Dict[str, Any]:
    """
    Make prediction and include feature importance explanation.
    
    Args:
        model_bundle: Loaded model bundle
        features: DataFrame with feature columns (single row)
        top_features: Number of top features to include
    
    Returns:
        Dictionary with prediction and explanation:
        - predicted_return: float
        - direction: str
        - confidence: float
        - top_features: list of {name, importance, value}
        - model_performance: dict with test metrics
    """
    # Get prediction
    prediction = predict_return(model_bundle, features)
    
    # Get top features with their values
    top_feature_importance = get_feature_importance(model_bundle, top_features)
    
    # Add actual feature values
    features_aligned = align_features(features, model_bundle.feature_names)
    for feat in top_feature_importance:
        feat['value'] = float(features_aligned[feat['name']].iloc[0])
    
    # Combine results
    result = {
        **prediction,
        'top_features': top_feature_importance,
        'model_performance': {
            'test_r2': model_bundle.metrics['test']['r2'],
            'test_directional_accuracy': model_bundle.metrics['test']['directional_accuracy'],
            'test_rmse': model_bundle.metrics['test']['rmse']
        }
    }
    
    return result


def batch_predict(model_bundle: ModelBundle,
                 features: pd.DataFrame) -> pd.DataFrame:
    """
    Make predictions for a batch of feature rows.
    
    Args:
        model_bundle: Loaded model bundle
        features: DataFrame with feature columns (multiple rows)
    
    Returns:
        DataFrame with original features plus prediction columns
    """
    # Get predictions
    prediction_result = predict_return(model_bundle, features, return_confidence=False)
    
    # Add to dataframe
    result_df = features.copy()
    result_df['predicted_return'] = prediction_result['predicted_return']
    result_df['predicted_direction'] = prediction_result['direction']
    
    return result_df


def validate_features(features: pd.DataFrame, 
                     model_bundle: ModelBundle) -> Dict[str, Any]:
    """
    Validate that features are in acceptable ranges.
    
    Args:
        features: DataFrame with feature columns
        model_bundle: Loaded model bundle
    
    Returns:
        Dict with validation results:
        - valid: bool
        - warnings: list of warning messages
        - errors: list of error messages
    """
    warnings_list = []
    errors_list = []
    
    # Check for NaN values
    nan_cols = features.columns[features.isna().any()].tolist()
    if nan_cols:
        errors_list.append(f"NaN values found in columns: {nan_cols}")
    
    # Check for infinite values
    inf_cols = features.columns[np.isinf(features).any()].tolist()
    if inf_cols:
        errors_list.append(f"Infinite values found in columns: {inf_cols}")
    
    # Check for extreme outliers (values > 10 std from training mean)
    # This is a simplified check - you could make it more sophisticated
    for col in features.columns:
        if col in model_bundle.feature_names:
            values = features[col]
            if (abs(values) > 100).any():  # Simple threshold
                warnings_list.append(f"Extreme values detected in {col}")
    
    return {
        'valid': len(errors_list) == 0,
        'warnings': warnings_list,
        'errors': errors_list
    }


# Convenience function for single-line usage
def quick_predict(features: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience function: Load model and make prediction in one call.
    
    Args:
        features: DataFrame with feature columns (single row)
    
    Returns:
        Prediction dictionary
    """
    model_bundle = load_model()
    return predict_with_explanation(model_bundle, features)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing model prediction interface...")
    
    try:
        # Load model
        model_bundle = load_model()
        print(f"✓ Model loaded: {model_bundle}")
        
        # Show model info
        print(f"\nModel Performance (Test Set):")
        print(f"  R² Score: {model_bundle.metrics['test']['r2']:.4f}")
        print(f"  RMSE: {model_bundle.metrics['test']['rmse']:.6f}")
        print(f"  Directional Accuracy: {model_bundle.metrics['test']['directional_accuracy']:.2%}")
        
        # Show top features
        print(f"\nTop 5 Features:")
        top_feats = get_feature_importance(model_bundle, 5)
        for i, feat in enumerate(top_feats, 1):
            print(f"  {i}. {feat['name']}: {feat['importance']:.4f}")
        
        print("\n✓ Model interface is working correctly")
        print(f"\nTo use this module:")
        print(f"  from predict import load_model, predict_with_explanation")
        print(f"  model_bundle = load_model()")
        print(f"  result = predict_with_explanation(model_bundle, features_df)")
        
    except FileNotFoundError:
        print("✗ Model not found. Please train the model first:")
        print("  python scripts/train_gbr_model.py --quick")
    except Exception as e:
        print(f"✗ Error: {e}")
