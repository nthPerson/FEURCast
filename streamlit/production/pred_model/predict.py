"""
FUREcast - Model Prediction Interface

This module provides functions to load the trained GBR model and make predictions.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# Setup paths - make them relative to THIS file's location
MODULE_DIR = Path(__file__).parent  # pred_model directory
MODELS_DIR = MODULE_DIR / "models"
DATA_PATH = MODULE_DIR.parent.parent.parent / "data" / "rich_features_SPLG_history_full.csv"

def load_model(models_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load trained GBR model and associated artifacts.
    
    Args:
        models_dir: Optional custom models directory. If None, uses default.
    
    Returns:
        Dictionary containing:
        - model: Trained GradientBoostingRegressor
        - scaler: Fitted StandardScaler
        - feature_names: List of feature column names
        - metrics: Model performance metrics
        - metadata: Training metadata
    
    Raises:
        FileNotFoundError: If model files don't exist
    """
    if models_dir is None:
        models_dir = MODELS_DIR
    
    # Check if models directory exists
    if not models_dir.exists():
        raise FileNotFoundError(
            f"Models directory not found: {models_dir}\n"
            f"Please train the model first using train_gbr_model.py"
        )
    
    # Check for required files
    required_files = ['gbr_model.pkl', 'scaler.pkl', 'feature_names.pkl', 'metrics.json']
    missing_files = [f for f in required_files if not (models_dir / f).exists()]
    
    if missing_files:
        raise FileNotFoundError(
            f"Missing model files: {missing_files}\n"
            f"Please train the model first using train_gbr_model.py"
        )
    
    # Load all artifacts
    model = joblib.load(models_dir / "gbr_model.pkl")
    scaler = joblib.load(models_dir / "scaler.pkl")
    feature_names = joblib.load(models_dir / "feature_names.pkl")
    
    with open(models_dir / "metrics.json", 'r') as f:
        metrics = json.load(f)
    
    # Load metadata if available
    metadata = {}
    metadata_path = models_dir / "model_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'metrics': metrics,
        'metadata': metadata
    }


def predict_return(features_df: pd.DataFrame, 
                   model_bundle: Dict[str, Any]) -> float:
    """
    Predict next-day return for given features.
    
    Args:
        features_df: DataFrame with feature columns (single row or multiple rows)
        model_bundle: Dictionary from load_model() containing model, scaler, etc.
    
    Returns:
        Predicted return as float (e.g., 0.0023 = 0.23% return)
    """
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    feature_names = model_bundle['feature_names']
    
    # Ensure features are in correct order and all present
    if not all(col in features_df.columns for col in feature_names):
        missing = [col for col in feature_names if col not in features_df.columns]
        raise ValueError(f"Missing required features: {missing[:5]}...")
    
    # Select and order features
    X = features_df[feature_names].copy()
    
    # Handle any remaining NaN or inf values
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any():
        # Fill with column means (from training data if possible)
        X = X.fillna(0)  # Conservative: fill with 0 after scaling
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    
    return float(prediction)


def predict_with_explanation(features_df: pd.DataFrame, 
                            model_bundle: Dict[str, Any],
                            top_n: int = 10) -> Dict[str, Any]:
    """
    Make prediction with feature importance explanation.
    
    Args:
        features_df: DataFrame with feature columns (single row or multiple rows)
        model_bundle: Dictionary from load_model() containing model, scaler, etc.
        top_n: Number of top features to return in explanation
    
    Returns:
        Dictionary with:
        - predicted_return: float
        - direction: 'up', 'down', or 'neutral'
        - confidence: str ('low', 'medium', 'high')
        - top_features: list of dicts with 'name' and 'importance'
        - metrics: model performance metrics
    """
    # Make prediction
    predicted_return = predict_return(features_df, model_bundle)
    
    # Get feature importance
    model = model_bundle['model']
    feature_names = model_bundle['feature_names']
    feature_importance = model.feature_importances_
    
    # Sort features by importance
    importance_df = pd.DataFrame({
        'name': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # Get top N features
    top_features = importance_df.head(top_n).to_dict('records')
    
    # Determine direction
    if predicted_return > 0.002:
        direction = 'up'
    elif predicted_return < -0.002:
        direction = 'down'
    else:
        direction = 'neutral'
    
    # Determine confidence based on magnitude
    abs_return = abs(predicted_return)
    if abs_return < 0.003:
        confidence = 'low'
    elif abs_return < 0.008:
        confidence = 'medium'
    else:
        confidence = 'high'
    
    return {
        'predicted_return': float(predicted_return),
        'direction': direction,
        'confidence': confidence,
        'top_features': top_features,
        'metrics': model_bundle.get('metrics', {})
    }


def batch_predict(features_df: pd.DataFrame, 
                  model_bundle: Dict[str, Any]) -> np.ndarray:
    """
    Make predictions for multiple rows of features.
    
    Args:
        features_df: DataFrame with feature columns (multiple rows)
        model_bundle: Dictionary from load_model()
    
    Returns:
        NumPy array of predictions
    """
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    feature_names = model_bundle['feature_names']
    
    # Select and order features
    X = features_df[feature_names].copy()
    
    # Handle any remaining NaN or inf values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    return predictions


# Self-test
if __name__ == "__main__":
    print("Testing prediction module...")
    
    try:
        # Load model
        model_bundle = load_model()
        print(f"✓ Model loaded successfully")
        print(f"  Features: {len(model_bundle['feature_names'])}")
        print(f"  Test R²: {model_bundle['metrics']['test']['r2']:.4f}")
        
        # Try to get latest features
        from get_latest_features import get_latest_features
        features = get_latest_features(1)
        print(f"✓ Latest features retrieved: {features.shape}")
        
        # Make prediction
        result = predict_with_explanation(features, model_bundle, top_n=5)
        print(f"\n✓ Prediction made:")
        print(f"  Return: {result['predicted_return']*100:.3f}%")
        print(f"  Direction: {result['direction']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"\n  Top 5 features:")
        for feat in result['top_features']:
            print(f"    - {feat['name']}: {feat['importance']:.4f}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
