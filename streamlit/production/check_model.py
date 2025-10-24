"""
Quick diagnostic to check if model files exist
"""
from pathlib import Path

pred_model_dir = Path(__file__).parent / "pred_model"
models_dir = pred_model_dir / "models"

print("=" * 60)
print("Model File Diagnostic")
print("=" * 60)

print(f"\nPred model directory: {pred_model_dir}")
print(f"Exists: {pred_model_dir.exists()}")

print(f"\nModels directory: {models_dir}")
print(f"Exists: {models_dir.exists()}")

if models_dir.exists():
    print("\nFiles in models directory:")
    for file in models_dir.iterdir():
        print(f"  ✓ {file.name} ({file.stat().st_size:,} bytes)")
else:
    print("\n❌ Models directory not found!")
    print("\nTo train the model:")
    print("  cd streamlit/production/pred_model")
    print("  ./train_and_evaluate.sh")

# Try importing
print("\n" + "=" * 60)
print("Import Test")
print("=" * 60)

try:
    import sys
    sys.path.insert(0, str(pred_model_dir))
    
    from predict import load_model
    from get_latest_features import get_latest_features
    
    print("✓ Imports successful")
    
    try:
        model_bundle = load_model()
        print("✓ Model loaded successfully")
        print(f"  Features: {len(model_bundle['feature_names'])}")
        print(f"  Test R²: {model_bundle['metrics']['test']['r2']:.4f}")
    except FileNotFoundError as e:
        print(f"❌ Model files not found: {e}")
        
except ImportError as e:
    print(f"❌ Import error: {e}")