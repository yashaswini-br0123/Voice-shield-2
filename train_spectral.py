import os
import argparse
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from backend.audio.preprocessing import load_and_preprocess_audio
from backend.audio.feature_extraction import extract_lfcc_features
from backend.config import settings


def train_spectral_model(dataset_dir="datasets", output_model_path=None):
    if output_model_path is None:
        output_model_path = settings.SPECTRAL_MODEL_PATH

    real_dir = os.path.join(dataset_dir, "real")
    fake_dir = os.path.join(dataset_dir, "fake")

    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print(f"Error: Dataset directories '{real_dir}' or '{fake_dir}' not found.")
        print("Run 'python datasets/generate_samples.py' first or create the directories.")
        return

    X = []
    y = []

    # Process Genuine Human Audio (Label 0)
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
    print(f"Extracting LFCC features from {len(real_files)} Genuine Human audio samples...")
    for file_path in real_files:
        try:
            audio, meta = load_and_preprocess_audio(file_path)
            lfcc_feat = extract_lfcc_features(audio, sr=meta["sample_rate"])
            X.append(lfcc_feat)
            y.append(0) # 0 = Human
        except Exception as e:
            print(f"Failed to process '{file_path}': {e}")

    # Process AI-Generated Audio (Label 1)
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
    print(f"Extracting LFCC features from {len(fake_files)} AI-Generated audio samples...")
    for file_path in fake_files:
        try:
            audio, meta = load_and_preprocess_audio(file_path)
            lfcc_feat = extract_lfcc_features(audio, sr=meta["sample_rate"])
            X.append(lfcc_feat)
            y.append(1) # 1 = AI / Synthetic
        except Exception as e:
            print(f"Failed to process '{file_path}': {e}")

    if len(X) == 0:
        print("No valid audio files processed.")
        return

    X = np.array(X)
    y = np.array(y)

    print(f"\nDataset shape: Features {X.shape}, Labels {y.shape}")

    # Train / Test split if dataset is sufficiently large
    if len(X) >= 6:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    # Train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Training Complete! Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Genuine Human", "AI Synthetic"]))

    # Save Model Weights
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump(clf, output_model_path)
    print(f"Saved Layer 3 Spectral model weights to '{output_model_path}'\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Layer 3 LFCC Spectral Classifier")
    parser.add_argument("--dataset_dir", type=str, default="datasets", help="Path to dataset folder containing real/ and fake/")
    parser.add_argument("--output_model", type=str, default=None, help="Output path for joblib model file")
    args = parser.parse_args()

    train_spectral_model(args.dataset_dir, args.output_model)
