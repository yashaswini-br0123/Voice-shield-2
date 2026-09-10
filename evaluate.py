import os
import argparse
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

from backend.audio.preprocessing import load_and_preprocess_audio
from backend.models.acoustic_detector import AcousticDetector
from backend.models.aasist_detector import AASISTDetector
from backend.models.spectral_detector import SpectralDetector
from backend.ensemble.fusion import EnsembleFusion


def evaluate_system(dataset_dir="datasets"):
    real_dir = os.path.join(dataset_dir, "real")
    fake_dir = os.path.join(dataset_dir, "fake")

    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print(f"Error: Dataset paths '{real_dir}' or '{fake_dir}' do not exist.")
        print("Run 'python datasets/generate_samples.py' first.")
        return

    # Instantiate detectors
    l1_model = AcousticDetector()
    l2_model = AASISTDetector()
    l3_model = SpectralDetector()
    fusion = EnsembleFusion()

    y_true = []
    
    # Store probability predictions for each model layer
    scores_l1 = []
    scores_l2 = []
    scores_l3 = []
    scores_ensemble = []

    # Gather test files
    real_files = [(os.path.join(real_dir, f), 0) for f in os.listdir(real_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
    fake_files = [(os.path.join(fake_dir, f), 1) for f in os.listdir(fake_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
    all_files = real_files + fake_files

    if not all_files:
        print("No test files found.")
        return

    print(f"Evaluating VoiceShield performance on {len(all_files)} audio samples...\n")

    for file_path, label in all_files:
        try:
            audio, meta = load_and_preprocess_audio(file_path)
            
            s1, d1 = l1_model.analyze(audio, sr=meta["sample_rate"])
            s2, d2 = l2_model.analyze(audio, sr=meta["sample_rate"])
            s3, d3 = l3_model.analyze(audio, sr=meta["sample_rate"])
            
            ens = fusion.fuse_scores(d1, d2, d3)
            s_ens = ens["ai_probability"]

            y_true.append(label)
            scores_l1.append(s1)
            scores_l2.append(s2)
            scores_l3.append(s3)
            scores_ensemble.append(s_ens)
            
        except Exception as e:
            print(f"Skipping file '{file_path}': {e}")

    y_true = np.array(y_true)
    
    def compute_metrics(probs, threshold=0.5):
        probs = np.array(probs)
        preds = (probs >= threshold).astype(int)
        
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, probs)
        except Exception:
            auc = 0.5

        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn + 1e-8)
        fnr = fn / (fn + tp + 1e-8)

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
            "fpr": fpr,
            "fnr": fnr,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn
        }

    m1 = compute_metrics(scores_l1)
    m2 = compute_metrics(scores_l2)
    m3 = compute_metrics(scores_l3)
    m_ens = compute_metrics(scores_ensemble)

    print("=" * 80)
    print("                VOICESHIELD SYSTEM EVALUATION REPORT")
    print("=" * 80)

    header = f"{'Metric':<18} | {'Layer 1 (Acoustic)':<18} | {'Layer 2 (Waveform)':<18} | {'Layer 3 (Spectral)':<18} | {'3-Layer Ensemble':<18}"
    print(header)
    print("-" * len(header))

    metrics_keys = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1"),
        ("ROC-AUC", "auc"),
        ("False Pos Rate", "fpr"),
        ("False Neg Rate", "fnr")
    ]

    for label_str, key in metrics_keys:
        val1 = f"{m1[key]*100:.2f}%"
        val2 = f"{m2[key]*100:.2f}%"
        val3 = f"{m3[key]*100:.2f}%"
        val_ens = f"{m_ens[key]*100:.2f}%"
        print(f"{label_str:<18} | {val1:<18} | {val2:<18} | {val3:<18} | {val_ens:<18}")

    print("-" * len(header))
    print("\nConfusion Matrix Summary (3-Layer Ensemble):")
    print(f"  True Positives (AI detected as AI):     {m_ens['tp']}")
    print(f"  True Negatives (Human detected as Human): {m_ens['tn']}")
    print(f"  False Positives (Human misclassified as AI): {m_ens['fp']}")
    print(f"  False Negatives (AI misclassified as Human): {m_ens['fn']}\n")
    print("Evaluation Complete!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VoiceShield Multi-Layer Engine")
    parser.add_argument("--dataset_dir", type=str, default="datasets", help="Directory containing real/ and fake/ subfolders")
    args = parser.parse_args()

    evaluate_system(args.dataset_dir)
