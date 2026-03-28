"""
train_model.py
==============
Step 1: Extracts MediaPipe landmarks from ASL dataset images → landmarks.csv
Step 2: Trains RandomForest + MLP classifiers, picks the better one
Step 3: Saves model to models/gesture_classifier.pkl + models/label_map.json

Usage:
    python train_model.py                          # full pipeline
    python train_model.py --skip-extraction        # if landmarks.csv already exists
    python train_model.py --samples-per-class 300  # limit images per letter (faster)

Runtime: ~15 mins on Mac M-series, ~25 mins on Windows CPU
"""

import os
import cv2
import csv
import json
import argparse
import time
import platform
import numpy as np
import mediapipe as mp
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────

TRAIN_DIR       = "data/raw/asl_alphabet_train"   # extracted Kaggle dataset
LANDMARKS_CSV   = "data/landmarks.csv"
MODEL_OUT       = "models/gesture_classifier.pkl"
LABEL_MAP_OUT   = "models/label_map.json"

# Letters to include — drop J and Z (dynamic gestures, need motion)
INCLUDE_LABELS  = list("ABCDEFGHIKLMNOPQRSTUVWXY") + [str(i) for i in range(10)]

SAMPLES_PER_CLASS = 500   # max images to process per class (enough for 95%+ accuracy)
MIN_DETECTION_CONF = 0.5  # lower = more samples extracted, fewer skipped

# ── MediaPipe setup ─────────────────────────────────────────────────────────────

mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# ── Feature extraction ──────────────────────────────────────────────────────────

def extract_features(hand_landmarks):
    """
    Normalize 21 landmarks into a 63-dim vector.
    Subtracts wrist position and scales by bounding box — invariant to
    hand position in frame and distance from camera.
    """
    lm = hand_landmarks.landmark

    # Raw coords
    coords = [(p.x, p.y, p.z) for p in lm]

    # Wrist = landmark 0, use as origin
    wx, wy, wz = coords[0]

    # Scale factor = bounding box diagonal
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)

    features = []
    for (x, y, z) in coords:
        features.extend([
            (x - wx) / scale,
            (y - wy) / scale,
            z - wz
        ])
    return features


def extract_landmarks_from_dataset(train_dir, output_csv, samples_per_class, include_labels):
    """
    Walks train_dir/LABEL/*.jpg, runs MediaPipe on each image,
    saves [label, x0,y0,z0, ...x20,y20,z20] rows to output_csv.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Find all class folders
    all_folders = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
        and d.upper() in [l.upper() for l in include_labels]
    ])

    if not all_folders:
        print(f"\nERROR: No class folders found in {train_dir}")
        print(f"Expected folders like: A/, B/, C/ ...")
        print(f"Found: {os.listdir(train_dir)[:10]}")
        return False

    print(f"\nFound {len(all_folders)} classes: {all_folders}")
    print(f"Processing up to {samples_per_class} images per class...\n")

    total_saved   = 0
    total_skipped = 0
    start_time    = time.time()

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        header = ["label"] + [f"{ax}{i}" for i in range(21) for ax in ["x","y","z"]]
        writer.writerow(header)

        with mp_hands.Hands(
            static_image_mode=True,       # important for images (not video)
            max_num_hands=1,
            min_detection_confidence=MIN_DETECTION_CONF
        ) as hands:

            for cls_idx, folder in enumerate(all_folders):
                cls_dir  = os.path.join(train_dir, folder)
                label    = folder.upper()

                # Get image files
                img_files = [
                    f for f in os.listdir(cls_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ][:samples_per_class]

                saved   = 0
                skipped = 0

                for img_file in img_files:
                    img_path = os.path.join(cls_dir, img_file)
                    img = cv2.imread(img_path)
                    if img is None:
                        skipped += 1
                        continue

                    # MediaPipe needs RGB
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)

                    if results.multi_hand_landmarks:
                        features = extract_features(results.multi_hand_landmarks[0])
                        writer.writerow([label] + [f"{v:.6f}" for v in features])
                        saved += 1
                    else:
                        skipped += 1

                total_saved   += saved
                total_skipped += skipped

                # Progress
                elapsed = time.time() - start_time
                pct     = (cls_idx + 1) / len(all_folders) * 100
                eta     = (elapsed / (cls_idx + 1)) * (len(all_folders) - cls_idx - 1)
                bar     = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                print(f"  [{bar}] {pct:5.1f}%  {label}: {saved} saved, {skipped} skipped  ETA: {eta:.0f}s")

    print(f"\nExtraction complete!")
    print(f"  Saved:   {total_saved} landmarks")
    print(f"  Skipped: {total_skipped} (no hand detected)")
    print(f"  Output:  {output_csv}")
    return True


# ── Training ────────────────────────────────────────────────────────────────────

def train(landmarks_csv, model_out, label_map_out):
    """
    Loads landmarks.csv, trains RandomForest + MLP, saves the better one.
    Prints full classification report + confusion matrix.
    """
    print("\nLoading dataset...")
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.preprocessing import LabelEncoder
    import joblib
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv(landmarks_csv)
    print(f"  Rows: {len(df)}")
    print(f"  Classes: {sorted(df['label'].unique())}")
    print(f"  Samples per class:\n{df['label'].value_counts().sort_index()}\n")

    X = df.drop("label", axis=1).values
    y = df["label"].values

    # Encode labels to integers
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Save label map
    label_map = {str(i): cls for i, cls in enumerate(le.classes_)}
    os.makedirs(os.path.dirname(label_map_out), exist_ok=True)
    with open(label_map_out, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Label map saved → {label_map_out}")

    # Train/test split (stratified — equal class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    results = {}

    # ── Model A: RandomForest (fast, interpretable) ─────────────────────────
    print("Training Model A: RandomForest...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        n_jobs=-1,          # use all CPU cores
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc   = accuracy_score(y_test, rf_preds)
    results["RandomForest"] = (rf, rf_acc, rf_preds)
    print(f"  Accuracy: {rf_acc*100:.2f}%  ({time.time()-t0:.1f}s)\n")

    # ── Model B: MLP Neural Network (usually higher accuracy) ───────────────
    print("Training Model B: MLP Neural Network...")
    t0 = time.time()
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=False
    )
    mlp.fit(X_train, y_train)
    mlp_preds = mlp.predict(X_test)
    mlp_acc   = accuracy_score(y_test, mlp_preds)
    results["MLP"] = (mlp, mlp_acc, mlp_preds)
    print(f"  Accuracy: {mlp_acc*100:.2f}%  ({time.time()-t0:.1f}s)\n")

    # ── Pick winner ──────────────────────────────────────────────────────────
    best_name  = max(results, key=lambda k: results[k][1])
    best_model, best_acc, best_preds = results[best_name]
    print(f"Winner: {best_name} ({best_acc*100:.2f}%)")

    # ── Save model ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump({
        "model":      best_model,
        "encoder":    le,
        "model_name": best_name,
        "accuracy":   best_acc,
        "classes":    list(le.classes_)
    }, model_out)
    print(f"Model saved → {model_out}\n")

    # ── Classification report ─────────────────────────────────────────────────
    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, best_preds, target_names=le.classes_))

    # ── Confusion matrix (saved as image) ────────────────────────────────────
    cm = confusion_matrix(y_test, best_preds)
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=le.classes_, yticklabels=le.classes_
    )
    plt.title(f"Confusion Matrix — {best_name} ({best_acc*100:.1f}% accuracy)")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    cm_path = "data/confusion_matrix.png"
    plt.savefig(cm_path, dpi=120)
    print(f"\nConfusion matrix saved → {cm_path}")
    print("(Include this in your FYP report!)\n")

    # ── Model comparison summary ──────────────────────────────────────────────
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    for name, (_, acc, _) in results.items():
        bar    = "█" * int(acc * 40)
        chosen = " ← saved" if name == best_name else ""
        print(f"  {name:15} {acc*100:6.2f}%  {bar}{chosen}")

    return best_acc


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ISL/ASL gesture classifier")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip image processing, use existing landmarks.csv")
    parser.add_argument("--samples-per-class", type=int, default=SAMPLES_PER_CLASS,
                        help=f"Max images per class (default: {SAMPLES_PER_CLASS})")
    parser.add_argument("--train-dir", default=TRAIN_DIR,
                        help=f"Path to dataset folder (default: {TRAIN_DIR})")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" ISL/ASL Gesture Classifier Training Pipeline")
    print("=" * 60)

    # Step 1: Extract landmarks
    if not args.skip_extraction:
        if not os.path.exists(args.train_dir):
            print(f"\nERROR: Dataset not found at {args.train_dir}")
            print("Make sure you extracted the Kaggle dataset to that folder.")
            exit(1)

        success = extract_landmarks_from_dataset(
            train_dir         = args.train_dir,
            output_csv        = LANDMARKS_CSV,
            samples_per_class = args.samples_per_class,
            include_labels    = INCLUDE_LABELS
        )
        if not success:
            exit(1)
    else:
        if not os.path.exists(LANDMARKS_CSV):
            print(f"ERROR: {LANDMARKS_CSV} not found. Run without --skip-extraction first.")
            exit(1)
        print(f"Skipping extraction, using existing {LANDMARKS_CSV}")

    # Step 2: Train
    accuracy = train(LANDMARKS_CSV, MODEL_OUT, LABEL_MAP_OUT)

    print("\n" + "=" * 60)
    print(f" Training complete! Final accuracy: {accuracy*100:.2f}%")
    print(f" Model: {MODEL_OUT}")
    print(f" Labels: {LABEL_MAP_OUT}")
    print("=" * 60)
    print("\nNext step: run the app!")
    print("  python app.py\n")