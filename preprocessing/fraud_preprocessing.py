"""
preprocessing.py
================
Credit Card Fraud Detection — Data Preprocessing Pipeline
Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Dataset facts:
  - 284,807 transactions | 492 frauds (0.17%) → severely imbalanced
  - Features V1–V28 : already PCA-transformed (no further encoding needed)
  - Features to scale: 'Time' and 'Amount' only
  - Target column   : 'Class' (0 = Legit, 1 = Fraud)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────


def load_data(filepath: str) -> pd.DataFrame:
    """Load the CSV dataset and print a quick summary."""
    df = pd.read_csv(filepath)

    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)
    print(f"Shape            : {df.shape}")
    print(f"Missing values   : {df.isnull().sum().sum()}")
    print(f"\nClass distribution:")
    print(df["Class"].value_counts())
    print(f"\nFraud percentage : {df['Class'].mean() * 100:.4f}%")
    print("=" * 50)

    return df


# ─────────────────────────────────────────────
# 2. FEATURE SCALING
# ─────────────────────────────────────────────


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale 'Time' and 'Amount' using StandardScaler.
    V1-V28 are already PCA-scaled — do NOT scale them again.
    """
    df = df.copy()
    scaler = StandardScaler()

    df["scaled_amount"] = scaler.fit_transform(df[["Amount"]])
    df["scaled_time"] = scaler.fit_transform(df[["Time"]])

    # Drop original unscaled columns
    df.drop(columns=["Amount", "Time"], inplace=True)

    # Move Class to the last column
    cols = [c for c in df.columns if c != "Class"] + ["Class"]
    df = df[cols]

    print("✔ 'Amount' and 'Time' scaled → 'scaled_amount', 'scaled_time'")
    return df


# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Stratified 80/20 split — preserves fraud ratio in both sets.
    random_state=42 ensures EVERY team member gets the same split.
    """
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,  # ← FIXED SEED — do not change
        stratify=y,
    )

    print(f"\n✔ Train size     : {X_train.shape[0]:,} samples")
    print(f"✔ Test size      : {X_test.shape[0]:,} samples")
    print(f"✔ Fraud in train : {y_train.sum()} ({y_train.mean() * 100:.3f}%)")
    print(f"✔ Fraud in test  : {y_test.sum()} ({y_test.mean() * 100:.3f}%)")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# 4. HANDLE CLASS IMBALANCE WITH SMOTE
# ─────────────────────────────────────────────


def apply_smote(X_train, y_train, random_state: int = 42):
    """
    Apply SMOTE only to the TRAINING set.
    Never apply SMOTE to the test set — it would make evaluation dishonest.

    Why SMOTE?
      - 0.17% fraud rate means model predicts 'Legit' for everything
        and still gets 99.8% accuracy — completely useless.
      - SMOTE generates synthetic fraud samples so model actually learns.
    """
    print("\n── Before SMOTE ──")
    print(pd.Series(y_train).value_counts())

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print("\n── After SMOTE ──")
    print(pd.Series(y_resampled).value_counts())
    print(f"\n✔ Training set balanced: {X_resampled.shape[0]:,} samples")

    return X_resampled, y_resampled


# ─────────────────────────────────────────────
# 5. EDA VISUALIZATIONS
# ─────────────────────────────────────────────


def plot_class_distribution(df: pd.DataFrame, save_dir: str = "../results/plots"):
    """Bar chart — Legit vs Fraud transactions."""
    os.makedirs(save_dir, exist_ok=True)
    counts = df["Class"].value_counts()
    labels = ["Legit (0)", "Fraud (1)"]

    plt.figure(figsize=(6, 4))
    sns.barplot(x=labels, y=counts.values, palette=["steelblue", "crimson"])
    plt.title("Class Distribution — Credit Card Transactions")
    plt.ylabel("Number of Transactions")
    for i, v in enumerate(counts.values):
        plt.text(i, v + 500, f"{v:,}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/class_distribution.png", dpi=150)
    plt.show()
    print(f"✔ Saved → {save_dir}/class_distribution.png")


def plot_amount_distribution(df: pd.DataFrame, save_dir: str = "../results/plots"):
    """Compare transaction Amount for Legit vs Fraud."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, label, color, title in zip(
        axes,
        [0, 1],
        ["steelblue", "crimson"],
        ["Legit Transactions", "Fraudulent Transactions"],
    ):
        data = df[df["Class"] == label]["Amount"]
        ax.hist(data, bins=50, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("Amount (USD)")
        ax.set_ylabel("Count")
        ax.set_yscale("log")

    plt.suptitle("Transaction Amount Distribution", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/amount_distribution.png", dpi=150)
    plt.show()
    print(f"✔ Saved → {save_dir}/amount_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame, save_dir: str = "../results/plots"):
    """Correlation heatmap of all features."""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(16, 10))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, linewidths=0.3, annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/correlation_heatmap.png", dpi=150)
    plt.show()
    print(f"✔ Saved → {save_dir}/correlation_heatmap.png")


# ─────────────────────────────────────────────
# 6. SAVE SPLITS — Team Lead runs this ONCE
# ─────────────────────────────────────────────


def save_splits(X_train, X_test, y_train, y_test, output_dir: str = "../cleaned_data"):
    """
    Save the preprocessed train/test splits to CSV files.
    Team lead runs this ONCE then uploads to shared Google Drive.
    All teammates load from Drive — guarantees identical data for fair comparison.
    """
    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    print(f"\n✅ Splits saved to '{output_dir}/'")
    print(f"   X_train.csv : {X_train.shape}")
    print(f"   X_test.csv  : {X_test.shape}")
    print(f"   y_train.csv : {y_train.shape}")
    print(f"   y_test.csv  : {y_test.shape}")
    print(
        "\n👉 Upload the cleaned_data/ folder to Google Drive and share with teammates!"
    )


# ─────────────────────────────────────────────
# 7. LOAD SPLITS — Every member uses this
# ─────────────────────────────────────────────


def load_splits(data_dir: str = "../cleaned_data"):
    """
    Load the saved train/test splits from CSV.
    Every team member calls this instead of running preprocessing from scratch.
    Guarantees everyone uses IDENTICAL data → fair model comparison.
    """
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").squeeze()

    print("✅ Splits loaded successfully!")
    print(f"   X_train : {X_train.shape}")
    print(f"   X_test  : {X_test.shape}")
    print(f"   y_train : {y_train.shape} | Fraud cases: {y_train.sum()}")
    print(f"   y_test  : {y_test.shape}  | Fraud cases: {y_test.sum()}")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# 8. MASTER PIPELINE
# ─────────────────────────────────────────────


def preprocess(
    filepath: str,
    apply_smote_flag: bool = True,
    visualize: bool = True,
    save_dir: str = None,
):
    """
    Full preprocessing pipeline. Returns ready-to-use splits.

    Parameters
    ----------
    filepath         : path to creditcard.csv
    apply_smote_flag : balance training data with SMOTE (default: True)
    visualize        : generate and save EDA plots (default: True)
    save_dir         : if provided, saves splits as CSV to this directory

    Returns
    -------
    X_train, X_test, y_train, y_test
    """

    # Step 1 — Load
    df = load_data(filepath)

    # Step 2 — EDA plots (before scaling so Amount shows real dollar values)
    if visualize:
        plot_class_distribution(df)
        plot_amount_distribution(df)
        plot_correlation_heatmap(df)

    # Step 3 — Scale Time and Amount
    df = scale_features(df)

    # Step 4 — Stratified split
    X_train, X_test, y_train, y_test = split_data(df)

    # Step 5 — SMOTE on training data only
    if apply_smote_flag:
        X_train, y_train = apply_smote(X_train, y_train)

    # Step 6 — Optionally save splits to disk
    if save_dir:
        save_splits(X_train, X_test, y_train, y_test, output_dir=save_dir)

    print("\n✅ Preprocessing complete! Data is ready for modeling.\n")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# QUICK TEST — run this file directly to verify
# ─────────────────────────────────────────────

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess(
        filepath="../dataset/creditcard.csv", apply_smote_flag=True, visualize=True
    )

    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")
    print(f"y_train : {y_train.shape}")
    print(f"y_test  : {y_test.shape}")
