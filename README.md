# 💳 Credit Card Fraud Detection
> Machine Learning Assignment — Model Comparison Study

A machine learning project that detects fraudulent credit card transactions by comparing 4 classification models: **Logistic Regression, Random Forest, Decision Tree, and SVM**.

---

## 👥 Team Members

| Member | Model Assigned | GitHub Username |
|--------|---------------|-----------------|
| [Name] | Logistic Regression | @username |
| [Name] | Random Forest | @username |
| [Name] | Decision Tree | @username |
| [Name] | SVM | @username |


---

## 📂 Project Structure

```
ML_Final_Project/
│
├── dataset/                        ← Raw CSV (NOT on GitHub — laptop & Drive only)
│   └── creditcard.csv
│
├── cleaned_data/                   ← Preprocessed splits (NOT on GitHub — Drive only)
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
│
├── preprocessing/
│   └── preprocessing.py            ← Scaling, SMOTE, train/test split, save/load
│
├── models/
│   ├── logistic_regression.py
│   ├── random_forest.py
│   ├── decision_tree.py
│   └── svm.py
│
├── notebooks/
│   ├── 00_preprocessing.ipynb      ← Team lead runs this ONCE
│   ├── 01_logistic_regression.ipynb
│   ├── 02_random_forest.ipynb
│   ├── 03_decision_tree.ipynb
│   └── 04_svm.ipynb
│
├── results/
│   └── plots/                      ← Confusion matrices, ROC curves, charts
│
├── .gitignore
└── README.md
```

---

## 📊 Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Fraud cases:** 492 (0.17%) — severely imbalanced
- **Features:** V1–V28 (PCA transformed), `Time`, `Amount`
- **Target:** `Class` → 0 = Legitimate, 1 = Fraud


---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/YOURNAME/credit-fraud-detection.git
cd credit-fraud-detection
```

### 2. Install Required Libraries
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn jupyter
```

### 3. Get the Shared Data (Google Drive)
- Ask the **team lead** to share the `FraudDetection` Google Drive folder with you
- It contains:
  - `dataset/creditcard.csv`
  - `cleaned_data/` — the preprocessed train/test splits

---

## 🚀 How to Run (Google Colab)

### Team Lead Only — Run Preprocessing Once
Open `notebooks/00_preprocessing.ipynb` in Colab and run all cells.
This saves the cleaned splits to the shared Google Drive folder.

### Every Member — Load Shared Data & Train Your Model
Add these cells at the top of your notebook:

```python
# Step 1 — Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 2 — Clone the repo (gets latest code)
!git clone https://github.com/YOURNAME/credit-fraud-detection.git
%cd credit-fraud-detection

# Step 3 — Install libraries
!pip install imbalanced-learn -q

# Step 4 — Load the shared splits
import sys
sys.path.append('/content/credit-fraud-detection/preprocessing')
from preprocessing import load_splits

X_train, X_test, y_train, y_test = load_splits(
    data_dir='/content/drive/MyDrive/FraudDetection/cleaned_data'
)
```

> ✅ All 4 members use the **exact same** X_train, X_test, y_train, y_test
> so the model comparison is fair.

---




### Branching (Recommended)
Each member works on their own branch to avoid conflicts:
```bash
# Create your branch (do this once)
git checkout -b yourname/svm

# Push your branch
git push origin yourname/svm
```

When done, create a **Pull Request** on GitHub to merge into `main`.

---

## 📏 Evaluation Metrics

All 4 models are evaluated using the same metrics:

| Metric | Why It Matters |
|--------|---------------|
| **Accuracy** | Overall correct predictions |
| **Precision** | Of predicted frauds, how many were real? |
| **Recall** | Of real frauds, how many did we catch? |
| **F1-Score** | Balance between Precision and Recall |
| **ROC-AUC** | Model's ability to separate fraud vs legit |

> ⚠️ Accuracy alone is misleading here — a model that predicts everything
> as "Legit" gets 99.8% accuracy but catches zero frauds!
> **Focus on F1-Score and Recall.**

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Programming language |
| Pandas / NumPy | Data manipulation |
| Scikit-learn | ML models & evaluation |
| Imbalanced-learn | SMOTE for class imbalance |
| Matplotlib / Seaborn | Visualizations |
| Google Colab | Cloud notebooks |
| Google Drive | Shared dataset storage |
| GitHub | Version control & collaboration |

---

## ❓ Common Issues

**Error: `creditcard.csv` not found**
→ Make sure Google Drive is mounted and the file path is correct.

**Error: `imbalanced-learn` not found**
→ Run `!pip install imbalanced-learn -q` in your Colab notebook.

**Error: `cleaned_data` folder not found**
→ The team lead hasn't run `00_preprocessing.ipynb` yet. Ask them to run it first.

**Merge conflict on GitHub**
→ Run `git pull origin main` before starting work each day.

---

## 📌 Important Reminders

- ❌ **Never commit** `creditcard.csv` or `cleaned_data/` to GitHub
- ✅ Always **git pull** before starting work
- ✅ All members must use the **same cleaned_data splits** for fair comparison
- ✅ Run SMOTE only on **training data**, never on test data