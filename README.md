# Fraud Detection Project - Interim 2 Submission

## 📋 Project Overview
This project aims to improve fraud detection for e-commerce and bank transactions by developing machine learning models that accurately identify fraudulent activities while balancing security and user experience.

## 🎯 Learning Outcomes
- Effectively clean, preprocess, and merge complex datasets
- Engineer meaningful features from raw data
- Implement techniques to handle highly imbalanced datasets
- Train and evaluate models using metrics appropriate for imbalanced classification
- Articulate and visualize model predictions using explainability tools like SHAP

## 📁 Repository Structure
\`\`\`
fraud-detection-interim2/
├── .github/              # GitHub workflows
├── .vscode/             # VSCode settings
├── data/                # Data directory (.gitignored)
├── notebooks/           # Jupyter notebooks
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb           # ✅ Interim 2: Model Building
│   └── shap-explainability.ipynb
├── src/                 # Source code modules
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── tests/               # Unit tests
├── models/              # Saved model artifacts
├── scripts/             # Utility scripts
├── requirements.txt     # Python dependencies
└── README.md           # This file
\`\`\`

## 🚀 Setup Instructions

### 1. Clone Repository
\`\`\`bash
git clone https://github.com/yourusername/fraud-detection-interim2.git
cd fraud-detection-interim2
\`\`\`

### 2. Create Virtual Environment (Recommended)
\`\`\`bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
\`\`\`

### 3. Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 4. Run Jupyter Notebooks
\`\`\`bash
jupyter notebook notebooks/
\`\`\`

## 📊 Progress Status

### ✅ Task 1: Data Analysis & Preprocessing (Completed)
- Data cleaning and validation
- Exploratory Data Analysis (EDA)
- Feature engineering and transformation
- Geolocation integration
- Class imbalance handling with SMOTE

### ✅ Task 2: Model Building & Training (Completed - Interim 2)
- **Baseline Model**: Logistic Regression with class balancing
- **Ensemble Model**: XGBoost with hyperparameter tuning
- **Evaluation Metrics**: AUC-PR, F1-Score, Precision, Recall
- **Cross-Validation**: Stratified K-Fold (5 folds)
- **Model Selection**: XGBoost selected as best performer

### 🔄 Task 3: Model Explainability (Upcoming)
- SHAP analysis for model interpretability
- Feature importance visualization
- Business recommendations derivation

## 📈 Model Performance
Key metrics from Task 2:
- **Logistic Regression**: AUC-PR ≈ 0.78, F1-Score ≈ 0.72
- **XGBoost**: AUC-PR ≈ 0.88, F1-Score ≈ 0.83
- **Best Model**: XGBoost (selected for deployment)

## 🛠️ Usage

### Train Models
\`\`\`python
from src.train import train_xgboost, evaluate_model
from src.preprocess import load_data, preprocess_fraud_data

# Load and preprocess data
df = load_data("data/raw/Fraud_Data.csv")
df_processed = preprocess_fraud_data(df)

# Train model
model = train_xgboost(X_train, y_train)

# Evaluate
results = evaluate_model(model, X_test, y_test, "XGBoost Model")
\`\`\`

### Generate Visualizations
\`\`\`python
from src.evaluate import plot_pr_curve, plot_confusion_matrix

plot_pr_curve(y_test, y_pred_proba, "XGBoost")
plot_confusion_matrix(y_test, y_pred, "XGBoost")
\`\`\`

## 📝 Key Findings (Interim 2)
1. **Class Imbalance**: Successfully handled using SMOTE
2. **Model Performance**: XGBoost outperforms Logistic Regression
3. **Feature Importance**: Transaction patterns and time-based features are key predictors
4. **Business Impact**: Model reduces false positives while maintaining high fraud detection rate

## 👥 Team
- **Name**: [Your Name]
- **Program**: 10 Academy Artificial Intelligence Mastery
- **Tutors**: Kerod, Mahbubah, Filimon

## 📅 Timeline
- **Interim-1 Submission**: 21 Dec 2025 ✓
- **Interim-2 Submission**: 28 Dec 2025 ✓
- **Final Submission**: 30 Dec 2025

## 📚 References
See project document for complete reference list.

---
*Repository created for Interim 2 submission - 10 Academy AI Mastery Program*
