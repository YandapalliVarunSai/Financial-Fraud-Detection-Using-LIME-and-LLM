# 💰 Financial Fraud Detection Using LIME and LLM

> **Stacked Machine Learning** meets **Explainable AI (LIME)** and **Large Language Models (LLM)** to build an **interpretable, high-accuracy financial fraud detection system**.  
> Designed for real-world banking transactions — blending performance, explainability, and trust.

🔗 **GitHub Repository:** [YandapalliVarunSai/Financial-Fraud-Detection-Using-LIME-and-LLM](https://github.com/YandapalliVarunSai/Financial-Fraud-Detection-Using-LIME-and-LLM)

---

## 🚀 Project Overview

This project develops an **intelligent, explainable, and accurate financial fraud detection system** using **Stacked Machine Learning**, **LIME (Explainable AI)**, and **Sonar Pro LLM** for natural-language interpretation.

It aims to:
- Detect fraudulent transactions with **91% accuracy**
- Explain **why** a prediction was made (via LIME)
- Generate **human-readable insights** from LIME using LLM

The system enhances **transparency**, **trust**, and **regulatory compliance** — bridging the gap between machine predictions and human understanding.

---

## 🧠 Architecture Overview

```
Raw Transaction Data (Kaggle)
        │
        ▼
 Data Preprocessing
 (Encoding, SMOTE, Scaling)
        │
        ▼
 Feature Selection
 (Random Forest Top Features)
        │
        ▼
 Model Training
 (LR, DT, RF, AdaBoost, XGBoost)
        │
        ▼
 Stacking Ensemble
 (Meta-learner: XGBoost)
        │
        ▼
 Explainability Layer (LIME)
        │
        ▼
 LLM Integration (Sonar Pro)
 → Converts LIME output into human-readable text
```

---

## ⚙️ Tools & Technologies

| Layer | Tool / Library | Purpose |
|-------|----------------|----------|
| **Language** | Python | Core implementation |
| **Data Processing** | pandas, numpy | Cleaning, encoding, scaling |
| **Modeling** | scikit-learn, XGBoost | Classification & stacking ensemble |
| **Resampling** | SMOTE (imblearn) | Handle class imbalance |
| **Explainability** | LIME | Feature-level interpretability |
| **AI Insights** | Sonar Pro (LLM) | Natural-language explanation of predictions |
| **Visualization** | matplotlib, seaborn | Charts & confusion matrices |
| **Evaluation** | sklearn.metrics | Accuracy, Precision, Recall, F1-score |
| **IDE** | Jupyter Notebook | Interactive analysis |

---

## 🧱 Model Architecture & Components

### ⚡ 1️⃣ Stacked Ensemble Learning
Combines multiple base learners:
- Logistic Regression  
- Decision Tree  
- Random Forest  
- AdaBoost  
- XGBoost  
**Meta Learner:** XGBoost  

✅ Achieved **91% accuracy**, outperforming individual models.

---

### 🧩 2️⃣ Explainable AI (LIME)
- Provides **local explanations** for each prediction.  
- Identifies which features (e.g., *City*, *State*, *Transaction_Device*) contributed to a decision.  
- Enables **model transparency** for auditors & analysts.

---

### 🧠 3️⃣ Large Language Model (Sonar Pro Integration)
- Converts LIME outputs into **clear natural-language narratives**.  
- Example:  
  > “The transaction was flagged as fraud due to unusual city and device patterns, despite a safe branch type.”

This boosts **trust, interpretability, and compliance-readiness**.

---

## 📊 Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|------------|----------|-----------|
| Logistic Regression | 64% | 0.61 | 0.63 | 0.62 |
| Random Forest | 73% | 0.72 | 0.73 | 0.72 |
| Decision Tree | 84% | 0.85 | 0.84 | 0.84 |
| AdaBoost | 77% | 0.78 | 0.76 | 0.77 |
| XGBoost | 88% | 0.87 | 0.88 | 0.88 |
| 🥇 **Stacking Classifier** | **91%** | **0.90** | **0.92** | **0.91** |

---

## 📦 Repository Structure

```
Financial-Fraud-Detection-Using-LIME-and-LLM/
│
├── data/
│   └── bank_fraud_transactions.csv
│
├── notebooks/
│   ├── 1_preprocessing_visualization.ipynb
│   ├── 2_model_training.ipynb
│   ├── 3_lime_explainability.ipynb
│   └── 4_llm_integration.ipynb
│
├── models/
│   ├── stacking_model.pkl
│   ├── xgboost_model.pkl
│   └── random_forest_importance.csv
│
├── results/
│   ├── confusion_matrices.png
│   ├── lime_explanations.png
│   └── performance_summary.csv
│
├── README.md
└── requirements.txt
```

---

## 🧰 Setup Instructions

### 1️⃣ Clone Repository
```bash
git clone https://github.com/YandapalliVarunSai/Financial-Fraud-Detection-Using-LIME-and-LLM.git
cd Financial-Fraud-Detection-Using-LIME-and-LLM
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run Jupyter Notebook
```bash
jupyter notebook
```

Then open the `.ipynb` files inside the `notebooks/` folder.

---

## 📊 Key Visualizations

| Visualization | Description |
|----------------|-------------|
| Confusion Matrices | Model classification performance |
| Feature Importance | Top 15 predictors via Random Forest |
| LIME Charts | Local feature impact visualization |
| Accuracy Table | Model performance comparison |
| Natural Language Insight | LLM summary of predictions |

---

## 🧩 Key Learnings

- Ensemble methods improve **robustness** and **accuracy**.  
- **SMOTE** effectively handles severe class imbalance.  
- **LIME** adds interpretability at instance-level.  
- **LLM** integration bridges the gap between **technical and business insights**.  
- Transparency and explainability increase **regulatory trust**.

---

## 🔮 Future Work

- [ ] Extend dataset across multiple banks  
- [ ] Integrate **deep learning (LSTM/Transformers)** for sequence modeling  
- [ ] Implement **real-time fraud detection APIs**  
- [ ] Explore **global explainability** (SHAP)  
- [ ] Deploy via **Streamlit or Flask dashboard**

---

## 👤 Author

**Varun Sai Yandapalli**  
