# 💊 Drug-Drug Interaction Prediction using Gene Expression

This project presents a complete pipeline to predict synergistic drug-drug interactions (DDIs) using **transcriptomic gene expression data** and **machine learning / deep learning models**.

---

## 📁 Project Structure

```
project/
├── data/               # Raw and processed CSV datasets (Z-score, PAS, labels)
├── features/           # Feature matrices and model inputs
├── models/             # Model training notebooks (MLP, XGBoost, Tabular Transformer)
├── interface/          # Streamlit dashboard + model files
├── preprocessing/      # Feature engineering and signature filtering
├── report/             # Final PDF report and supporting documents
```

---

## 🧠 Models Used

- ✅ Random Forest
- ✅ XGBoost
- ✅ Multilayer Perceptron (MLP)
- ✅ Tabular Transformer (Top 1000 gene features)

---

## 🔬 Data Sources

- **LINCS L1000 (GSE92742)**: Drug-induced Z-score gene expression profiles.
- **SYNERGxDB**: Drug combination synergy scores and pairings.
- **Pathway Annotations**: MSigDB Hallmark Pathways used for PAS feature extraction.
- [LINCS L1000 - GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742)
- [SYNERGxDB](https://doi.org/10.5281/zenodo.3823346)

---

## 🚀 Dashboard Features

Built using **Streamlit**, the interface includes:
- **Predict Tab**: Predict synergy for any two drugs.
- **Recommend Tab**: Suggest best synergistic partners.
- **Explain Tab**: GPT-powered explanation of predictions.

---

## 📦 Requirements

Install required packages using:

```bash
pip install -r requirements.txt
```
⚠️ Missing Files (Not Included in This Repo)
Some large files were excluded from version control (GitHub) due to size limits. Make sure to download or generate them before running the project:

features/pair_feature_matrix_labeled_full.csv — Full feature matrix (Top 1000 or all genes)

interface/tabtrans_model.h5 — Trained Tabular Transformer model

models/tabtrans_model.pkl — Pickled version of trained model

report/Graduation-Project-Report finalll.pdf — Full thesis report

These files are listed in .gitignore to avoid upload issues.
---

## 👩‍🔬 Authors

- Reem Ramadan
- Yomna Refaat
- Mariam Mostafa
- Nayra Elgazzaz
---
