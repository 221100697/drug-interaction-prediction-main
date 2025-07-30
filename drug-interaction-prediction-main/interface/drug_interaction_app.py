#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[5]:


# Load model and data
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\LINCS_DATA\project\features\pair_feature_matrix_labeled_top_1000_gene.csv")
    df.rename(columns={'drugA_name': 'drugA', 'drugB_name': 'drugB'}, inplace=True)
    return df

df = load_data()
feature_cols = [col for col in df.columns if col.startswith("gene")]
model = joblib.load(r"C:\LINCS_DATA\project\features\tabtrans_model.pkl")
all_drugs = sorted(pd.unique(df[['drugA', 'drugB']].values.ravel('K')))

# UI layout
st.title("🧪 Drug Interaction Dashboard")
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Predict Synergy", 
    "2️⃣ Recommend Drugs", 
    "3️⃣ Explain Prediction (GPT)", 
    "4️⃣ Heatmap"
])


# In[6]:


# Tab 1: Predict Synergy
with tab1:
    st.subheader("🔍 Predict Interaction Between Two Drugs")
    drugA = st.selectbox("Choose Drug A", all_drugs, key="drugA_predict")
    drugB = st.selectbox("Choose Drug B", all_drugs, key="drugB_predict")

    if drugA != drugB:
        row = df[((df["drugA"] == drugA) & (df["drugB"] == drugB)) |
                 ((df["drugA"] == drugB) & (df["drugB"] == drugA))]

        if not row.empty:
            X = row[feature_cols].values
            score = model.predict(X).ravel()[0]
            label = int(score > 0.5)
            st.success(f"Predicted Score: {score:.3f}")
            st.info(f"Predicted Label: {'Synergy (1)' if label == 1 else 'No Synergy (0)'}")
        else:
            st.warning("⚠️ This drug pair is not found in the dataset.")
    else:
        st.warning("⚠️ Please select two different drugs.")


# In[7]:


# Tab 2: Recommend Drugs
with tab2:
    st.subheader("💊 Recommend Compatible Drugs")
    drugA_rec = st.selectbox("Choose Drug A", all_drugs, key="drugA_recommend")
    rec_df = df[(df["drugA"] == drugA_rec) | (df["drugB"] == drugA_rec)].copy()
    if not rec_df.empty:
        X = rec_df[feature_cols].values
        scores = model.predict(X).ravel()
        rec_df["predicted_score"] = scores
        rec_df["predicted_label"] = (scores > 0.5).astype(int)
        rec_df_sorted = rec_df.sort_values(by="predicted_score", ascending=False)
        st.dataframe(rec_df_sorted[["drugA", "drugB", "predicted_score", "predicted_label"]].reset_index(drop=True))
    else:
        st.warning("No compatible drugs found in dataset.")


# In[8]:


# Tab 3: GPT Explanation (Mocked)
with tab3:
    st.subheader("🧠 Explain Prediction Using GPT")
    drugA_exp = st.selectbox("Drug A", all_drugs, key="drugA_exp")
    drugB_exp = st.selectbox("Drug B", all_drugs, key="drugB_exp")
    score_input = st.number_input("Predicted Score", min_value=0.0, max_value=1.0, step=0.01)
    label_input = st.selectbox("Predicted Label", [0, 1])

    if st.button("Generate Explanation"):
        explanation = f"Based on the predicted synergy score of {score_input:.2f} between {drugA_exp} and {drugB_exp}, and the predicted label {label_input}, this combination may {'enhance therapeutic effect' if label_input == 1 else 'lack synergy'}."
        st.info(explanation)


# In[10]:


# Tab 4: Heatmap
with tab4:
    st.subheader("🧬 Gene Feature Heatmap for Drug Pair")
    drugA_hm = st.selectbox("Drug A", all_drugs, key="drugA_hm")
    drugB_hm = st.selectbox("Drug B", all_drugs, key="drugB_hm")

    row = df[((df["drugA"] == drugA_hm) & (df["drugB"] == drugB_hm)) |
             ((df["drugA"] == drugB_hm) & (df["drugB"] == drugA_hm))]

    if not row.empty:
        genes = row[feature_cols].values.flatten()
        fig, ax = plt.subplots(figsize=(12, 1))
        sns.heatmap([genes], cmap="coolwarm", cbar=True, ax=ax)
        ax.set_yticklabels([f"{drugA_hm} vs {drugB_hm}"])
        ax.set_xticks([])
        st.pyplot(fig)
    else:
        st.warning("No gene data available for this pair.")

