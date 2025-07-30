import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, MultiHeadAttention, LayerNormalization, Add, Flatten, Dropout
import tensorflow as tf

# ✅ تحميل البيانات
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\LINCS_DATA\project\features\pair_feature_matrix_labeled_top_1000_gene.csv")
    df.rename(columns={'drugA_name': 'drugA', 'drugB_name': 'drugB'}, inplace=True)
    return df

# ✅ بناء موديل Tabular Transformer
def build_tabular_transformer(input_dim,
                              num_tokens=40,
                              token_dim=50,
                              num_heads=4,
                              ff_dim=128,
                              num_layers=2,
                              dropout_rate=0.2):
    tf.keras.backend.clear_session()
    inputs = Input(shape=(input_dim,), name="features")
    x = Dense(num_tokens * token_dim, activation="relu")(inputs)
    x = Reshape((num_tokens, token_dim))(x)
    for _ in range(num_layers):
        attn = MultiHeadAttention(num_heads=num_heads, key_dim=token_dim)(x, x)
        x = Add()([x, attn])
        x = LayerNormalization()(x)
        ff = Dense(ff_dim, activation="relu")(x)
        ff = Dense(token_dim)(ff)
        x = Add()([x, ff])
        x = LayerNormalization()(x)
        x = Dropout(dropout_rate)(x)
    x = Flatten()(x)
    outputs = Dense(1, activation="sigmoid", name="synergy")(x)
    model = Model(inputs, outputs, name="TabularTransformer_Top1000")
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name='auroc')])
    return model

# ✅ تحميل الموديل مع كاش آمن
@st.cache_resource
def load_model(input_dim):
    model = build_tabular_transformer(input_dim)
    model.load_weights(r"C:\LINCS_DATA\project\interface\TabularTransformer_Top1000.h5")
    return model

# ✅ تحميل الداتا والموديل
df = load_data()
feature_cols = [col for col in df.columns if col.startswith("gene")]
input_dim = len(feature_cols)
model = load_model(input_dim)
all_drugs = sorted(pd.unique(df[['drugA', 'drugB']].values.ravel('K')))

# ✅ واجهة Streamlit
st.title("🧪 Drug Interaction Dashboard")
tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ Predict", "2️⃣ Recommend", "3️⃣ Explain", "4️⃣ Heatmap"])

# Tab 1: Predict Drug-Drug Interaction
with tab1:
    st.subheader("🔍 Predict Drug-Drug Interaction")
    drugA = st.selectbox("Drug A", all_drugs, key="drugA")
    drugB = st.selectbox("Drug B", all_drugs, key="drugB")
    if drugA != drugB:
        row = df[((df["drugA"] == drugA) & (df["drugB"] == drugB)) |
                 ((df["drugA"] == drugB) & (df["drugB"] == drugA))]
        if not row.empty:
            X = row[feature_cols].values
            score = model.predict(X).ravel()[0]
            label = int(score > 0.5)
            st.success(f"Predicted Score: {score:.3f}")
            st.info(f"Predicted Label: {'Synergy' if label == 1 else 'No Synergy'}")
        else:
            st.warning("⚠️ This pair not in dataset.")
    else:
        st.warning("⚠️ Please select two different drugs.")

# Tab 2: Recommend Compatible Drugs
with tab2:
    st.subheader("💊 Recommend Compatible Drugs")
    drugA_rec = st.selectbox("Select Drug", all_drugs, key="rec_drug")
    rec_df = df[(df["drugA"] == drugA_rec) | (df["drugB"] == drugA_rec)].copy()
    if not rec_df.empty:
        X = rec_df[feature_cols].values
        rec_df["score"] = model.predict(X).ravel()
        rec_df["label"] = (rec_df["score"] > 0.5).astype(int)
        st.dataframe(rec_df[["drugA", "drugB", "score", "label"]].sort_values(by="score", ascending=False).reset_index(drop=True))
    else:
        st.warning("No recommendations found.")

# Tab 3: GPT Explanation
with tab3:
    st.subheader("🧠 Explanation (Simple Rule-Based)")
    drug1 = st.selectbox("Drug A", all_drugs, key="gptA")
    drug2 = st.selectbox("Drug B", all_drugs, key="gptB")
    score = st.slider("Predicted Score", 0.0, 1.0, 0.5, 0.01)
    label = st.selectbox("Predicted Label", [0, 1])
    
    if st.button("Explain"):
        # تفسير مبني على الشروط
        if label == 1 and score > 0.7:
            explanation = f"Strong positive synergy is predicted between {drug1} and {drug2}. This may suggest a complementary or potentiating effect at the gene expression level."
        elif label == 1:
            explanation = f"Possible positive interaction between {drug1} and {drug2}. Gene expression signatures indicate some synergy."
        elif score < 0.3:
            explanation = f"No synergy is expected between {drug1} and {drug2}. Their gene profiles suggest minimal or no interaction."
        else:
            explanation = f"The predicted interaction between {drug1} and {drug2} is weak or inconclusive based on gene expression."
        
        st.success(explanation)


# Tab 4: Heatmap
with tab4:
    st.subheader("🧬 Gene Heatmap for Drug Pair")
    d1 = st.selectbox("Drug A", all_drugs, key="hmA")
    d2 = st.selectbox("Drug B", all_drugs, key="hmB")
    row = df[((df["drugA"] == d1) & (df["drugB"] == d2)) |
             ((df["drugA"] == d2) & (df["drugB"] == d1))]
    if not row.empty:
        genes = row[feature_cols].values.flatten()
        fig, ax = plt.subplots(figsize=(12, 1))
        sns.heatmap([genes], cmap="coolwarm", cbar=True, ax=ax)
        ax.set_yticklabels([f"{d1} vs {d2}"])
        ax.set_xticks([])
        st.pyplot(fig)
    else:
        st.warning("No gene data found for selected pair.")
