import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="🧠 Customer Segment Predictor", layout="wide")
st.title("🧠 Customer Segment Predictor using KMeans")

# ----------------------------
# Strategy function (MOVE IT UP)
# ----------------------------
def strategy_for_cluster(cluster_id):
    return {
        0: "🎁 Send discount offers and loyalty rewards.",
        1: "💼 Upsell luxury or premium services.",
        2: "📣 Offer exclusive bundles and perks.",
        3: "🧪 A/B test offers and monitor response.",
        4: "📍 Focus on brand awareness and trust."
    }.get(cluster_id, "No strategy available.")

# ----------------------------
# Load data or ask for upload
# ----------------------------
csv_path = "Mall_Customers.csv"

if not os.path.exists(csv_path):
    st.warning("⚠️ 'Mall_Customers.csv' not found. Please upload a dataset.")
    uploaded_file = st.file_uploader("Upload your customer CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Uploaded dataset loaded.")
        df.to_csv(csv_path, index=False)
    else:
        st.stop()
else:
    df = pd.read_csv(csv_path)
    st.success("✅ Dataset loaded successfully.")

# ----------------------------
# Preprocess data
# ----------------------------
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Handle missing data
df.fillna({
    'Age': df['Age'].mean(),
    'Gender': 0,
    'Annual Income (k$)': df['Annual Income (k$)'].mean(),
    'Spending Score (1-100)': df['Spending Score (1-100)'].mean()
}, inplace=True)

features = ['Age', 'Gender', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KMeans model
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# ----------------------------
# Sidebar: New customer input
# ----------------------------
st.sidebar.header("➕ Enter New Customer Details")
name = st.sidebar.text_input("Customer Name (Optional)")
age = st.sidebar.slider("Age", 15, 70, 30)
gender = st.sidebar.radio("Gender", ["Male", "Female"])
income = st.sidebar.slider("Annual Income (k$)", 10, 150, 60)
score = st.sidebar.slider("Spending Score (1-100)", 1, 100, 50)

# Predict and optionally add new customer
if st.sidebar.button("Predict Segment & Save"):
    gender_num = 0 if gender == "Male" else 1
    new_customer = np.array([[age, gender_num, income, score]])
    new_customer_scaled = scaler.transform(new_customer)
    predicted_cluster = kmeans.predict(new_customer_scaled)[0]

    # Append to dataset
    new_row = pd.DataFrame([[name, age, gender_num, income, score, predicted_cluster]],
                           columns=['Customer Name'] + features + ['Cluster'])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(csv_path, index=False)

    st.success(f"🎯 Predicted Segment: Cluster {predicted_cluster}")
    st.markdown(f"💡 Strategy: **{strategy_for_cluster(predicted_cluster)}**")
else:
    predicted_cluster = None

# ----------------------------
# Cluster summary
# ----------------------------
st.subheader("📊 Cluster Profiles")
summary = df.groupby('Cluster')[features].mean().reset_index()
st.dataframe(summary.style.highlight_max(axis=0), height=300)

# ----------------------------
# Visualize clusters
# ----------------------------
st.subheader("📍 Visualizing Customer Segments")
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Cluster', data=df, palette='Set2', s=80)
if predicted_cluster is not None:
    plt.scatter(income, score, c='black', s=200, marker='X', label='New Customer')
plt.legend()
st.pyplot(fig)

# ----------------------------
# Download button
# ----------------------------
st.download_button("⬇️ Download Updated Dataset", df.to_csv(index=False), file_name="Mall_Customers_Updated.csv")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Built with ❤️ by Group 10")
