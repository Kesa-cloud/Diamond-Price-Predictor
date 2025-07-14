import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

# ---------------------------------------------
# Page configuration
st.set_page_config(
    page_title="💎 Diamond Price Predictor",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------
# Custom CSS for creative styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Montserrat', sans-serif;
    background-image: url('https://images.unsplash.com/photo-1589330694655-d1256a5b6de8');
    background-size: cover;
    background-position: center;
    color: #333333;
}

h1, h2, h3 {
    background: linear-gradient(to right, #5A189A, #F72585);
    -webkit-background-clip: text;
    color: transparent;
}

.stButton>button {
    background-color: #5A189A;
    color: white;
    border-radius: 8px;
    padding: 10px 24px;
    font-size: 16px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
}
.stButton>button:hover {
    background-color: #7B2CBF;
}
.price-display {
    font-size: 36px;
    font-weight: bold;
    color: #5A189A;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------
# App Title
st.title("💎 Diamond Price Predictor")
st.markdown("Predict the price of your diamond based on its characteristics. Adjust the parameters below and click Predict Price to see the estimated value.")

# ---------------------------------------------
# Load and train model
@st.cache_data
def load_data_and_train_model():
    data_path = "C:\\Users\\paulg\\Downloads\\archive (1)\diamonds.csv"
    df = pd.read_csv(data_path).drop(columns=['Unnamed: 0'], errors='ignore')
    df = df.dropna()

    label_encoders = {}
    categorical_cols = ['cut', 'color', 'clarity']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop('price', axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, label_encoders, df, mse, r2

model, label_encoders, df, mse, r2 = load_data_and_train_model()

# ---------------------------------------------
# Sidebar stats
st.sidebar.header("Model Performance")
st.sidebar.write(f"Mean Squared Error: {mse:.2f}")
st.sidebar.write(f"R² Score: {r2:.2f}")
st.sidebar.write("Dataset Info:")
st.sidebar.write(f"Total samples: {len(df)}")
st.sidebar.write(f"Features: {len(df.columns) - 1}")

# ---------------------------------------------
# Input widgets
col1, col2, col3 = st.columns(3)

with col1:
    carat = st.slider("Carat Weight", float(df['carat'].min()), float(df['carat'].max()), 1.0, 0.01)
    cut = st.selectbox("Cut Quality", ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
    color = st.selectbox("Color Grade", ['D', 'E', 'F', 'G', 'H', 'I', 'J'])

with col2:
    clarity = st.selectbox("Clarity", ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
    depth = st.slider("Depth (%)", float(df['depth'].min()), float(df['depth'].max()), 60.0, 0.1)
    table = st.slider("Table (%)", float(df['table'].min()), float(df['table'].max()), 55.0, 0.1)

with col3:
    x = st.slider("Length (x)", float(df['x'].min()), float(df['x'].max()), 5.0, 0.1)
    y = st.slider("Width (y)", float(df['y'].min()), float(df['y'].max()), 5.0, 0.1)
    z = st.slider("Height (z)", float(df['z'].min()), float(df['z'].max()), 3.0, 0.1)

# ---------------------------------------------
# Prediction
if st.button("Predict Price"):
    input_data = pd.DataFrame({
        'carat': [carat],
        'cut': [cut],
        'color': [color],
        'clarity': [clarity],
        'depth': [depth],
        'table': [table],
        'x': [x],
        'y': [y],
        'z': [z]
    })

    input_data['cut'] = label_encoders['cut'].transform(input_data['cut'])
    input_data['color'] = label_encoders['color'].transform(input_data['color'])
    input_data['clarity'] = label_encoders['clarity'].transform(input_data['clarity'])

    prediction = model.predict(input_data)

    st.balloons()
    st.markdown(f"""
    <div class="prediction-result">
        <h3>Predicted Diamond Price</h3>
        <div class="price-display">${prediction[0]:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Input Parameters")
    st.json({
        "Carat": carat,
        "Cut": cut,
        "Color": color,
        "Clarity": clarity,
        "Depth": f"{depth}%",
        "Table": f"{table}%",
        "Dimensions": f"{x} × {y} × {z} mm"
    })

    # ✅ Display model metrics after prediction
    st.markdown(f"""
    <div style="margin-top:40px;">
        <h3>Model Summary</h3>
        <ul>
            <li><strong>Total samples:</strong> {len(df)}</li>
            <li><strong>Mean Squared Error:</strong> {mse:.2f}</li>
            <li><strong>R² Score:</strong> {r2:.2f}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)