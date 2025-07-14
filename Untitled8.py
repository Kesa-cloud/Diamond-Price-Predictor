import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

# Set page config with diamond theme
st.set_page_config(
    page_title="Diamond Price Predictor",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for diamond theme
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-result {
        background-color: #e8f4f8;
        border-left: 5px solid #2196F3;
        padding: 20px;
        margin: 20px 0;
        border-radius: 5px;
    }
    .price-display {
        font-size: 32px;
        font-weight: bold;
        color: #2196F3;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# App title and description
st.title("💎 Diamond Price Predictor")
st.markdown("""
Predict the price of your diamond based on its characteristics.  
Adjust the parameters below and click **Predict Price** to see the estimated value.
""")

# Load data and train model
@st.cache_data
def load_data_and_train_model():
    # Load the diamonds dataset
    data_path = "C:/Users/samgi/OneDrive/Documents/diamonds.csv"
    df = pd.read_csv(data_path)
    
    # Data preprocessing
    # Remove any rows with missing values
    df = df.dropna()
    
    # Convert categorical variables to numerical using LabelEncoder
    label_encoders = {}
    categorical_cols = ['cut', 'color', 'clarity']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Prepare features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, label_encoders, df, mse, r2

# Load the model and data
model, label_encoders, df, mse, r2 = load_data_and_train_model()

# Display model performance in sidebar
st.sidebar.header("Model Performance")
st.sidebar.write(f"Mean Squared Error: {mse:.2f}")
st.sidebar.write(f"R² Score: {r2:.2f}")
st.sidebar.write("Dataset Info:")
st.sidebar.write(f"Total samples: {len(df)}")
st.sidebar.write(f"Features: {len(df.columns) - 1}")

# --- Input Fields ---
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

# --- Prediction Logic ---
if st.button("Predict Price"):
    # Prepare input data
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
    
    # Apply label encoding to categorical variables
    input_data['cut'] = label_encoders['cut'].transform(input_data['cut'])
    input_data['color'] = label_encoders['color'].transform(input_data['color'])
    input_data['clarity'] = label_encoders['clarity'].transform(input_data['clarity'])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display results
    st.balloons()
    st.markdown(f"""
    <div class="prediction-result">
        <h3>Predicted Diamond Price</h3>
        <div class="price-display">${prediction[0]:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show the input values
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
