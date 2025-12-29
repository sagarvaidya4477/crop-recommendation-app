import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Smart Crop Recommendation",
    page_icon="🌾",
    layout="wide"
)

# =========================
# CUSTOM CSS (🔥 LOOK & FEEL)
# =========================
st.markdown("""
<style>
.main {
    background-color: #f6fff8;
}
h1 {
    color: #2d6a4f;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(135deg,#95d5b2,#d8f3dc);
    box-shadow: 0px 8px 20px rgba(0,0,0,0.15);
    margin-bottom: 15px;
}
.result {
    font-size: 22px;
    font-weight: bold;
}
.footer {
    text-align: center;
    color: gray;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="card">
<h1>🌱 Smart Crop Recommendation System</h1>
<p>AI powered dashboard to recommend the <b>best crop</b> using soil & climate data</p>
</div>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("crop_recommendation(2).csv")

data = load_data()

# =========================
# FEATURES & TARGET
# =========================
features = ['temperature', 'humidity', 'rainfall', 'ph']
X = data[features]

le = LabelEncoder()
y = le.fit_transform(data['label'])

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# MODELS
# =========================
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(kernel='rbf', probability=True)
lin = LinearRegression()

knn.fit(X_train, y_train)
svm.fit(X_train, y_train)
lin.fit(X_train, y_train)

acc_knn = accuracy_score(y_test, knn.predict(X_test))
acc_svm = accuracy_score(y_test, svm.predict(X_test))
acc_lin = accuracy_score(y_test, np.round(lin.predict(X_test)).astype(int))

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.title("🌾 Enter Field Data")

temperature = st.sidebar.slider("🌡 Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.sidebar.slider("💧 Humidity (%)", 0.0, 100.0, 80.0)
rainfall = st.sidebar.slider("🌧 Rainfall (mm)", 0.0, 300.0, 200.0)
ph = st.sidebar.slider("🧪 Soil pH", 3.0, 10.0, 6.5)

input_data = np.array([[temperature, humidity, rainfall, ph]])
input_scaled = scaler.transform(input_data)

# =========================
# PREDICTION SECTION
# =========================
st.subheader("🔮 Prediction Results")

if st.button("🚜 Predict Best Crop"):
    crop_knn = le.inverse_transform(knn.predict(input_scaled))[0]
    crop_svm = le.inverse_transform(svm.predict(input_scaled))[0]
    crop_lin = le.inverse_transform([int(round(lin.predict(input_scaled)[0]))])[0]

    confidence = np.max(svm.predict_proba(input_scaled)) * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="card">
        <div class="result">🌾 KNN</div>
        <h2>{crop_knn}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
        <div class="result">🤖 SVM (Best)</div>
        <h2>{crop_svm}</h2>
        <p>Confidence: {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card">
        <div class="result">📉 Linear (Demo)</div>
        <h2>{crop_lin}</h2>
        </div>
        """, unsafe_allow_html=True)

# =========================
# MODEL PERFORMANCE
# =========================
st.subheader("📊 Model Performance Comparison")

acc_df = pd.DataFrame({
    "Model": ["Linear Regression", "KNN", "SVM"],
    "Accuracy": [acc_lin, acc_knn, acc_svm]
})

st.bar_chart(acc_df.set_index("Model"))

# =========================
# FOOTER
# =========================
st.markdown("""

""", unsafe_allow_html=True)


