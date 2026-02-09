import streamlit as st
import joblib
import numpy as np
from Database import save_prediction, get_all_predictions, delete_all_predictions

# Load model & encoders
model = joblib.load("Disease_model.joblib")
mlb = joblib.load("Symptom_encoder.joblib")
disease_encoder = joblib.load("Disease_encoder.joblib")

all_symptoms = list(mlb.classes_)

# Page config
st.set_page_config(
    page_title="AI Disease Prediction System",
    page_icon="🩺",
    layout="centered"
)

# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "selected_symptoms" not in st.session_state:
    st.session_state.selected_symptoms = []

# Login Page
def login_page():
    st.markdown(
        "<h1 style='text-align:center;'>🔐 Login</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center;color:gray;'>AI Disease Prediction System</p>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")

    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("➡️ Login", use_container_width=True):
            if username == "admin" and password == "admin":
                st.session_state.logged_in = True
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid username or password")

# Main App
def main_app():

    # -------- Sidebar --------
    with st.sidebar:
        st.title("📋 Project Info")
        st.markdown("""
        **AI Disease Prediction System**

        **Tech Used**
        - Python  
        - Scikit-learn  
        - Streamlit  
        - SQLite  

        **Model**
        - Random Forest  

        **Developer**
        Rohit Garg  
        Punjabi University, Patiala
        """)

        st.markdown("---")

        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.rerun()

    # -------- Header --------
    st.markdown(
        "<h1 style='text-align:center;'>🩺 AI Disease Prediction System</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center;color:gray;'>Select symptoms and get instant disease prediction</p>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # -------- Input Section --------
    st.subheader("🤒 Symptom Selection")

    st.session_state.selected_symptoms = st.multiselect(
        "Search & select symptoms",
        all_symptoms,
        default=st.session_state.selected_symptoms,
        placeholder="Type symptoms like fever, headache..."
    )

    col1, col2 = st.columns(2)

    with col1:
        predict_btn = st.button("🔮 Predict", use_container_width=True)

    with col2:
        reset_btn = st.button("🔄 Reset", use_container_width=True)

    if reset_btn:
        st.session_state.selected_symptoms = []
        st.rerun()

    # -------- Prediction --------
    if predict_btn:
        if len(st.session_state.selected_symptoms) < 3:
            st.warning("⚠ Please select at least one symptom.")
        else:
            input_vector = mlb.transform([st.session_state.selected_symptoms])
            pred = model.predict(input_vector)[0]
            disease = disease_encoder.inverse_transform([pred])[0]

            probs = model.predict_proba(input_vector)[0]
            confidence = probs[pred] * 100
            top3_idx = np.argsort(probs)[-3:][::-1]

            st.markdown("---")
            st.subheader("✅ Prediction Result")
            st.success(disease)
            st.info(f"🔎 Confidence: {confidence:.2f}%")

            st.markdown("### 📊 Top 3 Possible Diseases")
            for i in top3_idx:
                st.write(
                    f"- **{disease_encoder.inverse_transform([i])[0]}** : {probs[i]*100:.2f}%"
                )

            save_prediction(
                age=0,
                gender="NA",
                symptoms=st.session_state.selected_symptoms,
                disease=disease
            )

            st.markdown("---")
            st.subheader("💡 General Advice")
            st.success("Drink water 💧, take proper rest 🛏️ and consult a doctor if symptoms persist.")

            st.markdown(
                f"[🔍 Learn more about {disease}](https://www.google.com/search?q={disease.replace(' ', '+')}+treatment)"
            )

    # -------- History --------
    st.markdown("---")
    st.subheader("📜 Prediction History")

    if st.checkbox("Show history"):
        history = get_all_predictions()

        if history:
            for row in history:
                st.markdown(f"""
**🧾 ID:** {row[0]}  
**🤒 Symptoms:** {row[3]}  
**🧬 Disease:** {row[4]}  
**🕒 Time:** {row[5]}
""")
                st.markdown("---")

            if st.button("🗑 Clear History"):
                delete_all_predictions()
                st.success("History cleared")
                st.rerun()
        else:
            st.info("No history available.")

    # -------- Footer --------
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;'>Made with ❤️ by <b>Rohit Garg</b></div>",
        unsafe_allow_html=True
    )

# Controller
if not st.session_state.logged_in:
    login_page()
else:
    main_app()
