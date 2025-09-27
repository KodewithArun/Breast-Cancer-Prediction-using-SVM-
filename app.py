import streamlit as st
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Cancer Detection System",
    page_icon="🩺",
    layout="wide"
)

# Clean, professional CSS
st.markdown("""
<style>
    .main-header {
        background-color: #2c3e50;
        color: white;
        padding: 1.5rem;
        border-radius: 5px;
        margin-bottom: 2rem;
    }
    
    .section-header {
        background-color: #ecf0f1;
        padding: 0.8rem 1rem;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    
    .sample-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        margin-bottom: 1.5rem;
    }
    
    .result-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .result-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load('scaler.joblib')
        model = joblib.load('svm_model.joblib')
        return scaler, model
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'scaler.joblib' and 'svm_model.joblib' are available.")
        st.stop()

# Sample data
SAMPLE_DATA = {
    "Malignant Sample 1": [13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766,
                        0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023,
                        15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259],

    "Malignant Sample 2": [12.05, 17.8, 76.38, 446.8, 0.1098, 0.06836, 0.00561, 0.01095, 0.1848, 0.06181,
                        0.2896, 1.02, 1.844, 19.85, 0.005776, 0.01153, 0.001809, 0.004969, 0.02357, 0.001777,
                        13.75, 22.6, 86.73, 587.8, 0.1312, 0.1089, 0.01297, 0.03331, 0.2717, 0.06844],

    "Benign Sample 1": [18.0, 10.39, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                           1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                           25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189],

    "Benign Sample 2": [20.57, 17.77, 132.9, 1326.0, 0.08474, 0.07864, 0.0869, 0.07017, 0.1812, 0.05667,
                           0.5435, 0.7339, 3.398, 74.08, 0.005225, 0.01308, 0.0186, 0.0134, 0.01389, 0.003532,
                           24.99, 23.41, 158.8, 1956.0, 0.1238, 0.1866, 0.2416, 0.186, 0.275, 0.08902]
}

# Feature definitions
FEATURES = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# Header
st.markdown("""
<div class="main-header">
    <h1>Breast Cancer Detection System</h1>
    <p>Machine Learning-based Tumor Classification</p>
</div>
""", unsafe_allow_html=True)

# Load models
scaler, model = load_models()

# Sample Data Section
st.markdown('<div class="section-header"><h3>Sample Data</h3></div>', unsafe_allow_html=True)

st.markdown('<div class="sample-container">', unsafe_allow_html=True)
st.write("Use pre-loaded sample data to test the system:")

col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])

with col1:
    if st.button("Load Benign Sample 1", key="benign1"):
        for i, val in enumerate(SAMPLE_DATA["Benign Sample 1"]):
            st.session_state[f"feature_{i}"] = val
        st.rerun()

with col2:
    if st.button("Load Benign Sample 2", key="benign2"):
        for i, val in enumerate(SAMPLE_DATA["Benign Sample 2"]):
            st.session_state[f"feature_{i}"] = val
        st.rerun()

with col3:
    if st.button("Load Malignant Sample 1", key="malignant1"):
        for i, val in enumerate(SAMPLE_DATA["Malignant Sample 1"]):
            st.session_state[f"feature_{i}"] = val
        st.rerun()

with col4:
    if st.button("Load Malignant Sample 2", key="malignant2"):
        for i, val in enumerate(SAMPLE_DATA["Malignant Sample 2"]):
            st.session_state[f"feature_{i}"] = val
        st.rerun()

with col5:
    if st.button("Clear All", key="clear"):
        for i in range(30):
            st.session_state[f"feature_{i}"] = 0.0
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Input Form Section
st.markdown('<div class="section-header"><h3>Feature Input</h3></div>', unsafe_allow_html=True)

with st.form("feature_form"):
    # Organize features in groups of 10
    st.subheader("Mean Features (1-10)")
    cols1 = st.columns(5)
    user_input = []
    
    for i in range(10):
        col_idx = i % 5
        with cols1[col_idx]:
            val = st.number_input(
                FEATURES[i].replace('_', ' ').title(),
                min_value=0.0,
                format="%.4f",
                key=f"feature_{i}",
                value=st.session_state.get(f"feature_{i}", 0.0)
            )
            user_input.append(val)
    
    st.subheader("Error Features (11-20)")
    cols2 = st.columns(5)
    
    for i in range(10, 20):
        col_idx = (i-10) % 5
        with cols2[col_idx]:
            val = st.number_input(
                FEATURES[i].replace('_', ' ').title(),
                min_value=0.0,
                format="%.4f",
                key=f"feature_{i}",
                value=st.session_state.get(f"feature_{i}", 0.0)
            )
            user_input.append(val)
    
    st.subheader("Worst Features (21-30)")
    cols3 = st.columns(5)
    
    for i in range(20, 30):
        col_idx = (i-20) % 5
        with cols3[col_idx]:
            val = st.number_input(
                FEATURES[i].replace('_', ' ').title(),
                min_value=0.0,
                format="%.4f",
                key=f"feature_{i}",
                value=st.session_state.get(f"feature_{i}", 0.0)
            )
            user_input.append(val)
    
    # Submit button
    st.markdown("---")
    submitted = st.form_submit_button("Analyze Tumor", type="primary")

# Results Section
if submitted:
    st.markdown('<div class="section-header"><h3>Analysis Results</h3></div>', unsafe_allow_html=True)
    
    if all(val > 0 for val in user_input):
        try:
            # Prediction
            input_data = np.array(user_input).reshape(1, -1)
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            
            # Try to get probabilities if available
            try:
                confidence = model.predict_proba(input_scaled)[0]
                has_proba = True
            except AttributeError:
                # SVM not trained with probability=True
                has_proba = False
                confidence = None
            
            # Display results
            if prediction == 0:
                confidence_text = f"<p><strong>Confidence: {confidence[0]*100:.1f}%</strong></p>" if has_proba else ""
                st.markdown(f"""
                <div class="result-success">
                    <h4>Result: BENIGN</h4>
                    <p>The tumor is classified as benign (non-cancerous).</p>
                    {confidence_text}
                </div>
                """, unsafe_allow_html=True)
            else:
                confidence_text = f"<p><strong>Confidence: {confidence[1]*100:.1f}%</strong></p>" if has_proba else ""
                st.markdown(f"""
                <div class="result-danger">
                    <h4>Result: MALIGNANT</h4>
                    <p>The tumor is classified as malignant (cancerous).</p>
                    {confidence_text}
                </div>
                """, unsafe_allow_html=True)
            
            # Show confidence metrics only if probabilities are available
            if has_proba:
                st.subheader("Confidence Scores")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Benign Probability", f"{confidence[0]*100:.2f}%")
                with col2:
                    st.metric("Malignant Probability", f"{confidence[1]*100:.2f}%")
            else:
                st.info("Note: Confidence scores are not available. The model was trained without probability estimation.")
                
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
    else:
        st.warning("Please enter values greater than 0 for all features.")

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <h4>Important Medical Disclaimer</h4>
    <p>This system is designed for research and educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.</p>
</div>
""", unsafe_allow_html=True)