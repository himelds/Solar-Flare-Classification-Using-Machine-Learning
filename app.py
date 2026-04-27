import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Solar Flare Predictor & XAI", layout="wide", initial_sidebar_state="expanded")

st.title("☀️ Solar Flare Classification")
st.markdown("""
This application predicts the class of a solar flare (C, M, or X) based on input characteristics and uses **SHAP (SHapley Additive exPlanations)** to provide a 'Glass Box' view into the AI's decision-making process.
""")

import os

@st.cache_resource
def load_artifacts():
    try:
        if not os.path.exists('artifacts/model.pkl') or not os.path.exists('artifacts/feature_names.pkl'):
            print("Model artifacts not found. Training model automatically on the cloud... This may take a minute.")
            from src.components.data_ingestion import DataIngestion
            from src.components.data_transformation import DataTransformation
            from src.components.model_trainer import ModelTrainer
            
            # Run the pipeline
            ingestion = DataIngestion()
            train_data_path, test_data_path = ingestion.initiate_data_ingestion()
            
            transformation = DataTransformation()
            train_arr, test_arr, feature_names = transformation.initiate_data_transformation(train_data_path, test_data_path)
            
            trainer = ModelTrainer()
            trainer.initiate_model_trainer(train_arr, test_arr)
            
            with open('artifacts/feature_names.pkl', 'wb') as f:
                pickle.dump(feature_names, f)
                
            print("Model trained successfully!")

        with open('artifacts/model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('artifacts/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return model, feature_names
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}. Please ensure the pipeline has been run.")
        return None, None

model, feature_names = load_artifacts()

if model is not None:
    st.sidebar.header("Configure Flare Features")
    
    def user_input_features():
        # Provide sensible default inputs based on typical data values
        intensity = st.sidebar.slider('Intensity', min_value=10, max_value=100, value=20)
        flare_duration = st.sidebar.slider('Flare Duration (mins)', min_value=1, max_value=200, value=15)
        time_to_peak = st.sidebar.slider('Time to Peak (mins)', min_value=1, max_value=100, value=5)
        start_hour = st.sidebar.slider('Start Hour', 0, 23, 12)
        start_dayofweek = st.sidebar.slider('Start Day of Week', 0, 6, 2)
        start_month = st.sidebar.slider('Start Month', 1, 12, 6)
        avg_intensity = st.sidebar.number_input('Historical Region Avg Intensity', value=30.0)
        region = st.sidebar.number_input('Encoded Region', min_value=0, max_value=10000, value=100)
        observatory = st.sidebar.number_input('Encoded Observatory', min_value=0, max_value=20, value=3)

        data = {
            'region': region,
            'intensity': intensity,
            'observatory': observatory,
            'flare_duration': flare_duration,
            'time_to_peak': time_to_peak,
            'start_hour': start_hour,
            'start_dayofweek': start_dayofweek,
            'start_month': start_month,
            'avg_intensity': avg_intensity
        }
        
        # Ensure ordering matches model training
        ordered_data = {fn: data[fn] for fn in feature_names}
        return pd.DataFrame(ordered_data, index=[0])

    input_df = user_input_features()

    st.subheader("Selected Features")
    st.dataframe(input_df)

    if st.button("Predict Class", type="primary"):
        with st.spinner("Analyzing data and generating explanations..."):
            prediction = model.predict(input_df)
            pred_prob = model.predict_proba(input_df)
            
            classes = {0: 'C-Class (Common, minor)', 1: 'M-Class (Moderate, can cause radio blackouts)', 2: 'X-Class (Severe, major disruption)'}
            
            st.success(f"### Predicted: {classes[prediction[0]]}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Probability of C-Class", f"{pred_prob[0][0]*100:.1f}%")
            col2.metric("Probability of M-Class", f"{pred_prob[0][1]*100:.1f}%")
            col3.metric("Probability of X-Class", f"{pred_prob[0][2]*100:.1f}%")
            
            st.divider()
            st.subheader("🔍 Explainable AI (SHAP)")
            st.markdown("The chart below illustrates *why* the model made this prediction. Features pushing the prediction higher are shown in red, and those pushing it lower are in blue.")
            
            # SHAP Explanation
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            
            class_idx = int(prediction[0])
            
            # Since SHAP for single instances is tricky to plot with standard summary_plot,
            # we manually create a beautiful horizontal bar chart for the local explanation.
            # SHAP 0.51.0+ returns an ndarray of shape (n_samples, n_features, n_classes)
            if isinstance(shap_values, list):
                local_shap_values = shap_values[class_idx][0]
            else:
                local_shap_values = shap_values[0, :, class_idx]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#ff0051' if val > 0 else '#008bfb' for val in local_shap_values]
            
            y_pos = np.arange(len(feature_names))
            ax.barh(y_pos, local_shap_values, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_names)
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlabel('SHAP Value (Impact on model output)')
            ax.set_title(f'Feature Contributions for {classes[class_idx].split(" ")[0]} Prediction')
            
            # Add grid for readability
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            
            st.pyplot(fig)
