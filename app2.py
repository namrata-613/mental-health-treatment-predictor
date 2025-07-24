import streamlit as st
import pandas as pd
import pickle

# --- Load Model and Encoders ---
try:
    with open('best_lgbm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('training_columns.pkl', 'rb') as f:
        training_columns = pickle.load(f)
except FileNotFoundError:
    st.error("Model or encoder files not found. Please run the analysis script first to generate them.")
    st.stop()

# --- Streamlit App Interface ---
st.set_page_config(page_title="Mental Health Treatment Prediction", layout="wide")
st.title("🧠 Mental Health Treatment Prediction")
st.write("This app predicts whether an individual in the tech workplace is likely to seek treatment for a mental health condition. Fill out the form below to get a prediction.")

# --- Input Form ---
st.header("Enter Your Information")
with st.form(key='prediction_form'):
    # Create two columns for the input fields
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", encoders['Gender'].classes_)
        occupation = st.selectbox("Occupation", encoders['Occupation'].classes_)
        self_employed = st.selectbox("Are you self-employed?", encoders['self_employed'].classes_)
        family_history = st.selectbox("Do you have a family history of mental illness?", encoders['family_history'].classes_)
        days_indoors = st.selectbox("How many days have you been indoors due to mental health?", encoders['Days_Indoors'].classes_)
        growing_stress = st.selectbox("Are you experiencing growing stress?", encoders['Growing_Stress'].classes_)
        changes_habits = st.selectbox("Have you experienced changes in your habits?", encoders['Changes_Habits'].classes_)

    with col2:
        mental_health_history = st.selectbox("Do you have a personal mental health history?", encoders['Mental_Health_History'].classes_)
        mood_swings = st.selectbox("Do you experience mood swings?", encoders['Mood_Swings'].classes_)
        coping_struggles = st.selectbox("Are you struggling to cope?", encoders['Coping_Struggles'].classes_)
        work_interest = st.selectbox("Have you lost interest in your work?", encoders['Work_Interest'].classes_)
        social_weakness = st.selectbox("Are you experiencing social weakness?", encoders['Social_Weakness'].classes_)
        mental_health_interview = st.selectbox("Would you discuss a mental health issue with a potential employer?", encoders['mental_health_interview'].classes_)
        care_options = st.selectbox("Are you aware of care options for mental health?", encoders['care_options'].classes_)

    submit_button = st.form_submit_button(label='Get Prediction')

# --- Prediction Logic ---
if submit_button:
    # Create a dictionary from the user's input
    custom_data = {
        'Gender': gender,
        'Occupation': occupation,
        'self_employed': self_employed,
        'family_history': family_history,
        'Days_Indoors': days_indoors,
        'Growing_Stress': growing_stress,
        'Changes_Habits': changes_habits,
        'Mental_Health_History': mental_health_history,
        'Mood_Swings': mood_swings,
        'Coping_Struggles': coping_struggles,
        'Work_Interest': work_interest,
        'Social_Weakness': social_weakness,
        'mental_health_interview': mental_health_interview,
        'care_options': care_options
    }

    # Convert to DataFrame
    custom_df = pd.DataFrame([custom_data])

    # Encode the custom data
    custom_df_encoded = custom_df.copy()
    for col in custom_df_encoded.columns:
        if col in encoders:
            le = encoders[col]
            custom_df_encoded[col] = le.transform(custom_df_encoded[col])

    # Create the Symptom_Score feature
    symptom_cols = [
        'Growing_Stress', 'Changes_Habits', 'Mood_Swings',
        'Coping_Struggles', 'Work_Interest', 'Social_Weakness'
    ]
    custom_df_encoded['Symptom_Score'] = custom_df_encoded[symptom_cols].sum(axis=1)

    # Ensure column order matches the model's training data
    custom_df_encoded = custom_df_encoded[training_columns]

    # Make prediction
    prediction_encoded = model.predict(custom_df_encoded)
    prediction_proba = model.predict_proba(custom_df_encoded)

    # Inverse transform to get the label
    prediction = encoders['treatment'].inverse_transform(prediction_encoded)

    # --- Display Prediction Result ---
    st.subheader("Prediction Result")
    if prediction[0] == 'Yes':
        st.success("The model predicts that seeking **treatment is likely**.")
    else:
        st.info("The model predicts that seeking **treatment is not likely**.")

    # Display probabilities
    st.write("Prediction Confidence:")
    st.write(f"Probability of 'No Treatment': {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of 'Treatment': {prediction_proba[0][1]:.2f}")
