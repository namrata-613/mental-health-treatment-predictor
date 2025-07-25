# Mental Health in the Tech Workplace: A Predictive Analysis

https://mental-health-treatment-predictor.streamlit.app/

### Overview

This project analyzes a dataset on mental health in the tech industry to identify key factors that influence whether an individual seeks treatment for a mental health condition. A machine learning model was built and deployed as a Streamlit web application to predict this outcome based on user-provided data.

### Key Findings & Model Performance

The final LightGBM model achieved a **73% accuracy**, which is a significant improvement over a 50% baseline guess for this complex problem of predicting human behavior.

The most valuable outcome was identifying the key predictors:
* **Family History:** A family history of mental illness was the strongest predictor of seeking treatment.
* **Coping Struggles:** An individual's self-reported difficulty in coping was also highly influential.
* **Care Options:** Awareness of available care options also played a significant role.

These insights suggest that workplace mental health initiatives could be more effective by focusing on resources for those with a family predisposition and by providing better tools for coping with stress.

### Tech Stack
* **Analysis:** Python, Pandas, Scikit-learn, XGBoost, LightGBM
* **Visualization:** Matplotlib, Seaborn
* **Deployment:** Streamlit, GitHub

### How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/namrata-613/mental-health-treatment-predictor.git
    cd mental-health-treatment-predictor
    ```
2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

---
