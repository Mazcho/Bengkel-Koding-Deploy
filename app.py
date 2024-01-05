# import itertools
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import streamlit as st
# import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import json

# ======== Setting title and logo page =========
st.set_page_config(
    page_title="Hungarian Heart Disease",
    page_icon=":heart:"
)
# ===================================

#========= Load MOdel ==========================
# Load the pre-trained model
algoritma_terpilih = st.selectbox("Algoritma", options=["K-NN", "RandomForest", "xGBoost"])

if algoritma_terpilih == "K-NN":
    with open('knnmodel.pkl', 'rb') as file:
        model = pickle.load(file)

elif algoritma_terpilih == "RandomForest":
    with open('randomforest_Oversampled_normalisasi.pkl', 'rb') as file:
        model = pickle.load(file)

elif algoritma_terpilih == "xGBoost":
    with open('xgBoost_tuning.pkl', 'rb') as file:
        model = pickle.load(file)


#========== Load Data Set ======================
df = pd.read_csv("Hungarian_Data.csv")
#===============================================

#========== Oversampling Data Set ==============
df_copy = df.copy()

featured = df_copy.drop("target",axis=1)
target = df_copy["target"]

smote = SMOTE(random_state=42)
featured_smote_resampled, target_smote_resampled = smote.fit_resample(featured, target)

normalisasiminmax = MinMaxScaler()
featured_smote_resampled_normalisasi = normalisasiminmax.fit_transform(featured_smote_resampled)

X_train, X_test, y_train, y_test = train_test_split(featured_smote_resampled_normalisasi, target_smote_resampled, test_size=0.2, random_state=42,stratify = target_smote_resampled)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display the accuracy
st.write(f"Accuracy dari pickel: ",round(accuracy*100,1))

# ===================================================
st.title("Hungarian Heart Disease")
# Assume min_max_values.json contains the minimum and maximum values for normalization
with open("min_max_values.json", "r") as json_file:
    min_max_values = json.load(json_file)

tab1,tab2 = st.tabs(["Single Predixt","Multi Predict"])

with tab1:
# Example inputs (you need to obtain these values from your UI components)
    age = st.number_input("Age")
    sex = int(st.selectbox("Sex", options=["Male", "Female"]) == "Male")
    st.write(sex)
    # cp = ["Typical angina", "atypical angina", "non-anginal pain", "asymptomatic"].index(
    #     st.selectbox("Chest Pain Type", ["Typical angina", "atypical angina", "non-anginal pain", "asymptomatic"])
    # )
    # st.write(cp)

    # anemia = st.number_input("Masukan anemia", value=0)
    cp_options = {
        1 : "Typical angina",
        2 : "Atypical angina",
        3 : "Non-anginal pain",
        4 : "Asymptomatic"
    }
    cp = st.selectbox("Chest Pain Type", options=list(cp_options.keys()), format_func=lambda x: cp_options[x])
    selected_value = cp_options[cp]
    st.write(cp)

    trestbps = st.number_input("Trestbps mm Hg")
    chol = st.number_input("Serum Cholestoral in mg/dl")
    fbs = int(st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["True", "False"]) == "True")
    st.write(fbs)
    restecg = ["Normal", "St-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(
        st.selectbox("Resting Electrocardiographic Results", ["Normal", "St-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    )
    st.write(restecg)
    thalac = st.number_input("Maximum Heart Rate Achieved")
    exang = int(st.selectbox("Exercise Induced Angina", options=["No", "Yes"]) == "Yes")
    st.write(exang)
    oldpeak = st.number_input("OldPeak")

    predict_button = st.button("Predict", type="primary")
    result = "-"

    data_input = [[age, sex, cp, trestbps, chol, fbs, restecg, thalac, exang, oldpeak]]
    datauser = pd.DataFrame(data_input, columns=["Age", "Sex", "ChestPainType", "Trestbps", "Chol", "Fbs", "Restecg", "Thalac", "Exang", "OldPeak"])
    st.write(datauser)
    
    if predict_button:
        # Now, you can use normalized_inputs for prediction
        prediction = model.predict(data_input)
        st.write("Prediction:", prediction[0])  # Assuming prediction is a single value, you may need to adjust if it's an array

        if prediction[0] == 0:
            st.write("Healthy")
        else:
            st.write("Heart Disease")

with tab2:
    file_uploaded = st.file_uploader("Upload a CSV file", type='csv')

    if file_uploaded:
        uploaded_df = pd.read_csv(file_uploaded)
        prediction_arr = model.predict(uploaded_df)

        bar = st.progress(0)
        status_text = st.empty()

        value_pred = []
        result_arr = []

        for prediction in prediction_arr:
            if prediction == 0:
                result = "Healthy"
            else:
                result = "Unhealty"
            
            value_pred.append(prediction)
            result_arr.append(result)
        
        uploaded_result = pd.DataFrame({"Prediction Value": value_pred, "Prediction Result": result_arr})


        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(uploaded_df)
        with col2:
            st.dataframe(uploaded_result)

