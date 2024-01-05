# ini import library
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import streamlit as st
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#from xgboost import XGBClassifier
import joblib

import json

# ======== Setting title and logo page =========
st.set_page_config(
    page_title="Hungarian Heart Disease",
    page_icon=":heart:"
)

st.title(":red[_HUNGARIAN_] HEART :green[_WEBSITE_] ")
# ===================================


#========== Load Data Set ======================
df = pd.read_csv("Hungarian_Data.csv")
#===============================================

#========= Load MOdel ==========================
# Load the pre-trained model
with st.sidebar:
    menu = st.radio(
        "Chhose Menu",
        ("Home", "App")
    )
if menu == "Home":
    st.header("Hi, Welcome to the Hungarian heart disease prediction website")
    col11,col12 = st.columns(2)
    with col11:
        st.image("1614245470hungary-heart-patriotic-symbol-removebg-preview.png")
    with col12:
        st.write("Welcome to Hungarian Heart, the superior platform for heart disease prediction in the Hungarian region! Based on data from a number of individuals in the region, we are proud to provide accurate predictions using some of the leading models. Our K-Nearest Neighbors (KNN) model has achieved an accuracy of 93.6%, while the XGBoost model shows an accuracy of 91.4%, and the Random Forest model reaches 92%. With this performance, Hungarian Heart is ready to help you detect and prevent potential risks of heart disease. Let's take care of your health together! 💓")
    
    st.header("Analisis Singkat")
    st.write("In this brief analysis, 0.0 indicates that the person is healthy, and more than 0.0 indicates that the person has heart disease. From the data presented, there are 187 people who have good heart health (0.0), and as many as 106 people have a diagnosis of heart disease.")
    st.write("")

    #membuat grafik
    tingkat_serangan_jantung_by_umur = df.groupby(["age","target"]).size().unstack()
    st.bar_chart(tingkat_serangan_jantung_by_umur)


    st.header("chest pain type analysis Data")
    col9,col10 = st.columns(2)
    with col9:
        st.write("The data below shows that many people experience chest disease without symptoms. Asymptomatic chest pain refers to a condition in which a person does not feel chest pain or clinical symptoms that are usually associated with heart disease. Asymptomatic itself means without symptoms or not showing any disease. In the context of chest pain, people who experience chest pain without symptoms do not experience the pain or discomfort that is generally associated with heart problems.")
    with col10:
            # Membuat mapping antara nilai dan label
        mapping_cp = {
            1: "Typical angina",
            2: "Atypical angina",
            3: "Non-anginal pain",
            4: "Asymptomatic"
        }
        # Mengganti label pada sumbu x dengan mapping
        chest_pain_type = df["cp"].map(mapping_cp).value_counts()

        # Menampilkan diagram batang dengan label yang sudah diganti
        st.bar_chart(chest_pain_type)

elif menu == "App":
    st.title("Hungarian Heart Disease")

    algoritma_terpilih = st.selectbox("Pilih Algoritma", options=["K-NN", "RandomForest", "xGBoost"])

    if algoritma_terpilih == "K-NN":
        with open('knnmodel.pkl', 'rb') as file:
            model = pickle.load(file)

    elif algoritma_terpilih == "RandomForest":
        with open('randomforest_Oversampled_normalisasi.pkl', 'rb') as file:
            model = pickle.load(file)

    elif algoritma_terpilih == "xGBoost":
        with open('xgBoost_tuning.pkl', 'rb') as file:
            model = pickle.load(file)


  

    #========== Oversampling Data Set ==============
    df_copy = df.copy()

    featured = df_copy.drop("target",axis=1)
    target = df_copy["target"]

    smote = SMOTE(random_state=42)
    featured_smote_resampled, target_smote_resampled = smote.fit_resample(featured, target)

    normalisasiminmax = MinMaxScaler()
    featured_smote_resampled_normalisasi = normalisasiminmax.fit_transform(featured_smote_resampled)

    X_train, X_test, y_train, y_test = train_test_split(featured_smote_resampled_normalisasi, target_smote_resampled, test_size=0.2, random_state=42,stratify = target_smote_resampled)

    # hasil prediksi model dalam %
    y_pred = model.predict(X_test)

    # kalkuliasi akurasi
    accuracy = accuracy_score(y_test, y_pred)


    # Display the accuracy
    st.write(f"Accuracy dari pickel: ",round(accuracy*100,1))

    # ===================================================

    # Assume min_max_values.json contains the minimum and maximum values for normalization
    with open("min_max_values.json", "r") as json_file:
        min_max_values = json.load(json_file)

    tab1,tab2 = st.tabs(["Single Predixt","Multi Predict"])

    with tab1:
        col1,col2 = st.columns(2)
    # membuat kolom inputan untuk single prediksi
        #kolom umur
        with col1:
            age = st.number_input("Age")

            #kolom sex, 1 untuk cowok 0 untuk cewek
            sex = int(st.selectbox("Sex", options=["Male", "Female"]) == "Male")

            #Chest pain tipenya
            cp_options = {
                1 : "Typical angina",
                2 : "Atypical angina",
                3 : "Non-anginal pain",
                4 : "Asymptomatic"
            }
            cp = st.selectbox("Chest Pain Type", options=list(cp_options.keys()), format_func=lambda x: cp_options[x])
            selected_value = cp_options[cp]

            #inputan untuk kolom trestbps
            trestbps = st.number_input("Trestbps mm Hg")

            #inputan untuk kolom chol
            chol = st.number_input("Serum Cholestoral in mg/dl")

        with col2:
            #inputan untuk fbs dengan urutan sesuai index
            fbs = int(st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["True", "False"]) == "True")

            restecg = ["Normal", "St-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(
                st.selectbox("Resting Electrocardiographic Results", ["Normal", "St-T Wave Abnormality", "Left Ventricular Hypertrophy"])
            )

            # inputan thalac
            thalac = st.number_input("Maximum Heart Rate Achieved")

            #inputan exan
            exang = int(st.selectbox("Exercise Induced Angina", options=["No", "Yes"]) == "Yes")

            #inputan oldpeak
            oldpeak = st.number_input("OldPeak")

        #memb uat tombol prediksi
        predict_button = st.button("Predict", type="primary")
        result = "-"

        #membuat data input menjadi dataframe
        data_input = [[age, sex, cp, trestbps, chol, fbs, restecg, thalac, exang, oldpeak]]
        datauser = pd.DataFrame(data_input, columns=["Age", "Sex", "ChestPainType", "Trestbps", "Chol", "Fbs", "Restecg", "Thalac", "Exang", "OldPeak"])
        
        # Normalize the inputs based on min_max_values
        normalized_inputs = [
        (value - min_max_values[key]["min"]) / (min_max_values[key]["max"] - min_max_values[key]["min"])
        for key, value in zip(min_max_values.keys(), [age, sex, cp, trestbps, chol, fbs, restecg, thalac, exang, oldpeak])
        ]

        # Reshape the normalized inputs into a DataFrame
        datauser_normalized = pd.DataFrame([normalized_inputs], columns=["Age", "Sex", "ChestPainType", "Trestbps", "Chol", "Fbs", "Restecg", "Thalac", "Exang", "OldPeak"])
        
        #ini untuk triger ketika tombol predik dipencet
        if predict_button:
            st.write(datauser)
            # Now, you can use normalized_inputs for prediction
            prediction = model.predict(datauser_normalized)

            if prediction[0] == 0:
                st.success("👍 You are Healthy")
            else:
                st.error(" 💔 You are diagnosed with heart disease")

    with tab2:
        #ini untuk multi prediksi
        #upload csv
        uploaded_df = st.file_uploader("Upload a CSV file", type='csv')

    # Normalisasi input di dalam tab 2
        if uploaded_df:
            # Menggunakan st.file_uploader untuk mengunggah file CSV
            uploaded_df = pd.read_csv(uploaded_df)

            # Load JSON values from file
            with open("min_max_values.json", "r") as json_file:
                min_max_values_json = json.load(json_file)

            # Ensure that the DataFrame is not empty before attempting to iterate
            if not uploaded_df.empty:
                # Iterate through each column in the uploaded DataFrame
                for column in uploaded_df.columns:
                    # Check if the column is present in min_max_values_json
                    if column in min_max_values_json:
                        # Normalize the column values
                        uploaded_df[column] = (uploaded_df[column] - min_max_values_json[column]["min"]) / (min_max_values_json[column]["max"] - min_max_values_json[column]["min"])

                # Convert specific columns to float64
                float_columns = ["trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak"]
                uploaded_df[float_columns] = uploaded_df[float_columns].astype(float)

                # Membuat DataFrame untuk hasil prediksi
                result_df = pd.DataFrame(columns=["Prediction Value", "Prediction Result"])

                # Iterasi melalui setiap baris di DataFrame yang diunggah
                for index, row in uploaded_df.iterrows():
                    # Buat DataFrame untuk satu baris hasil prediksi
                    data_tab2_normalized = pd.DataFrame([row], columns=uploaded_df.columns)

                    # Lakukan prediksi dan tambahkan hasilnya ke dalam DataFrame result_df
                    prediction_tab2 = model.predict(data_tab2_normalized)
                    result_df = result_df.append({"Prediction Value": prediction_tab2[0], "Prediction Result": "Healthy" if prediction_tab2[0] == 0 else "Unhealthy"}, ignore_index=True)
                
                combined_df = pd.concat([uploaded_df, result_df], axis=1)
                # Menampilkan DataFrame yang diunggah dan hasil prediksinya
                st.dataframe(combined_df)
            else:
                st.warning("Uploaded DataFrame is empty.")

