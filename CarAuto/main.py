import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import streamlit as st
import joblib
import os


# Функция для загрузки данных
def load_data(file_path):
    return pd.read_csv(file_path)


# Функция для обучения модели
def train_model(data):
    X = data.drop(['MSRP'], axis=1)
    y = data['MSRP']

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

    model.fit(X, y)
    return model


# Streamlit интерфейс
def main():
    st.title("🚗 Прогнозирование цены автомобиля")

    uploaded_file = st.file_uploader("Загрузите файл с данными (CSV)", type="csv")

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.success("Данные успешно загружены!")

        if st.button("Обучить модель"):
            model = train_model(data)
            joblib.dump(model, 'model.joblib')
            st.success("Модель успешно обучена!")

    if os.path.exists('model.joblib'):
        model = joblib.load('model.joblib')
        st.header("Прогнозирование цены")

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                make = st.text_input("Марка (Make)", "Toyota")
                model_name = st.text_input("Модель (Model)", "Camry")
                year = st.number_input("Год выпуска", 1990, 2023, 2015)
                engine_hp = st.number_input("Мощность двигателя (л.с.)", 0, 1000, 200)

            with col2:
                transmission_type = st.selectbox(
                    "Тип трансмиссии",
                    ["MANUAL", "AUTOMATIC", "AUTOMATED_MANUAL"])
                vehicle_size = st.selectbox(
                    "Размер автомобиля",
                    ["Compact", "Midsize", "Large"])
                highway_mpg = st.number_input("Расход на трассе", 0, 200, 30)

            if st.form_submit_button("Спрогнозировать цену"):
                input_data = pd.DataFrame({
                    'Make': [make],
                    'Model': [model_name],
                    'Year': [year],
                    'Engine HP': [engine_hp],
                    'Transmission Type': [transmission_type],
                    'Vehicle Size': [vehicle_size],
                    'highway MPG': [highway_mpg]
                }, index=[0])

                prediction = model.predict(input_data)[0]
                st.success(f"Прогнозируемая цена: ${prediction:,.2f}")


if __name__ == "__main__":
    main()