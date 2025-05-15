import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import joblib
import os


# Функция для загрузки и предварительной обработки данных
def load_and_preprocess_data(file_path):
    # Загрузка данных
    data = pd.read_csv(file_path)
    # Удаление ненужных столбцов
    if 'Number of Doors' in data.columns:
        data.drop('Number of Doors', axis=1, inplace=True)

    return data

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://i.imgur.com/l2Y88ny.jpg");
        background-size: cover;
        background-position: center;
    }
    .cus {
        font-size: 22px !important; 
        color: yellow;
        text-align: center; 
        padding: 10px; 
        border-radius: 10px; 
        background-color: rgba(0, 0, 0, 0.5); /* Полупрозрачный фон */
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.5); /* Тень */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Функция для получения уникальных марок из данных
def get_unique_makes(data):
    return sorted(data['Make'].unique())


# Функция для получения моделей по выбранной марке
def get_models_for_make(data, selected_make):
    return sorted(data[data['Make'] == selected_make]['Model'].unique())


# Функция для обучения модели
def train_model(data):
    # Разделение на признаки и целевую переменную
    X = data.drop(['MSRP'], axis=1)
    y = data['MSRP']

    # Определение числовых и категориальных признаков
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Создание трансформеров для числовых и категориальных признаков
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Объединение трансформеров
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Создание пайплайна с предобработкой и моделью
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение модели
    model.fit(X_train, y_train)

    # Оценка модели
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model MSE: {mse}")
    print(f"Model R2 Score: {r2}")

    return model, data  # Возвращаем также данные для получения списка марок


# Функция для сохранения модели и данных
def save_model_and_data(model, data, model_path='model.joblib', data_path='data.joblib'):
    joblib.dump(model, model_path)
    joblib.dump(data, data_path)


# Функция для загрузки модели и данных
def load_model_and_data(model_path='model.joblib', data_path='data.joblib'):
    if os.path.exists(model_path) and os.path.exists(data_path):
        model = joblib.load(model_path)
        data = joblib.load(data_path)
        return model, data
    return None, None


# Streamlit интерфейс
def main():
    st.title("🚗 Прогнозирование цены автомобиля")

    # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите файл с данными (CSV)", type="csv")

    if uploaded_file is not None:
        data = load_and_preprocess_data(uploaded_file)
        st.success("Данные успешно загружены и обработаны!")

        # Обучение модели
        if st.button("Обучить модель"):
            with st.spinner("Обучение модели..."):
                model, data = train_model(data)
                save_model_and_data(model, data)
                st.success("Модель успешно обучена и сохранена!")

                # Показать метрики
                y = data['MSRP']
                X = data.drop(['MSRP'], axis=1)
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)

                st.metric("R2 Score модели на всех данных", f"{r2:.4f}")

    # Прогнозирование
    st.header("Прогнозирование цены")

    model, data = load_model_and_data()

    if model is not None and data is not None:
        st.success("Модель загружена и готова к прогнозированию!")

        # Получаем список уникальных марок
        unique_makes = get_unique_makes(data)

        # Форма для ввода данных
        with st.form("prediction_form"):
            st.subheader("Введите параметры автомобиля")

            col1, col2 = st.columns(2)

            with col1:
                # Выбор марки из выпадающего списка
                make = st.selectbox("Марка (Make)", unique_makes)

                # Получаем модели для выбранной марки
                models_for_make = get_models_for_make(data, make)
                model_name = st.selectbox("Модель (Model)", models_for_make)

                year = st.number_input("Год выпуска (Year)", min_value=1990, max_value=2023, value=2015)
                engine_fuel_type = st.selectbox(
                    "Тип топлива (Engine Fuel Type)",
                    ["premium unleaded (required)", "regular unleaded", "diesel", "flex-fuel (unleaded/E85)",
                     "electric"]
                )
                engine_hp = st.number_input("Мощность двигателя (л.с.)", min_value=0, max_value=1000, value=200)
                engine_cylinders = st.number_input("Количество цилиндров", min_value=0, max_value=16, value=4)

            with col2:
                transmission_type = st.selectbox(
                    "Тип трансмиссии (Transmission Type)",
                    ["MANUAL", "AUTOMATIC", "AUTOMATED_MANUAL", "DIRECT_DRIVE"]
                )
                driven_wheels = st.selectbox(
                    "Привод (Driven_Wheels)",
                    ["rear wheel drive", "front wheel drive", "all wheel drive", "four wheel drive"]
                )
                market_category = st.text_input("Категория рынка (Market Category)", "Luxury")
                vehicle_size = st.selectbox(
                    "Размер автомобиля (Vehicle Size)",
                    ["Compact", "Midsize", "Large"]
                )
                vehicle_style = st.text_input("Тип кузова (Vehicle Style)", "Sedan")
                highway_mpg = st.number_input("Расход на трассе (highway MPG)", min_value=0, max_value=200, value=30)
                city_mpg = st.number_input("Расход в городе (city mpg)", min_value=0, max_value=200, value=20)
                popularity = st.number_input("Популярность (Popularity)", min_value=0, max_value=10000, value=3000)

            submit_button = st.form_submit_button("Спрогнозировать цену")

            if submit_button:
                # Создаем DataFrame с введенными данными
                input_data = pd.DataFrame({
                    'Make': [make],
                    'Model': [model_name],
                    'Year': [year],
                    'Engine Fuel Type': [engine_fuel_type],
                    'Engine HP': [engine_hp],
                    'Engine Cylinders': [engine_cylinders],
                    'Transmission Type': [transmission_type],
                    'Driven_Wheels': [driven_wheels],
                    'Market Category': [market_category],
                    'Vehicle Size': [vehicle_size],
                    'Vehicle Style': [vehicle_style],
                    'highway MPG': [highway_mpg],
                    'city mpg': [city_mpg],
                    'Popularity': [popularity]
                })
                # Делаем прогноз
                prediction = model.predict(input_data)[0]
                # Отображаем результат
                st.success(f"Прогнозируемая цена автомобиля: ${prediction:,.2f}")
    else:
        st.warning("Модель не обучена. Пожалуйста, загрузите данные и обучите модель.")
if __name__ == "__main__":
    main()