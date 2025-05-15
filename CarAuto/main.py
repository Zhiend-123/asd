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

# Добавляем CSS стили
def set_custom_style():
    st.markdown("""
    <style>
        /* Основные стили */
        .stApp {
                background-image: url("https://i.imgur.com/l2Y88ny.jpg");
        background-size: cover;
        background-position: center;
            color: #333333;
        }

        /* Заголовки */
        h1 {
            color: #2a3f5f;
            text-align: center;
            margin-bottom: 30px;
        }

        h2 {
            color: #2a3f5f;
            border-bottom: 2px solid #2a3f5f;
            padding-bottom: 10px;
        }

        /* Кнопки */
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 24px;
            border: none;
            font-weight: bold;
            transition: all 0.3s;
        }

        .stButton>button:hover {
            background-color: #45a049;
            transform: scale(1.05);
        }

        /* Формы */
        .stTextInput>div>div>input, 
        .stNumberInput>div>div>input,
        .stSelectbox>div>div>select {
            border-radius: 5px;
            border: 1px solid #ced4da;
            padding: 8px 12px;
        }

        /* Карточки */
        .stAlert {
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }

        /* Вкладки */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 0 25px;
            border-radius: 5px 5px 0 0;
            background-color: #e9ecef;
        }

        .stTabs [aria-selected="true"] {
            background-color: #2a3f5f;
            color: white;
        }

        /* Прогноз цены */
        .price-prediction {
            font-size: 24px;
            font-weight: bold;
            color: #2a3f5f;
            text-align: center;
            padding: 20px;
            background-color: #e8f5e9;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }

        /* Колонки */
        .stColumn {
            padding: 0 15px;
        }
    </style>
    """, unsafe_allow_html=True)


# Функция для загрузки и предварительной обработки данных
def load_and_preprocess_data(file_path):
    # Загрузка данных
    data = pd.read_csv(file_path)

    # Удаление ненужных столбцов
    if 'Number of Doors' in data.columns:
        data.drop('Number of Doors', axis=1, inplace=True)

    return data


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
    # Устанавливаем стили
    set_custom_style()

    # Заголовок с иконкой
    st.markdown("<h1 style='text-align: center;'>🚗 Прогнозирование цены автомобиля</h1>", unsafe_allow_html=True)

    # Используем вкладки для разделения функционала
    tab1, tab2 = st.tabs(["Обучение модели", "Прогнозирование цены"])

    with tab1:
        st.header("Загрузка данных и обучение модели")

        # Загрузка данных
        uploaded_file = st.file_uploader("Выберите CSV файл с данными", type="csv",
                                         help="Файл должен содержать данные об автомобилях")

        if uploaded_file is not None:
            data = load_and_preprocess_data(uploaded_file)
            st.success("✅ Данные успешно загружены и обработаны!")

            # Показать превью данных
            if st.checkbox("Показать первые 5 строк данных"):
                st.dataframe(data.head())

            # Обучение модели
            if st.button("Обучить модель", key="train_button"):
                with st.spinner("Идет обучение модели... Это может занять несколько минут"):
                    model, data = train_model(data)
                    save_model_and_data(model, data)
                    st.success("🎉 Модель успешно обучена и сохранена!")

                    # Показать метрики
                    y = data['MSRP']
                    X = data.drop(['MSRP'], axis=1)
                    y_pred = model.predict(X)
                    r2 = r2_score(y, y_pred)

                    st.metric("Качество модели (R2 Score)", f"{r2:.4f}",
                              help="R2 Score показывает, насколько хорошо модель предсказывает цены. Значение от 0 до 1, где 1 - идеальное предсказание")

    with tab2:
        st.header("Прогнозирование цены автомобиля")

        model, data = load_model_and_data()

        if model is not None and data is not None:
            st.success("✔ Модель загружена и готова к прогнозированию")

            # Получаем список уникальных марок
            unique_makes = get_unique_makes(data)

            # Форма для ввода данных
            with st.form("prediction_form"):
                st.subheader("Параметры автомобиля")

                col1, col2 = st.columns(2)

                with col1:
                    # Выбор марки из выпадающего списка
                    make = st.selectbox("Марка", unique_makes, index=0,
                                        help="Выберите марку автомобиля из списка")

                    # Получаем модели для выбранной марки
                    models_for_make = get_models_for_make(data, make)
                    model_name = st.selectbox("Модель", models_for_make, index=0,
                                              help="Выберите модель автомобиля")

                    year = st.number_input("Год выпуска", min_value=1990, max_value=2023, value=2015,
                                           help="Укажите год выпуска автомобиля")

                    engine_fuel_type = st.selectbox(
                        "Тип топлива",
                        ["premium unleaded (required)", "regular unleaded", "diesel",
                         "flex-fuel (unleaded/E85)", "electric"],
                        index=0,
                        help="Выберите тип используемого топлива"
                    )

                    engine_hp = st.number_input("Мощность двигателя (л.с.)", min_value=0, max_value=1000, value=200,
                                                help="Укажите мощность двигателя в лошадиных силах")

                    engine_cylinders = st.number_input("Количество цилиндров", min_value=0, max_value=16, value=4,
                                                       help="Укажите количество цилиндров двигателя")

                with col2:
                    transmission_type = st.selectbox(
                        "Тип трансмиссии",
                        ["MANUAL", "AUTOMATIC", "AUTOMATED_MANUAL", "DIRECT_DRIVE"],
                        index=1,
                        help="Выберите тип коробки передач"
                    )

                    driven_wheels = st.selectbox(
                        "Привод",
                        ["rear wheel drive", "front wheel drive", "all wheel drive", "four wheel drive"],
                        index=1,
                        help="Выберите тип привода автомобиля"
                    )

                    market_category = st.text_input("Категория рынка", "Luxury",
                                                    help="Укажите рыночную категорию автомобиля")

                    vehicle_size = st.selectbox(
                        "Размер автомобиля",
                        ["Compact", "Midsize", "Large"],
                        index=1,
                        help="Выберите размер автомобиля"
                    )

                    vehicle_style = st.text_input("Тип кузова", "Sedan",
                                                  help="Укажите тип кузова автомобиля")

                    highway_mpg = st.number_input("Расход на трассе (mpg)", min_value=0, max_value=200, value=30,
                                                  help="Укажите расход топлива на трассе (миль на галлон)")

                    city_mpg = st.number_input("Расход в городе (mpg)", min_value=0, max_value=200, value=20,
                                               help="Укажите расход топлива в городе (миль на галлон)")

                    popularity = st.number_input("Популярность", min_value=0, max_value=10000, value=3000,
                                                 help="Укажите показатель популярности модели")

                submit_button = st.form_submit_button("Рассчитать цену",
                                                      help="Нажмите для расчета прогнозируемой цены")

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

                    # Отображаем результат с красивым оформлением
                    st.markdown(f"""
                    <div class="price-prediction">
                        Прогнозируемая цена автомобиля:<br>
                        <span style="font-size: 36px; color: #4CAF50;">${prediction:,.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("ℹ Модель не обучена. Пожалуйста, перейдите на вкладку 'Обучение модели' и загрузите данные")


if __name__ == "__main__":
    main()
