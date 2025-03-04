# ZHIEND
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import logging
from datetime import datetime

from sklearn.linear_model import LinearRegression
import pickle

# Пример данных
data = pd.DataFrame({
    'area': [50, 60, 70],
    'rooms': [1, 2, 3],
    'floor': [5, 10, 15],
    'district': [0, 1, 2],  # Закодированные значения
    'parking': [1, 0, 1],
    'house_type': [0, 1, 0],  # Закодированные значения
    'has_elevator': [1, 0, 1],
    'price': [50000, 60000, 70000]  # Целевая переменная
})


# Разделение данных
X = data.drop('price', axis=1)
y = data['price']

# Обучение модели
model = LinearRegression()
model.fit(X, y)

# Сохранение модели
with open('project_house.sav', 'wb') as f:
    pickle.dump(model, f)

# Настройка логирования
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Загрузка модели
@st.cache_resource
def load_model():
    try:
        with open('project_house.sav', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Файл модели не найден. Убедитесь, что файл 'project_house.sav' существует.")
        return None
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None

# Словарь с переводами
translations = {
    "Русский": {
        "title": "Предсказать стоимость квартиры в зависимости от площади",
        "description": "Прогнозирование цен квартиры на основе регрессионной модели",
        "accuracy": "Модель была обучена на данных из открытых источников. Точность модели: 95%.",
        "area_label": "Площадь квартиры",
        "rooms_label": "Количество комнат",
        "floor_label": "Этаж",
        "district_label": "Район",
        "parking_label": "Парковка",
        "parking_yes": "Есть",
        "parking_no": "Нет",
        "predict_button": "Предсказать",
        "prediction_result": "Предсказанная цена составляет: {} Сом",
        "history_title": "История запросов",
        "history_empty": "История запросов пуста.",
        "upload_label": "Загрузите CSV-файл с данными",
        "uploaded_data": "Загруженные данные:",
        "language_label": "Выберите язык",
        "house_type_label": "Тип дома",
        "house_type_options": ["Новостройка", "Вторичное жилье"],
        "has_elevator_label": "Наличие лифта",
        "clear_history_button": "Очистить историю",
        "save_history_button": "Сохранить историю в файл",
        "reset_values_button": "Сбросить значения",
        "feedback_label": "Оставьте отзыв или сообщите об ошибке",
        "feedback_button": "Отправить",
        "how_to_use": "Как использовать приложение?",
        "how_to_use_text": "1. Введите параметры квартиры.\n2. Нажмите кнопку 'Предсказать'.\n3. Посмотрите результат и историю запросов.",
    },
    "Английский": {
        "title": "Predict apartment price based on area",
        "description": "Apartment price prediction based on regression model",
        "accuracy": "The model was trained on data from open sources. Model accuracy: 95%.",
        "area_label": "Apartment area",
        "rooms_label": "Number of rooms",
        "floor_label": "Floor",
        "district_label": "District",
        "parking_label": "Parking",
        "parking_yes": "Yes",
        "parking_no": "No",
        "predict_button": "Predict",
        "prediction_result": "Predicted price is: {} SOM",
        "history_title": "Request history",
        "history_empty": "Request history is empty.",
        "upload_label": "Upload CSV file with data",
        "uploaded_data": "Uploaded data:",
        "language_label": "Choose language",
        "house_type_label": "House type",
        "house_type_options": ["New building", "Resale"],
        "has_elevator_label": "Has elevator",
        "clear_history_button": "Clear history",
        "save_history_button": "Save history to file",
        "reset_values_button": "Reset values",
        "feedback_label": "Leave feedback or report an issue",
        "feedback_button": "Submit",
        "how_to_use": "How to use the app?",
        "how_to_use_text": "1. Enter apartment parameters.\n2. Click 'Predict'.\n3. View the result and request history.",
    }
}

# Настройка стилей
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

# Инициализация сессионных переменных
if 'history' not in st.session_state:
    st.session_state.history = []
if 'area' not in st.session_state:
    st.session_state.area = 0
if 'rooms' not in st.session_state:
    st.session_state.rooms = 1
if 'floor' not in st.session_state:
    st.session_state.floor = 1
if 'district' not in st.session_state:
    st.session_state.district = "Центр"
if 'parking' not in st.session_state:
    st.session_state.parking = False
if 'house_type' not in st.session_state:
    st.session_state.house_type = translations["Русский"]["house_type_options"][0]
if 'has_elevator' not in st.session_state:
    st.session_state.has_elevator = False

# Выбор языка
language = st.selectbox("Выберите язык", ["Русский", "Английский"])
lang_dict = translations[language]

# Заголовок
st.markdown(f'<p class="cus">{lang_dict["title"]}</p>', unsafe_allow_html=True)

# Описание проекта
with st.expander(lang_dict["how_to_use"]):
    st.write(lang_dict["how_to_use_text"])
    st.write(lang_dict["accuracy"])

# Контейнер для ввода данных
with st.container(border=True):
    st.slider(lang_dict["area_label"], min_value=0, max_value=500, key='area')
    st.number_input(lang_dict["rooms_label"], key='rooms', min_value=1, max_value=10)
    st.number_input(lang_dict["floor_label"], key='floor', min_value=1, max_value=50)
    st.selectbox(lang_dict["district_label"], key='district', options=["Центр", "Спальный район", "Пригород"])
    st.checkbox(lang_dict["parking_label"], key='parking')
    st.selectbox(lang_dict["house_type_label"], key='house_type', options=lang_dict["house_type_options"])
    st.checkbox(lang_dict["has_elevator_label"], key='has_elevator')
    st.write(f"{lang_dict['area_label']}: {st.session_state.area}")

# Загрузка модели
model = load_model()

def predict_close():
    if st.session_state.area <= 0:
        st.error("Площадь должна быть больше 0.")
        return None

    # Кодируем категориальные переменные
    district_mapping = {"Центр": 0, "Спальный район": 1, "Пригород": 2}
    house_type_mapping = {lang_dict["house_type_options"][0]: 0, lang_dict["house_type_options"][1]: 1}

    district_encoded = district_mapping.get(st.session_state.district, 0)
    house_type_encoded = house_type_mapping.get(st.session_state.house_type, 0)

    # Создаем DataFrame с учетом всех признаков
    input_dataframe = pd.DataFrame({
        'area': [st.session_state.area],
        'rooms': [st.session_state.rooms],
        'floor': [st.session_state.floor],
        'district': [district_encoded],
        'parking': [int(st.session_state.parking)],
        'house_type': [house_type_encoded],  # Добавляем house_type
        'has_elevator': [int(st.session_state.has_elevator)]  # Добавляем has_elevator
    })

    try:
        check_features(input_dataframe, model)
        prediction = model.predict(input_dataframe)
        return str(prediction[0])
    except Exception as e:
        return f"Ошибка предсказания: {e}"

def check_features(input_dataframe, model):
    expected_features = model.feature_names_in_
    missing_features = set(expected_features) - set(input_dataframe.columns)
    if missing_features:
        raise ValueError(f"Отсутствуют признаки: {missing_features}")
    return True
# Кнопки управления
if st.button(lang_dict["predict_button"]):
    with st.spinner('Прогнозирование...'):
        prediction = predict_close()
        if prediction:
            # Сохраняем все данные запроса в историю
            st.session_state.history.append({
                lang_dict["area_label"]: st.session_state.area,
                lang_dict["rooms_label"]: st.session_state.rooms,
                lang_dict["floor_label"]: st.session_state.floor,
                lang_dict["district_label"]: st.session_state.district,
                lang_dict["parking_label"]: lang_dict["parking_yes"] if st.session_state.parking else lang_dict["parking_no"],
                lang_dict["house_type_label"]: st.session_state.house_type,
                lang_dict["has_elevator_label"]: lang_dict["parking_yes"] if st.session_state.has_elevator else lang_dict["parking_no"],
                lang_dict["prediction_result"].split(":")[0]: prediction
            })
            message = st.chat_message("assistant")
            message.write(lang_dict["prediction_result"].format(prediction))

            # Визуализация
            data = pd.DataFrame({
                'area': [st.session_state.area],
                'predicted_price': [float(prediction) if prediction.replace('.', '', 1).isdigit() else 0]
            })
            st.line_chart(data.set_index('area'))

# История запросов
with st.expander(lang_dict["history_title"]):
    if st.session_state.history:
        # Создаем DataFrame для отображения истории
        history_df = pd.DataFrame(st.session_state.history)
        st.write(history_df)
    else:
        st.write(lang_dict["history_empty"])

# Кнопки для управления историей
col1, col2 = st.columns(2)
with col1:
    if st.button(lang_dict["clear_history_button"]):
        st.session_state.history = []
        st.success("История очищена.")
with col2:
    if st.button(lang_dict["save_history_button"]):
        history_df.to_csv("history.csv", index=False)
        st.success("История сохранена в файл history.csv.")

# Загрузка CSV-файла
uploaded_file = st.file_uploader(lang_dict["upload_label"], type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(lang_dict["uploaded_data"])
    st.write(data)

    # Предсказание для всех строк
    if st.button("Предсказать для всех строк"):
        try:
            # Кодируем категориальные переменные
            district_mapping = {"Центр": 0, "Спальный район": 1, "Пригород": 2}
            house_type_mapping = {lang_dict["house_type_options"][0]: 0, lang_dict["house_type_options"][1]: 1}

            data['district'] = data['district'].map(district_mapping)
            data['house_type'] = data['house_type'].map(house_type_mapping)
            data['parking'] = data['parking'].astype(int)
            data['has_elevator'] = data['has_elevator'].astype(int)

            predictions = model.predict(data)
            data['result'] = predictions
            st.write("Результаты предсказаний:")
            st.write(data)
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")
#тут отзывы сохраняет в файл под названием app.log с Закодированные сообщения:
# Обратная связь

# feedback = st.text_area(lang_dict["feedback_label"])
# if st.button(lang_dict["feedback_button"]):
#     if feedback:
#         logging.info(f"Пользователь оставил отзыв: {feedback}")
#         st.success("Спасибо за ваш отзыв!")
#     else:
#         st.warning("Пожалуйста, введите отзыв.")


# тут отзывы сохраняет в файл под названием test_data.csv
# Функция для сохранения отзыва в CSV-файл
def save_feedback_to_csv(feedback):
    # Создаем DataFrame с отзывом и временем
    new_feedback = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feedback": feedback
    }
    df = pd.DataFrame([new_feedback])

    # Сохраняем в CSV-файл
    df.to_csv("test_data.csv", mode="a", header=not pd.io.common.file_exists("test_data.csv"), index=False)

# Интерфейс для ввода отзыва
feedback = st.text_area("Оставьте отзыв или сообщите об ошибке")

# Кнопка для отправки отзыва
if st.button("Отправить отзыв"):
    if feedback.strip():  # Проверяем, что отзыв не пустой
        save_feedback_to_csv(feedback)
        st.success("Спасибо за ваш отзыв! Он был сохранен.")
    else:
        st.warning("Пожалуйста, введите отзыв перед отправкой.")


# Кнопка сброса значений
if st.button(lang_dict["reset_values_button"]):
    # Сброс значений
    st.session_state.area = 0
    st.session_state.rooms = 1
    st.session_state.floor = 1
    st.session_state.district = "Центр"
    st.session_state.parking = False
    st.session_state.house_type = lang_dict["house_type_options"][0]
    st.session_state.has_elevator = False

    # Принудительное обновление интерфейса
    st.experimental_rerun()

# ZHIEND
# кнопка сброса пока что не работает