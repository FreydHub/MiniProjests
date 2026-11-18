"""
Главное приложение для классификации эмоций
Поддерживает два режима: загрузка фото и потоковое видео
"""
import streamlit as st
import requests
import json
from PIL import Image
import io
import cv2
import numpy as np
import time
from datetime import datetime
import base64
import threading
from typing import Optional
import queue

# Настройка страницы
st.set_page_config(
    page_title="Классификатор эмоций",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .emotion-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .info-box {
        background-color: #667eea;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Инициализация session state
if 'mode' not in st.session_state:
    st.session_state.mode = 'photo'
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'meme_shown' not in st.session_state:
    st.session_state.meme_shown = False

# Заголовок
st.markdown('<h1 class="main-header">😊 Классификатор эмоций: Лица и Мемы</h1>', unsafe_allow_html=True)

# Боковая панель
st.sidebar.title("⚙️ Настройки")
api_url = st.sidebar.text_input("API URL", value="http://localhost:8000")

# Выбор режима работы
st.sidebar.markdown("### 📋 Режим работы")
mode = st.sidebar.radio(
    "Выберите режим",
    ["📸 Загрузка фото", "📹 Потоковое видео"],
    index=0 if st.session_state.mode == 'photo' else 1
)

st.session_state.mode = 'photo' if mode == "📸 Загрузка фото" else 'video'

# Выбор типа анализа
st.sidebar.markdown("### 🎯 Тип анализа")
analysis_type = st.sidebar.radio(
    "Что анализируем?",
    ["👤 Лицо человека", "😄 Мем/Изображение"],
    help="Выберите 'Лицо' для анализа эмоций на лице, 'Мем' для анализа эмоций в меме или изображении"
)
detect_face = analysis_type == "👤 Лицо человека"

# Функция для отправки изображения на API
def classify_image(image_bytes: bytes, filename: str = "image.jpg", detect_face: bool = True) -> Optional[dict]:
    """Отправка изображения на API для классификации"""
    try:
        files = {"file": (filename, image_bytes, "image/jpeg")}
        params = {"detect_face": detect_face}
        response = requests.post(f"{api_url}/classify", files=files, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Ошибка API: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Не удается подключиться к API. Убедитесь, что сервер запущен на " + api_url)
        return None
    except Exception as e:
        st.error(f"❌ Ошибка: {str(e)}")
        return None

# Функция для получения мема
def get_meme(emotion: str, api_url: str) -> Optional[Image.Image]:
    """Получение случайного мема для эмоции"""
    try:
        response = requests.get(f"{api_url}/meme/{emotion}/base64", timeout=5)
        if response.status_code == 200:
            data = response.json()
            image_data = data['image'].split(',')[1]  # Убираем data:image/jpeg;base64,
            image_bytes = base64.b64decode(image_data)
            return Image.open(io.BytesIO(image_bytes))
        return None
    except Exception as e:
        st.warning(f"Не удалось загрузить мем: {e}")
        return None

# Функция для отображения результатов
def display_results(result: dict, api_url: str):
    """Отображение результатов анализа эмоций"""
    emotion_emojis = {
        'angry': '😠', 'disgust': '🤢', 'fear': '😨',
        'happy': '😊', 'sad': '😢', 'surprise': '😲', 'neutral': '😐'
    }
    
    dominant_emotion = result['dominant_emotion']
    confidence = result['confidence']
    emoji = emotion_emojis.get(dominant_emotion, '😐')
    
    st.markdown(f"""
    <div class="emotion-card">
        <h2>{emoji} {dominant_emotion.upper()}</h2>
        <h3>Уверенность: {confidence:.1%}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Визуализация всех эмоций
    st.subheader("📊 Распределение эмоций")
    emotions = result['emotions']
    
    # Отображение прогресс-баров для каждой эмоции
    for emotion, prob in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
        emoji_icon = emotion_emojis.get(emotion, '😐')
        st.write(f"{emoji_icon} **{emotion.title()}**: {prob:.1%}")
        st.progress(prob)
    
    # Отображение мема
    meme_available = result.get('meme_available', False)
    if meme_available:
        st.markdown("---")
        st.subheader("😄 Случайный мем для этой эмоции")
        
        col_meme1, col_meme2 = st.columns([1, 1])
        with col_meme1:
            if st.button("🎲 Показать мем", key="show_meme"):
                with st.spinner("Загрузка мема..."):
                    meme_image = get_meme(dominant_emotion, api_url)
                    if meme_image:
                        st.session_state.meme_image = meme_image
                        st.session_state.meme_shown = True
        
        with col_meme2:
            if st.button("🔄 Новый мем", key="new_meme"):
                with st.spinner("Загрузка нового мема..."):
                    meme_image = get_meme(dominant_emotion, api_url)
                    if meme_image:
                        st.session_state.meme_image = meme_image
                        st.session_state.meme_shown = True
        
        if 'meme_image' in st.session_state and st.session_state.meme_shown:
            st.image(st.session_state.meme_image, caption=f"Мем для эмоции: {dominant_emotion}", use_column_width=True)
    else:
        st.info("💡 Мемы для этой эмоции пока не загружены. См. MEMES_DOWNLOAD_GUIDE.md для инструкций по загрузке.")

# Режим загрузки фото
if st.session_state.mode == 'photo':
    st.subheader("📸 Загрузка фото для анализа эмоций")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Информационное сообщение в зависимости от типа анализа
        if detect_face:
            info_text = """
            <div class="info-box">
                <strong>Инструкция для анализа лица:</strong><br>
                1. Загрузите фотографию с лицом человека<br>
                2. Нажмите кнопку "Анализировать эмоции"<br>
                3. Получите результат анализа эмоций на лице
            </div>
            """
            upload_help = "Загрузите фотографию лица человека для анализа эмоций"
        else:
            info_text = """
            <div class="info-box">
                <strong>Инструкция для анализа мемов:</strong><br>
                1. Загрузите мем или изображение<br>
                2. Нажмите кнопку "Анализировать эмоции"<br>
                3. Получите результат анализа эмоций в изображении
            </div>
            """
            upload_help = "Загрузите мем или изображение для анализа эмоций"
        
        st.markdown(info_text, unsafe_allow_html=True)
        
        # Загрузка файла
        uploaded_file = st.file_uploader(
            "Выберите изображение",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            help=upload_help
        )
        
        if uploaded_file is not None:
            # Отображение загруженного изображения
            image = Image.open(uploaded_file)
            st.image(image, caption="Загруженное изображение", use_column_width=True)
            
            # Кнопка анализа
            if st.button("🔍 Анализировать эмоции", type="primary"):
                with st.spinner("Анализируем эмоции..."):
                    # Конвертация в bytes
                    img_bytes = uploaded_file.getvalue()
                    result = classify_image(img_bytes, uploaded_file.name, detect_face=detect_face)
                    
                    if result:
                        st.session_state.last_result = result
                        analysis_mode = "лица" if detect_face else "мема"
                        st.success(f"✅ Анализ {analysis_mode} завершен!")
                        st.rerun()
    
    with col2:
        if st.session_state.last_result:
            result = st.session_state.last_result
            # Показать режим анализа
            if 'mode' in result:
                mode_text = "Режим: Лицо" if result['mode'] == 'face' else "Режим: Мем"
                st.info(f"📊 {mode_text}")
            display_results(result, api_url)
        else:
            if detect_face:
                st.info("👈 Загрузите изображение с лицом и нажмите 'Анализировать эмоции'")
            else:
                st.info("👈 Загрузите мем или изображение и нажмите 'Анализировать эмоции'")
            
            # Информация о поддерживаемых эмоциях
            st.markdown("### ℹ️ Поддерживаемые эмоции")
            emotion_emojis = {
                'angry': '😠', 'disgust': '🤢', 'fear': '😨',
                'happy': '😊', 'sad': '😢', 'surprise': '😲', 'neutral': '😐'
            }
            
            emotions_info = [
                ("angry", "злость"),
                ("disgust", "отвращение"),
                ("fear", "страх"),
                ("happy", "радость"),
                ("sad", "грусть"),
                ("surprise", "удивление"),
                ("neutral", "нейтральное")
            ]
            
            for emotion, description in emotions_info:
                emoji = emotion_emojis.get(emotion, '😐')
                st.write(f"{emoji} **{emotion.title()}** - {description}")

# Режим потокового видео
else:
    st.subheader("📹 Потоковое видео с камеры")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <strong>Инструкция:</strong><br>
            1. Разрешите доступ к камере<br>
            2. Направьте камеру на лицо<br>
            3. Нажмите кнопку для захвата кадра<br>
            4. Эмоции будут проанализированы автоматически
        </div>
        """, unsafe_allow_html=True)
        
        # Использование streamlit camera_input
        camera_image = st.camera_input(
            "Включите камеру для анализа эмоций",
            help="Нажмите на камеру для захвата кадра"
        )
        
        if camera_image is not None:
            # Автоматический анализ при захвате кадра
            if not st.session_state.processing:
                st.session_state.processing = True
                
                with st.spinner("Анализируем эмоции..."):
                    # Конвертация изображения
                    img_bytes = camera_image.getvalue()
                    result = classify_image(img_bytes, "camera_frame.jpg", detect_face=detect_face)
                    
                    if result:
                        st.session_state.last_result = result
                        st.session_state.processing = False
                        
                        # Отображение результата на изображении
                        img_array = np.array(Image.open(io.BytesIO(img_bytes)))
                        
                        # Добавление текста с эмоцией на изображение
                        dominant_emotion = result['dominant_emotion']
                        confidence = result['confidence']
                        
                        # Конвертация в OpenCV формат для обработки
                        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        
                        # Добавление текста с эмоцией
                        text = f"{dominant_emotion.upper()}: {confidence:.1%}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1
                        thickness = 2
                        
                        # Получение размеров текста
                        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                        
                        # Рисование прямоугольника для текста
                        cv2.rectangle(img_cv, (10, 10), (text_width + 20, text_height + 30), (0, 0, 0), -1)
                        cv2.putText(img_cv, text, (15, text_height + 20), 
                                  font, font_scale, (0, 255, 0), thickness)
                        
                        # Конвертация обратно в RGB
                        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                        
                        st.image(img_rgb, caption="Результат анализа", use_column_width=True)
                        
                        analysis_mode = "лица" if detect_face else "мема"
                        st.success(f"✅ Анализ {analysis_mode} завершен!")
            
            # Кнопка для повторного анализа
            if st.button("🔄 Анализировать снова", type="primary"):
                st.session_state.processing = False
                st.rerun()
    
    with col2:
        if st.session_state.last_result:
            result = st.session_state.last_result
            # Показать режим анализа
            if 'mode' in result:
                mode_text = "Режим: Лицо" if result['mode'] == 'face' else "Режим: Мем"
                st.info(f"📊 {mode_text}")
            display_results(result, api_url)
            
            # Время последнего анализа
            st.caption(f"Последний анализ: {datetime.now().strftime('%H:%M:%S')}")
        else:
            if detect_face:
                st.info("👈 Включите камеру и захватите кадр с лицом для анализа")
            else:
                st.info("👈 Включите камеру и захватите кадр с изображением/мемом для анализа")
            
            # Статистика и информация
            st.markdown("### ℹ️ Информация")
            if detect_face:
                st.markdown("""
                **Как это работает (анализ лица):**
                1. Камера захватывает кадр с вашим лицом
                2. Система определяет лицо на изображении
                3. Нейронная сеть анализирует эмоции
                4. Результаты отображаются в реальном времени
                
                **Советы для лучших результатов:**
                - Убедитесь, что лицо хорошо освещено
                - Расположитесь лицом к камере
                - Сохраняйте нейтральное или выразительное выражение лица
                """)
            else:
                st.markdown("""
                **Как это работает (анализ мемов):**
                1. Камера захватывает кадр с изображением/мемом
                2. Система обрабатывает всё изображение
                3. Нейронная сеть анализирует эмоции в изображении
                4. Результаты отображаются в реальном времени
                
                **Советы для лучших результатов:**
                - Убедитесь, что изображение хорошо видно
                - Изображение должно быть четким
                - Мемы с выраженными эмоциями анализируются лучше
                """)

# Футер
st.markdown("---")
st.markdown("### ℹ️ О проекте")
st.markdown("""
Этот проект использует машинное обучение для классификации эмоций на лицах людей и в мемах.

**Режимы работы:**
- 📸 **Загрузка фото**: Загрузите фотографию лица или мем для анализа
- 📹 **Потоковое видео**: Анализ эмоций с камеры в реальном времени

**Типы анализа:**
- 👤 **Лицо человека**: Анализ эмоций на лице (автоматическое обнаружение лица)
- 😄 **Мем/Изображение**: Анализ эмоций в меме или изображении (обрабатывается всё изображение)

**Технологии:**
- FastAPI (бэкенд)
- TensorFlow/Keras (ML модель)
- OpenCV (обработка изображений и детекция лиц)
- Streamlit (фронтенд)

**Датасеты:**
- Мемы из датасета Kaggle memotion_dataset_7k
- FER 2013
""")

# Проверка доступности API
st.sidebar.markdown("### 🔌 Статус API")
try:
    response = requests.get(f"{api_url}/health", timeout=2)
    if response.status_code == 200:
        health_data = response.json()
        if health_data.get('model_loaded'):
            st.sidebar.success("✅ API онлайн\n✅ Модель загружена")
        else:
            st.sidebar.warning("⚠️ API онлайн\n❌ Модель не загружена")
    else:
        st.sidebar.error("❌ API недоступен")
except:
    st.sidebar.error("❌ API недоступен\nПроверьте подключение")

# Информация о доступных эмоциях
st.sidebar.markdown("### 📝 Доступные эмоции")
try:
    response = requests.get(f"{api_url}/emotions", timeout=2)
    if response.status_code == 200:
        emotions_data = response.json()
        for emotion in emotions_data['emotions']:
            emoji = {'angry': '😠', 'disgust': '🤢', 'fear': '😨',
                    'happy': '😊', 'sad': '😢', 'surprise': '😲', 'neutral': '😐'}
            st.sidebar.write(f"{emoji.get(emotion, '😐')} {emotion.title()}")
except:
    st.sidebar.write("😠 Angry\n🤢 Disgust\n😨 Fear\n😊 Happy\n😢 Sad\n😲 Surprise\n😐 Neutral")

