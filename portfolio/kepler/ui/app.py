import json
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000")
FEATURE_LENGTH = int(os.getenv("FEATURE_LENGTH", "111"))
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", "256"))
DEFAULT_VECTOR = ", ".join(["0.0"] * FEATURE_LENGTH)
DEFAULT_SEQUENCE = ", ".join(["0.0"] * SEQUENCE_LENGTH)

st.set_page_config(page_title="Kepler Transit Detection", layout="wide")
st.title("Kepler Transit Detection")
st.caption("Выбор модели для табличных признаков и последовательностей световой кривой")

with st.sidebar:
    st.subheader("Настройки")
    api_url = st.text_input("API URL", API_URL)
    kepid = st.number_input("KIC / kepid", min_value=0, value=0, step=1)
    source = st.text_input("Источник", value="streamlit-ui")
    model_type = st.selectbox(
        "Модель",
        ["auto", "logreg", "random_forest", "xgboost", "lightgbm", "lstm_improved", "lstm_cnn_lstm"],
        index=0,
    )
    default_mode = "sequence" if model_type in {"lstm_improved", "lstm_cnn_lstm"} else "feature_vector"
    input_mode = st.radio("Тип входа", ["feature_vector", "sequence"], index=0 if default_mode == "feature_vector" else 1)

left, right = st.columns([3, 2])

with left:
    st.subheader("Отправка запроса")
    if input_mode == "feature_vector":
        payload_text = st.text_area("Вектор признаков", value=DEFAULT_VECTOR, height=220)
        expected_len = FEATURE_LENGTH
    else:
        payload_text = st.text_area("Последовательность", value=DEFAULT_SEQUENCE, height=220)
        expected_len = SEQUENCE_LENGTH

    if st.button("Сделать предсказание", type="primary"):
        try:
            values = [float(x.strip()) for x in payload_text.replace("\n", " ").split(",") if x.strip()]
            if len(values) != expected_len:
                st.error(f"Ожидается {expected_len} значений, получено {len(values)}")
            else:
                body = {"model_type": model_type, "kepid": int(kepid) if kepid else None, "source": source}
                if input_mode == "feature_vector":
                    body["feature_vector"] = values
                else:
                    body["sequence"] = values
                response = requests.post(f"{api_url}/predict", json=body, timeout=60)
                response.raise_for_status()
                st.success("Предсказание получено")
                st.json(response.json())
        except Exception as exc:
            st.error(str(exc))

with right:
    st.subheader("Состояние сервиса")
    if st.button("Проверить health"):
        try:
            response = requests.get(f"{api_url}/health", timeout=10)
            response.raise_for_status()
            st.json(response.json())
        except Exception as exc:
            st.error(str(exc))

    st.subheader("Последние запросы")
    if st.button("Обновить журнал"):
        try:
            response = requests.get(f"{api_url}/recent?limit=10", timeout=10)
            response.raise_for_status()
            rows = response.json()
            st.dataframe(pd.DataFrame(rows))
        except Exception as exc:
            st.error(str(exc))

st.markdown("### Пример JSON-запроса")
example_payload = {"model_type": model_type, "kepid": 757076, "source": "manual"}
if input_mode == "feature_vector":
    example_payload["feature_vector"] = [0.0] * FEATURE_LENGTH
else:
    example_payload["sequence"] = [0.0] * SEQUENCE_LENGTH
st.code(json.dumps(example_payload, ensure_ascii=False, indent=2), language="json")

st.markdown("### Полезные ссылки")
st.markdown(f"- API docs: {api_url}/docs")
st.markdown(f"- API metrics: {api_url}/metrics")
st.markdown("- Prometheus: http://localhost:9090")
