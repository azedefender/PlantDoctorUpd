# app.py – Streamlit приложение для диагностики болезней растений (исправленная версия)
import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
from torchvision import transforms
import json
import os

# =============== КОНФИГУРАЦИЯ ===============
IMG_SIZE = 224
MODEL_PATH = "best_model.onnx"
CLASS_NAMES_PATH = "class_names.json"

# Проверка наличия модели и файла классов
if not os.path.exists(MODEL_PATH):
    st.error(f"Файл модели {MODEL_PATH} не найден. Сначала запустите train.py для обучения.")
    st.stop()
if not os.path.exists(CLASS_NAMES_PATH):
    st.error(f"Файл {CLASS_NAMES_PATH} не найден.")
    st.stop()

# Загрузка ONNX-сессии и списка классов
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

# Преобразование для инференса (torchvision, как и при обучении)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =============== ИНТЕРФЕЙС ===============
st.set_page_config(page_title="Диагностика растений", page_icon="🌱")
st.title("🌱 Диагностика болезни растения по листу")
st.markdown("""
Загрузите фотографию листа **крупным планом**, желательно на однородном фоне при хорошем освещении.  
Модель определит культуру и вероятное заболевание (или здоровое состояние).
""")

uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное изображение", use_container_width=True)

    # Предобработка и инференс
    img_np = np.array(image)  # для эвристик качества
    input_tensor = transform(image).unsqueeze(0).numpy()

    outputs = session.run(None, {input_name: input_tensor})[0]
    probabilities = np.exp(outputs) / np.sum(np.exp(outputs), axis=1)
    top2_idx = np.argsort(probabilities[0])[-2:][::-1]
    top2_probs = probabilities[0][top2_idx]

    # Эвристики для предупреждений (используем img_np)
    gray = np.mean(img_np, axis=2)
    brightness = np.mean(gray) / 255.0
    edge_std = np.std(gray - np.median(gray))
    h, w, _ = img_np.shape
    center_region = gray[int(0.3*h):int(0.7*h), int(0.3*w):int(0.7*w)]
    periphery_std = np.std(gray) - np.std(center_region)

    warnings = []
    if brightness < 0.35:
        warnings.append("⚠️ **Низкая освещённость** – точность предсказания может быть снижена.")
    if edge_std > 45 or periphery_std > 20:
        warnings.append("⚠️ **Сложный/неоднородный фон** – модель обучалась на однородном фоне, результат может быть неточным.")
    if top2_probs[0] < 0.75:
        warnings.append("ℹ️ **Низкая уверенность модели**. Рекомендуется переснять лист на светлом однородном фоне.")
    if not warnings:
        st.success("✅ Изображение хорошего качества, условия близки к идеальным.")

    # Вывод результатов
    st.subheader("🔍 Результаты диагностики")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="1-й вариант", value=class_names[top2_idx[0]], delta=f"{top2_probs[0]*100:.1f}%")
    with col2:
        st.metric(label="2-й вариант", value=class_names[top2_idx[1]], delta=f"{top2_probs[1]*100:.1f}%")

    if warnings:
        for w in warnings:
            st.warning(w)

    st.markdown("---")
    st.caption(
        "**Ограничения модели:** "
        "• Обучена на датасете PlantVillage (38 классов культур и болезней); "
        "• Не распознаёт редкие заболевания, повреждения вредителями или неизвестные классы; "
        "• Чувствительна к качеству снимка: однородный фон, хорошее освещение, лист полностью виден; "
        "• Не является заменой профессиональной фитопатологической экспертизы."
    )
