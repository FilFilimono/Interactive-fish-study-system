import gradio as gr
import requests
import json
import os
import cv2
import numpy as np
from PIL import Image
from gradio_image_prompter import ImagePrompter

# Базовый адрес нашего бэкенда
API_BASE_URL = "http://127.0.0.1:8000"
PROCESS_URL = f"{API_BASE_URL}/api/process_media"
CHAT_URL = f"{API_BASE_URL}/api/chat"

def extract_first_frame(video_path):
    """Вытаскивает первый кадр из видео для расстановки точек SAM2"""
    if not video_path:
        return None
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return {"image": frame, "points": []}
    return None

def toggle_media_inputs(media_type, img_mode, vid_mode):
    """Динамически показывает нужные окна ввода в зависимости от выбора"""
    is_photo = (media_type == "Фото")
    is_video = (media_type == "Видео")

    return (
        gr.update(visible=is_photo), # Колонка фото
        gr.update(visible=is_video), # Колонка видео
        gr.update(visible=(is_photo and img_mode == "detection_img")), # Обычное фото
        gr.update(visible=(is_photo and img_mode == "segmentation_img")), # Фото с точками
        gr.update(visible=(is_video and vid_mode == "segmentation_video")), # Первый кадр видео с точками
        gr.update(visible=is_photo), # Вывод фото
        gr.update(visible=is_video)  # Вывод видео
    )

def process_data(media_type, img_det, img_seg, vid_path, vid_seg, img_mode, vid_mode):
    """Отправка данных на бэкенд"""
    file_path = None
    points_str = ""
    mode = ""

    # 1. Сбор данных для ФОТО
    if media_type == "Фото":
        mode = img_mode
        if mode == "detection_img":
            if not img_det: return gr.update(), gr.update(), "⚠️ Пожалуйста, загрузите фотографию.", ""
            file_path = img_det
        else: # segmentation_img
            # Исправлена ошибка с NumPy (if not array)
            if img_seg is None or img_seg.get("image") is None:
                return gr.update(), gr.update(), "⚠️ Пожалуйста, загрузите фотографию.", ""
            
            image_obj = img_seg["image"]
            # Если ImagePrompter вернул путь
            if isinstance(image_obj, str): 
                file_path = image_obj
            # Если ImagePrompter вернул матрицу (NumPy)
            else: 
                file_path = "temp_upload.jpg"
                Image.fromarray(image_obj).save(file_path)
            
            # Собираем точки
            pts = img_seg.get("points", [])
            if pts:
                coords = [[p[0], p[1]] for p in pts]
                labels = [p[2] for p in pts]
                points_str = json.dumps({"coords": coords, "labels": labels})

    # 2. Сбор данных для ВИДЕО
    else:
        mode = vid_mode
        if not vid_path: return gr.update(), gr.update(), "⚠️ Пожалуйста, загрузите видео.", ""
        file_path = vid_path
        
        # Если сегментация видео, собираем точки с первого кадра
        if mode == "segmentation_video" and vid_seg is not None:
            pts = vid_seg.get("points", [])
            if pts:
                coords = [[p[0], p[1]] for p in pts]
                labels = [p[2] for p in pts]
                points_str = json.dumps({"coords": coords, "labels": labels})

    # 3. Отправка на сервер
    try:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            data = {"mode": mode, "points": points_str}
            response = requests.post(PROCESS_URL, files=files, data=data)
            
        if response.status_code != 200:
            return gr.update(), gr.update(), f"❌ Ошибка сервера: {response.text}", ""

        result = response.json()
        session_id = result["session_id"]
        media_url = f"{API_BASE_URL}{result['processed_media_url']}"
        
        # Добавляем призыв к чату
        llm_reply = result["initial_llm_reply"] + "\n\n***\n💬 *Задай любые другие вопросы в чате ниже!*"

        if media_type == "Фото":
            return gr.update(value=media_url), gr.update(), llm_reply, session_id
        else:
            return gr.update(), gr.update(value=media_url), llm_reply, session_id

    except Exception as e:
        return gr.update(), gr.update(), f"❌ Ошибка: {str(e)}", ""

def chat_with_bot(user_message, chat_history, session_id):
    """Свободный чат, который работает даже без загрузки фото"""
    # Если фото не загружено, ИИ отвечает шутливо и завлекает пользователя
    if not session_id:
        bot_reply = "Привет! Я Повелитель Семи Морей! 🌊 Я могу ответить на любые общие вопросы об океане и рыбах. Но будет ГОРАЗДО круче, если ты загрузишь мне фото или видео! Мои нейросети смогут распознать вид, выделить его на экране и рассказать тебе всё о конкретной рыбке. Попробуй! 😉"
        chat_history.append((user_message, bot_reply))
        return "", chat_history

    # Если фото загружено, общаемся по контексту
    try:
        data = {"session_id": session_id, "user_message": user_message}
        response = requests.post(CHAT_URL, data=data)
        
        if response.status_code == 200:
            reply = response.json()["reply"]
            chat_history.append((user_message, reply))
        else:
            chat_history.append((user_message, f"❌ Ошибка сервера."))
            
    except Exception as e:
        chat_history.append((user_message, f"❌ Ошибка связи."))

    return "", chat_history

# ==========================================
# ПОСТРОЕНИЕ ИНТЕРФЕЙСА
# ==========================================

with gr.Blocks(title="Interactive Fish Study System") as app:
    gr.Markdown("# 🐟 Interactive Fish Study System")
    gr.Markdown("Загрузите фото или видео рыбы, выберите режим обработки и узнайте о ней всё от Повелителя Семи Морей!")
    
    session_state = gr.State("")

    with gr.Row():
        # --- ЛЕВАЯ КОЛОНКА ---
        with gr.Column(scale=1):
            media_selector = gr.Radio(choices=["Фото", "Видео"], value="Фото", label="Что будем изучать?")
            
            # БЛОК ФОТО
            with gr.Column(visible=True) as photo_col:
                img_mode = gr.Radio(
                    choices=[("YOLO Детекция", "detection_img"), ("SAM2+ResNet Сегментация", "segmentation_img")], 
                    value="detection_img", label="Режим обработки фото"
                )
                img_input_det = gr.Image(label="Загрузите фото", type="filepath")
                img_input_seg = ImagePrompter(label="Загрузите фото (Левый клик = Объект, Правый клик = Фон)", visible=False)
            
            # БЛОК ВИДЕО
            with gr.Column(visible=False) as video_col:
                vid_mode = gr.Radio(
                    choices=[("YOLO Трекинг", "tracking_video"), ("SAM2 Видео-сегментация", "segmentation_video")], 
                    value="tracking_video", label="Режим обработки видео"
                )
                vid_input = gr.Video(label="Загрузите видео (.mp4)")
                # Окно с первым кадром для точек появится только при выборе SAM2
                vid_frame_seg = ImagePrompter(label="Укажите рыбу на ПЕРВОМ кадре (Левый клик = Объект, Правый = Фон)", visible=False)
                
                gr.Markdown("*⚠️ Внимание: Сегментация SAM2 на видео — это сверхтяжелая нейросеть. Нагрев макбука и время ожидания 1-3 минуты абсолютно нормальны!*")

            submit_btn = gr.Button("🔍 Распознать и изучить", variant="primary")

        # --- ПРАВАЯ КОЛОНКА ---
        with gr.Column(scale=1):
            out_img = gr.Image(label="Результат обработки (Фото)")
            out_vid = gr.Video(label="Результат обработки (Видео)", visible=False)
            
            gr.Markdown("### 📜 Энциклопедическая справка:")
            out_fact = gr.Markdown("*Здесь появится стартовая информация о найденной рыбе...*")

    gr.Markdown("---")
    
    # --- НИЖНИЙ БЛОК ЧАТА ---
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 💬 Чат с Повелителем Семи Морей")
            chatbot = gr.Chatbot(label="Диалог", height=400)
            with gr.Row():
                msg_input = gr.Textbox(show_label=False, placeholder="Спроси что-нибудь прямо сейчас...", scale=4)
                send_btn = gr.Button("Отправить", scale=1)

    # --- ЛОГИКА ИНТЕРФЕЙСА ---
    
    # Обновление окон при переключении кнопок
    input_changers = [media_selector, img_mode, vid_mode]
    for changer in input_changers:
        changer.change(
            fn=toggle_media_inputs,
            inputs=[media_selector, img_mode, vid_mode],
            outputs=[photo_col, video_col, img_input_det, img_input_seg, vid_frame_seg, out_img, out_vid]
        )

    # Вытаскиваем первый кадр из видео, когда оно загружено
    vid_input.upload(
        fn=extract_first_frame,
        inputs=[vid_input],
        outputs=[vid_frame_seg]
    )

    # Запуск обработки
    submit_btn.click(
        fn=process_data,
        inputs=[media_selector, img_input_det, img_input_seg, vid_input, vid_frame_seg, img_mode, vid_mode],
        outputs=[out_img, out_vid, out_fact, session_state]
    )

    # Отправка сообщений в чат
    msg_input.submit(fn=chat_with_bot, inputs=[msg_input, chatbot, session_state], outputs=[msg_input, chatbot])
    send_btn.click(fn=chat_with_bot, inputs=[msg_input, chatbot, session_state], outputs=[msg_input, chatbot])