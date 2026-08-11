import gradio as gr
import requests
import json
import os
import random
import numpy as np
from PIL import Image
from gradio_image_prompter import ImagePrompter

API_BASE_URL = os.getenv("NLP_URL", "http://nlp:8000")
PUBLIC_API_URL = os.getenv("PUBLIC_NLP_URL", "http://localhost:8000")
PROCESS_URL = f"{API_BASE_URL}/api/process_media"
CHAT_URL = f"{API_BASE_URL}/api/chat"

def extract_first_frame(video_path):
    if not video_path: return None
    import cv2
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return {"image": frame, "points": []}
    return None

def toggle_media_inputs(media_type):
    is_photo = (media_type == "Фото")
    return gr.update(visible=is_photo), gr.update(visible=not is_photo), gr.update(visible=is_photo), gr.update(visible=not is_photo)

def toggle_vid_prompter(vid_mode):
    return gr.update(visible=(vid_mode == "segmentation_video"))

def process_data(media_type, img_data, vid_path, vid_seg, img_mode, vid_mode):
    file_path = None
    points_str = ""
    mode = ""
    is_temp_file = False

    if media_type == "Фото":
        mode = img_mode
        if img_data is None:
            return gr.update(), gr.update(), "⚠️ Пожалуйста, загрузите фотографию.", ""
        
        image_obj = img_data.get("image")
        if image_obj is None:
            return gr.update(), gr.update(), "⚠️ Пожалуйста, загрузите фотографию.", ""
        
        if isinstance(image_obj, str): 
            file_path = image_obj
        else: 
            file_path = "temp_upload.jpg"
            Image.fromarray(image_obj).save(file_path)
            is_temp_file = True 
        
        if mode == "segmentation_img":
            pts = img_data.get("points", [])
            if pts:
                coords = [[p[0], p[1]] for p in pts]
                labels = [p[2] for p in pts]
                points_str = json.dumps({"coords": coords, "labels": labels})
    else:
        mode = vid_mode
        if not vid_path: return gr.update(), gr.update(), "⚠️ Пожалуйста, загрузите видео.", ""
        file_path = vid_path
        
        if mode == "segmentation_video" and vid_seg is not None:
            pts = vid_seg.get("points", [])
            if pts:
                coords = [[p[0], p[1]] for p in pts]
                labels = [p[2] for p in pts]
                points_str = json.dumps({"coords": coords, "labels": labels})

    try:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            data = {"mode": mode, "points": points_str}
            response = requests.post(PROCESS_URL, files=files, data=data)
            
        if response.status_code != 200:
            return gr.update(), gr.update(), f"❌ Ошибка сервера: {response.text}", ""

        result = response.json()
        session_id = result["session_id"]
        media_url = f"{PUBLIC_API_URL}{result['processed_media_url']}"
        
        llm_reply = result["initial_llm_reply"] + "\n\n***\n💬 *Задай любые другие вопросы в чате ниже!*"

        if media_type == "Фото":
            return gr.update(value=media_url), gr.update(), llm_reply, session_id
        else:
            return gr.update(), gr.update(value=media_url), llm_reply, session_id

    except Exception as e:
        return gr.update(), gr.update(), f"❌ Ошибка: {str(e)}", ""
    finally:
      
        if is_temp_file and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

def chat_with_bot(user_message, chat_history, session_id):
    if not session_id:
        phrases = [
            "Привет! Я Повелитель Семи Морей! Загрузи фото или видео, чтобы я показал тебе магию нейросетей, и мы изучим рыб вместе!",
            "Океан полон тайн! Отправь мне фото или видео рыб, и я расскажу тебе о них всё. А пока я с нетерпением жду твоих файлов 🐟",
            "Без загруженного фото или видео я могу только травить морские байки. Загрузи медиа наверху, чтобы мы начали исследование!"
        ]
        chat_history.append((user_message, random.choice(phrases)))
        return "", chat_history

    try:
        data = {"session_id": session_id, "user_message": user_message}
        response = requests.post(CHAT_URL, data=data)
        
        if response.status_code == 200:
            reply = response.json()["reply"]
            chat_history.append((user_message, reply))
        else:
            chat_history.append((user_message, "❌ Ошибка сервера."))
            
    except Exception as e:
        chat_history.append((user_message, "❌ Ошибка связи."))

    return "", chat_history

with gr.Blocks(title="Interactive Fish Study System", delete_cache=(3600, 3600)) as app:
    gr.Markdown("# 🐟 Interactive Fish Study System")
    gr.Markdown("Загрузите фото или видео рыбы, выберите режим обработки и узнайте о ней всё от Повелителя Семи Морей!")
    
    session_state = gr.State("")

    with gr.Row():
        with gr.Column(scale=1):
            media_selector = gr.Radio(choices=["Фото", "Видео"], value="Фото", label="Что будем изучать?")
            
            with gr.Column(visible=True) as photo_col:
                img_mode = gr.Radio(
                    choices=[("YOLO Детекция", "detection_img"), ("SAM2+ResNet Сегментация", "segmentation_img")], 
                    value="detection_img", label="Режим обработки фото"
                )
                img_input = ImagePrompter(label="Загрузите фото. Левая кнопка мыши: точка на рыбу. Правая кнопка: исключить фон. (Работает только в Сегментации)")         

            with gr.Column(visible=False) as video_col:
                vid_mode = gr.Radio(
                    choices=[("YOLO Трекинг", "tracking_video"), ("SAM2 Видео-сегментация", "segmentation_video")], 
                    value="tracking_video", label="Режим обработки видео"
                )
                vid_input = gr.Video(label="Загрузите видео (.mp4)")
                vid_frame_seg = ImagePrompter(label="Укажите рыбу на ПЕРВОМ кадре (Только для режима Сегментации SAM2)", visible=False)

            submit_btn = gr.Button("🔍 Распознать и изучить", variant="primary")

        with gr.Column(scale=1):
            out_img = gr.Image(label="Результат обработки (Фото)")
            out_vid = gr.Video(label="Результат обработки (Видео)", visible=False)
            
            gr.Markdown("### 📜 Энциклопедическая справка:")
            out_fact = gr.Markdown("*Здесь появится стартовая информация о найденной рыбе...*")

    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 💬 Чат с Повелителем Семи Морей")
            chatbot = gr.Chatbot(label="Диалог", height=400)
            with gr.Row():
                msg_input = gr.Textbox(show_label=False, placeholder="Спроси что-нибудь прямо сейчас...", scale=4)
                send_btn = gr.Button("Отправить", scale=1)
    
    media_selector.change(
        fn=toggle_media_inputs,
        inputs=[media_selector],
        outputs=[photo_col, video_col, out_img, out_vid]
    )

    vid_mode.change(
        fn=toggle_vid_prompter,
        inputs=[vid_mode],
        outputs=[vid_frame_seg]
    )

    vid_input.upload(
        fn=extract_first_frame,
        inputs=[vid_input],
        outputs=[vid_frame_seg]
    )

    submit_btn.click(
        fn=process_data,
        inputs=[media_selector, img_input, vid_input, vid_frame_seg, img_mode, vid_mode],
        outputs=[out_img, out_vid, out_fact, session_state]
    )

    msg_input.submit(fn=chat_with_bot, inputs=[msg_input, chatbot, session_state], outputs=[msg_input, chatbot])
    send_btn.click(fn=chat_with_bot, inputs=[msg_input, chatbot, session_state], outputs=[msg_input, chatbot])
    
    if __name__ == "__main__":
        app.launch(server_name="0.0.0.0", server_port=7860)