import requests
import json
import os


BASE_URL = "http://127.0.0.1:8000/api"

def run_full_test():
    print("Начинаем тестирование бэкенда Interactive Fish Study System...\n")
    
    test_image_path = "clownfish.jpg" 
    
    if not os.path.exists(test_image_path):
        print(f"Ошибка: Файл {test_image_path} не найден! Положи любую картинку в папку со скриптом и назови её test_fish.jpg.")
        return

    print("1. Отправляем фото на сервер (Режим: Детекция YOLO)...")
    url_process = f"{BASE_URL}/process_media"
    
    with open(test_image_path, "rb") as image_file:
        files = {"file": ("test_fish.jpg", image_file, "image/jpeg")}
        data = {
            "mode": "detection_img",
            "points": ""
        }
        
        response = requests.post(url_process, files=files, data=data)
    
    if response.status_code != 200:
        print(f"Ошибка сервера: {response.text}")
        return

    result = response.json()
    print("Фото успешно обработано!")
    print(f"Найденный класс: {result['fish_class']}")
    print(f"Ссылка на обработанное фото: {result['processed_media_url']}")
    print(f"Ответ от ИИ:\n{result['initial_llm_reply']}\n")
    
    session_id = result["session_id"]

    
    print(f"🗣️ 2. Тестируем продолжение диалога (Сессия: {session_id})...")
    url_chat = f"{BASE_URL}/chat"
    
    chat_payload = {
        "session_id": session_id,
        "user_message": "А можно ли держать эту рыбу в домашнем аквариуме? Ответь коротко."
    }
    
    chat_response = requests.post(url_chat, data=chat_payload)
    
    if chat_response.status_code == 200:
        chat_result = chat_response.json()
        print("Чат работает корректно!")
        print(f"Ответ ИИ: {chat_result['reply']}")
    else:
        print(f"Ошибка чата: {chat_response.text}")

if __name__ == "__main__":
    run_full_test()