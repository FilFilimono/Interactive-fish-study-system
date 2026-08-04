import os
import glob
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq()

def generate_fish_info(fish_class_name):

    system_prompt = """Ты профессиональный ихтиолог и повелитель семи морей. Система компьютерного зрения обнаружила объект.
Твоя задача рассказать об этой рыбе строго по следующей структуре:
1. Название класса на латинском языке + название на русском языке
2. Места обитания
3. Чем питается
4. 5 интересных фактов об этом классе

Отвечай емко, увлекательно (в образе повелителя сеим морей, для того чтобы создать подходящую атмосферу) и только на русском языке. 
Не придумывай несуществующих фактов.
Используй Markdown для красивого оформления (жирный шрифт, списки)."""

    user_prompt = f"Система компьютерного зрения распознала объект: {fish_class_name}. Выведи информацию о нем."

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.4,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Ошибка при генерации ответа от LLM: {e}"

def process_latest_cv_result():
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    payload_dir = os.path.join(current_dir, "..", "data", "output", "nlp_payload")
    payload_dir = os.path.normpath(payload_dir)
    
    if not os.path.exists(payload_dir):
        print(f"Папка {payload_dir} не найдена. Создайте её и положите туда JSON от CV.")
        return

    json_files = glob.glob(os.path.join(payload_dir, "*.json"))
    
    if not json_files:
        print(f"В папке {payload_dir} нет файлов с результатами CV.")
        return

   
    latest_file = json_files[0]
    print(f"Читаем данные из: {latest_file}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
   
    try:
        fish_class = data["detected_objects"][0]["class"]
        confidence = data["detected_objects"][0]["confidence"]
        print(f"Найдена рыба: {fish_class} (уверенность {confidence})")
    except KeyError:
        print("Неверный формат JSON.")
        return

    print("Генерируем ответ через Llama 3...\n")
    print("-" * 50)
    
    
    result_text = generate_fish_info(fish_class)
    
    print(result_text)
    print("-" * 50)

if __name__ == "__main__":
    process_latest_cv_result()