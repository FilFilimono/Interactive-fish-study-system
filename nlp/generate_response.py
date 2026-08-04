import os
import glob
import json
from dotenv import load_dotenv
from groq import Groq


load_dotenv()
client = Groq()

def chat_with_user():


    messages = [
        {
            "role": "system",
            "content": """Ты — профессиональный ихтиолог. Система компьютерного зрения будет передавать тебе названия найденных рыб.
Твоя первая задача — рассказать о найденной рыбе строго по структуре:
1. Название класса на латинском языке
2. Места обитания
3. Чем питается
4. 5 интересных фактов об этом классе

Для всех последующих вопросов пользователя — просто отвечай как вежливый эксперт, опираясь на контекст предыдущей беседы. Отвечай только на русском языке, используй Markdown."""
        }
    ]

    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    payload_dir = os.path.normpath(os.path.join(current_dir, "..", "data", "output", "nlp_payload"))
    
    if not os.path.exists(payload_dir):
        print(f"Папка {payload_dir} не найдена.")
        return

    json_files = glob.glob(os.path.join(payload_dir, "*.json"))
    if not json_files:
        print("Не найдено результатов распознавания (JSON файлов).")
        return


    latest_file = json_files[0]
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    try:
        fish_class = data["detected_objects"][0]["class"]
        confidence = data["detected_objects"][0]["confidence"]
        print(f"Найдена рыба: {fish_class} (Уверенность: {confidence})\n")
    except KeyError:
        print("Неверный формат JSON.")
        return

    
    first_prompt = f"Система компьютерного зрения распознала объект: {fish_class}. Выведи стартовую информацию о нем по нашей структуре."
    messages.append({"role": "user", "content": first_prompt})

    print("Бот: Формирую энциклопедическую справку...\n")
    
    
    response = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=800
    )
    
    bot_reply = response.choices[0].message.content
    print(bot_reply)
    
    
    messages.append({"role": "assistant", "content": bot_reply})
    
    print("\n" + "="*50)
    print("Чат открыт! Задавайте любые уточняющие вопросы (для выхода напишите 'выход').")
    print("="*50 + "\n")

    
    while True:
        user_input = input("Вы: ")
        
    
        if user_input.lower() in ['выход', 'exit', 'quit', 'q']:
            print("Бот: До свидания! Буду рад рассказать о других рыбах.")
            break
            
        if not user_input.strip():
            continue

       
        messages.append({"role": "user", "content": user_input})
        
        try:
            
            chat_completion = client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",
                temperature=0.6, 
                max_tokens=500
            )
            
            reply = chat_completion.choices[0].message.content
            print(f"\nБот: {reply}\n")
            
        
            messages.append({"role": "assistant", "content": reply})
            
        except Exception as e:
            print(f"\nОшибка API: {e}\n")

if __name__ == "__main__":
    chat_with_user()