import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq()

def get_initial_fact(fish_class: str) -> dict:
    
    if "," in fish_class:
        system_prompt = f"""
Пользователь загрузил видео, и наша нейросеть обнаружила следующие виды рыб: {fish_class}.
Оформи свой ответ строго по этой структуре:

На видео обнаружено видов рыб: [Укажи количество].

### 🐟 [Название первой рыбы на русском]
* **Краткое описание:** [Одно емкое предложение об этой рыбе].
* **Интересный факт:** [Один самый удивительный факт].

### 🐟 [Название второй рыбы на русском]
* **Краткое описание:** [Одно емкое предложение об этой рыбе].
* **Интересный факт:** [Один самый удивительный факт].

(Продолжай так для каждой рыбы из списка).
Отвечай как Повелитель Семи Морей, с легким морским колоритом, но по делу!
"""
    else:
        system_prompt = f"""
Ты профессиональный ихтиолог и повелитель семи морей. Система компьютерного зрения распознала объект: {fish_class}.
Твоя первая задача рассказать о найденной рыбе строго по следующей структуре:

1. Название класса на латинском языке + название на русском языке
2. Места обитания
3. Чем питается
4. 5 интересных фактов об этом классе

Для всех последующих вопросов пользователя просто отвечай как эксперт и повелитель семи морей, опираясь на контекст предыдущей беседы.
Отвечай емко, увлекательно (в образе повелителя семи морей, для того чтобы создать подходящую атмосферу) и только на русском языке. 
Не придумывай несуществующих фактов.
Используй Markdown для красивого оформления (жирный шрифт, списки).
"""
    
    messages = [{"role": "system", "content": system_prompt}]
    
    try:
        response = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=800
        )
        bot_reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": bot_reply})
        
        return {"reply": bot_reply, "messages_history": messages}
    except Exception as e:
        raise Exception(f"Ошибка LLM (Initial Fact): {e}")

def get_chat_response(messages_history: list, user_input: str) -> dict:
    messages_history.append({"role": "user", "content": user_input})
    
    try:
        response = client.chat.completions.create(
            messages=messages_history,
            model="llama-3.3-70b-versatile",
            temperature=0.6,
            max_tokens=500
        )
        bot_reply = response.choices[0].message.content
        messages_history.append({"role": "assistant", "content": bot_reply})
        
        return {"reply": bot_reply, "messages_history": messages_history}
    except Exception as e:
        raise Exception(f"Ошибка LLM (Chat): {e}")