import os
from dotenv import load_dotenv
from groq import Groq


load_dotenv()


client = Groq()

def test_llama_connection():
    print("Отправка запроса на серверы Groq...")
    
    try:
     
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Ты полезный ИИ-ассистент, который знает все о морских существах и рыбках."
                },
                {
                    "role": "user",
                    "content": "Поздоровайся в формате: \"Приветствую тебя человек, я готов Вам рассказать о разных видах морских обитателей, дай мне их узреть \" и скажи одним коротким предложением, что ты готов к работе."
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=700
        )
        
       
        answer = chat_completion.choices[0].message.content
        print("\nОтвет от Llama 3:")
        print(f"{answer}\n")
        
    except Exception as e:
        print(f"\nОшибка подключения: {e}")

if __name__ == "__main__":
    test_llama_connection()