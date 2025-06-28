from flask import Flask, render_template, request, jsonify
import httpx
import asyncio
import os
from typing import List, Dict

app = Flask(__name__)

# Конфигурация для OpenRouter API
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', 'your-api-key-here')
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

class StoryGenerator:
    def __init__(self):
        self.client = httpx.AsyncClient()
    
    async def generate_story(self, prompt: str) -> str:
        """Генерирует историю на основе затравки"""
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "openai/gpt-3.5-turbo",
            "messages": [
                {
                    "role": "system", 
                    "content": "Ты талантливый писатель. Создавай увлекательные истории на основе затравки пользователя. Пиши 2-3 параграфа."
                },
                {
                    "role": "user", 
                    "content": f"Напиши начало истории на основе этой затравки: {prompt}"
                }
            ],
            "max_tokens": 500,
            "temperature": 0.8
        }
        
        try:
            response = await self.client.post(OPENROUTER_URL, json=data, headers=headers)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            return f"Ошибка при генерации истории: {str(e)}"
    
    async def generate_continuations(self, story: str) -> List[str]:
        """Генерирует варианты продолжения истории"""
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        continuations = []
        
        for i in range(3):  # Генерируем 3 варианта
            data = {
                "model": "openai/gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "system", 
                        "content": "Ты талантливый писатель. Предложи краткое описание того, что может произойти дальше в истории."
                    },
                    {
                        "role": "user", 
                        "content": f"История: {story}\n\nПредложи вариант того, что должно произойти дальше (1-2 предложения):"
                    }
                ],
                "max_tokens": 100,
                "temperature": 0.9
            }
            
            try:
                response = await self.client.post(OPENROUTER_URL, json=data, headers=headers)
                response.raise_for_status()
                result = response.json()
                continuations.append(result['choices'][0]['message']['content'])
            except Exception as e:
                continuations.append(f"Ошибка: {str(e)}")
        
        return continuations

story_generator = StoryGenerator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_story():
    prompt = request.json.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'Необходимо указать затравку истории'}), 400
    
    # Запускаем асинхронную функцию в синхронном контексте
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        story = loop.run_until_complete(story_generator.generate_story(prompt))
        continuations = loop.run_until_complete(story_generator.generate_continuations(story))
        return jsonify({
            'story': story,
            'continuations': continuations
        })
    finally:
        loop.close()

@app.route('/continue', methods=['POST'])
def continue_story():
    current_story = request.json.get('story', '')
    chosen_continuation = request.json.get('continuation', '')
    
    if not current_story or not chosen_continuation:
        return jsonify({'error': 'Необходимо указать текущую историю и выбранное продолжение'}), 400
    
    # Объединяем историю с выбранным продолжением
    extended_story = f"{current_story}\n\n{chosen_continuation}"
    
    # Генерируем новые варианты продолжения
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        continuations = loop.run_until_complete(story_generator.generate_continuations(extended_story))
        return jsonify({
            'story': extended_story,
            'continuations': continuations
        })
    finally:
        loop.close()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
