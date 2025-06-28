from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx
import os
from typing import List

app = FastAPI(
    title="Генератор историй",
    description="Веб-сервис для генерации интерактивных историй",
)

# Настройка шаблонов
templates = Jinja2Templates(directory="templates")

# Конфигурация для OpenRouter API
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "your-api-key-here")
OPENROUTER_API_KEY = (
    "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"
)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = 'google/gemini-2.5-flash-lite-preview-06-17'


# Pydantic модели для запросов
class StoryPrompt(BaseModel):
    prompt: str
    characters: str = ""


class StoryContinuation(BaseModel):
    story: str
    continuation: str


class StoryResponse(BaseModel):
    story: str
    continuations: List[str] = []


class StoryGenerator:
    def __init__(self):
        self.client = httpx.AsyncClient()

    async def generate_story(self, prompt: str, characters: str = "") -> str:
        """Генерирует историю на основе затравки и описания персонажей"""
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        user_content = f"Напиши начало истории на основе этой затравки: {prompt}"
        if characters.strip():
            user_content += f"\n\nОписание персонажей: {characters}"
        
        data = {
            "model": "google/gemini-2.5-flash",
            "messages": [
                {
                    "role": "system",
                    "content": "Ты талантливый писатель. Создавай увлекательные истории на основе затравки пользователя. Пиши 2-3 параграфа.",
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            "max_tokens": 500,
            "temperature": 0.8,
        }

        try:
            response = await self.client.post(
                OPENROUTER_URL,
                json=data,
                headers=headers,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Ошибка при генерации истории: {str(e)}"

    async def generate_continuations(self, story: str) -> List[str]:
        """Генерирует варианты продолжения истории"""
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        continuations = []

        for i in range(3):  # Генерируем 3 варианта
            data = {
                "model": MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "Ты талантливый писатель. Предложи краткое описание того, что может произойти дальше в истории.",
                    },
                    {
                        "role": "user",
                        "content": f"История: {story}\n\nПредложи вариант того, что должно произойти дальше (1-2 предложения):",
                    },
                ],
                "max_tokens": 100,
                "temperature": 0.9,
            }

            try:
                response = await self.client.post(
                    OPENROUTER_URL, json=data, headers=headers
                )
                response.raise_for_status()
                result = response.json()
                continuations.append(result["choices"][0]["message"]["content"])
            except Exception as e:
                continuations.append(f"Ошибка: {str(e)}")

        return continuations

    async def close(self):
        """Закрывает HTTP клиент"""
        await self.client.aclose()


story_generator = StoryGenerator()


@app.on_event("shutdown")
async def shutdown_event():
    await story_generator.close()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate", response_model=StoryResponse)
async def generate_story(story_prompt: StoryPrompt):
    if not story_prompt.prompt.strip():
        raise HTTPException(
            status_code=400, detail="Необходимо указать затравку истории"
        )

    story = await story_generator.generate_story(story_prompt.prompt, story_prompt.characters)
    continuations = await story_generator.generate_continuations(story)

    return StoryResponse(story=story, continuations=continuations)


@app.post("/continue", response_model=StoryResponse)
async def continue_story(story_continuation: StoryContinuation):
    if (
        not story_continuation.story.strip()
        or not story_continuation.continuation.strip()
    ):
        raise HTTPException(
            status_code=400,
            detail="Необходимо указать текущую историю и выбранное продолжение",
        )

    # Объединяем историю с выбранным продолжением
    extended_story = f"{story_continuation.story}\n\n{story_continuation.continuation}"
    
    # Генерируем новые варианты продолжения
    continuations = await story_generator.generate_continuations(extended_story)

    return StoryResponse(story=extended_story, continuations=continuations)
