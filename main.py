from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx
import os
from typing import List
from datetime import datetime
from loguru import logger

from pydantic_settings import BaseSettings, SettingsConfigDict

# Настройка логирования с loguru
logger.add(
    "story_generator.log",
    rotation="10 MB",
    retention="7 days",
)

app = FastAPI(
    title="Генератор историй",
    description="Веб-сервис для генерации интерактивных историй",
)

# Настройка шаблонов
templates = Jinja2Templates(directory="templates")

# Конфигурация для OpenRouter API
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = 'google/gemini-2.5-flash'


# Pydantic модели для запросов
class StoryPrompt(BaseModel):
    prompt: str
    characters: str = ""


class StoryContinuation(BaseModel):
    story: str
    continuation: str


class StoryDownload(BaseModel):
    story: str


class StoryResponse(BaseModel):
    story: str
    continuations: List[str] = []


class Settings(BaseSettings):
    OPENROUTER_API_KEY: str
    model_config = SettingsConfigDict(env_file='.env')


settings = Settings()  # type: ignore


class StoryGenerator:
    def __init__(self):
        self.client = httpx.AsyncClient()

    async def generate_story(self, prompt: str, characters: str = "") -> str:
        """Генерирует историю на основе затравки и описания персонажей"""
        headers = {
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        user_content = f"Напиши начало истории на основе этой затравки: {prompt}"
        if characters.strip():
            user_content += f"\n\nОписание персонажей: {characters}"

        data = {
            "model": MODEL,
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
            "max_tokens": 2000,
            "temperature": 0.8,
        }

        try:
            logger.info(f"Генерация истории - Промпт: {prompt[:100]}...")
            response = await self.client.post(
                OPENROUTER_URL,
                json=data,
                headers=headers,
            )
            response.raise_for_status()
            result = response.json()
            generated_story = result["choices"][0]["message"]["content"]
            logger.info(f"История сгенерирована успешно - Длина: {len(generated_story)} символов")
            return generated_story
        except Exception as e:
            logger.error(f"Ошибка при генерации истории: {str(e)}")
            return f"Ошибка при генерации истории: {str(e)}"

    async def generate_continuations(self, story: str) -> List[str]:
        """Генерирует варианты продолжения истории"""
        headers = {
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        data = {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Ты талантливый писатель. Предложи 3 разных варианта того, что может произойти дальше в истории. Каждый вариант должен быть одним коротким предложением. Пронумеруй варианты от 1 до 3, каждый с новой строки.",
                },
                {
                    "role": "user",
                    "content": f"История: {story}\n\nПредложи 3 кратких варианта того, что должно произойти дальше:",
                },
            ],
            "max_tokens": 150,
            "temperature": 0.9,
        }

        try:
            logger.info(f"Генерация продолжений - История: {len(story)} символов")
            response = await self.client.post(
                OPENROUTER_URL, json=data, headers=headers
            )
            response.raise_for_status()
            result = response.json()

            # Парсим ответ, разделяя по строкам и убирая нумерацию
            response_text = result["choices"][0]["message"]["content"]
            logger.info(f"Получен ответ для продолжений: {response_text}")
            lines = response_text.strip().split('\n')

            continuations = []
            for line in lines:
                # Убираем нумерацию (1., 2., 3. и т.д.) и лишние пробелы
                clean_line = line.strip()
                if clean_line and (clean_line[0].isdigit() or clean_line.startswith('•') or clean_line.startswith('-')):
                    # Убираем первые символы до первого пробела или точки
                    parts = clean_line.split('.', 1)
                    if len(parts) > 1:
                        clean_line = parts[1].strip()
                    else:
                        parts = clean_line.split(' ', 1)
                        if len(parts) > 1:
                            clean_line = parts[1].strip()

                if clean_line:
                    continuations.append(clean_line)

            # Если не удалось распарсить, возвращаем весь текст как один вариант
            if not continuations:
                continuations = [response_text.strip()]

            # Ограничиваем до 3 вариантов
            final_continuations = continuations[:3]
            logger.info(f"Сгенерировано {len(final_continuations)} продолжений: {final_continuations}")
            return final_continuations

        except Exception as e:
            logger.error(f"Ошибка при генерации продолжений: {str(e)}")
            return [f"Ошибка: {str(e)}"]

    async def continue_story_with_direction(self, story: str, direction: str) -> str:
        """Генерирует продолжение истории на основе выбранного направления"""
        headers = {
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        data = {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Ты талантливый писатель. Продолжи историю в соответствии с указанным направлением развития сюжета. Напиши 2-3 новых параграфа, которые органично продолжают историю.",
                },
                {
                    "role": "user",
                    "content": f"Текущая история: {story}\n\nНаправление развития: {direction}\n\nПродолжи историю в этом направлении:",
                },
            ],
            "max_tokens": 2000,
            "temperature": 0.8,
        }

        try:
            logger.info(f"Продолжение истории - Направление: {direction[:100]}...")
            response = await self.client.post(
                OPENROUTER_URL,
                json=data,
                headers=headers,
            )
            response.raise_for_status()
            result = response.json()
            continuation = result["choices"][0]["message"]["content"]
            logger.info(f"Продолжение сгенерировано успешно - Длина: {len(continuation)} символов")
            return continuation
        except Exception as e:
            logger.error(f"Ошибка при продолжении истории: {str(e)}")
            return f"Ошибка при продолжении истории: {str(e)}"

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

    # Генерируем продолжение истории на основе выбранного направления
    new_part = await story_generator.continue_story_with_direction(
        story_continuation.story, story_continuation.continuation
    )

    # Объединяем историю с новым сгенерированным продолжением
    extended_story = f"{story_continuation.story}\n\n{new_part}"

    # Генерируем новые варианты продолжения
    continuations = await story_generator.generate_continuations(extended_story)

    return StoryResponse(story=extended_story, continuations=continuations)


@app.post("/download")
async def download_story(story_download: StoryDownload):
    if not story_download.story.strip():
        raise HTTPException(
            status_code=400,
            detail="Нет истории для скачивания"
        )

    # Создаем имя файла с текущей датой и временем
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"story_{timestamp}.txt"

    # Очищаем текст от HTML тегов если они есть
    clean_story = story_download.story.replace('<br>', '\n').replace('<br/>', '\n')

    return Response(
        content=clean_story,
        media_type="text/plain",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )
