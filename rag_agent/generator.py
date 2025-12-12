"""Генерация ответа на основе найденных чанков с помощью llama-cpp.

В этом модуле собирается промпт в требуемом формате (4 раздела) и передаётся
в локальную GGUF-модель через `llama-cpp-python`. Модель загружается лениво и
кешируется, чтобы не тратить время на повторную инициализацию в UI.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence

try:  # pragma: no cover - опциональная зависимость
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - fallback при отсутствующей зависимости
    Llama = None  # type: ignore[assignment]


def format_chunk(chunk: dict) -> str:
    """Форматирует чанк для вставки в промпт (с ID и chunk_id)."""

    ticket_id = chunk.get("ticket_id")
    chunk_id = chunk.get("chunk_id")
    text = chunk.get("text", "").strip()
    return f"- [ID={ticket_id} #{chunk_id}] {text}"


def build_prompt(context: str, question: str) -> str:
    """Формирует промпт в требуемой структуре с 4 разделами."""

    return f"""Ты — инженер технической поддержки банка, который помогает разбирать обращения клиентов
по мобильному и веб-банку. Твоя задача — по новому инциденту и историям прошлых заявок
сформулировать понятный и практичный план действий для инженера 1-й линии.


Правила:
- Отвечай ТОЛЬКО по-русски.
- Не пиши «воды» и общих фраз, используй конкретику.
- НЕ копируй текст контекста и самого запроса дословно.
- НЕ повторяй формулировки инструкций, вместо них подставь реальные пункты.
- Строго соблюдай структуру из ЧЕТЫРЁХ разделов (1–4), никаких других заголовков.


КОНТЕКСТ:
{context}


НОВЫЙ ИНЦИДЕНТ:
\"\"\"{question}\"\"\"


Сначала проанализируй новый инцидент и сопоставь его с контекстом.
Затем дай ответ СТРОГО в ЧЕТЫРЁХ разделах:


1) Описание проблемы:
- ...


2) Возможные причины:
- ...


3) Рекомендуемые действия:
- ...


4) Следующие шаги/эскалация:
- ...
"""


def prepare_context(chunks: Sequence[dict]) -> str:
    """Собирает текстовый контекст из найденных чанков."""

    return "\n".join(format_chunk(chunk) for chunk in chunks)


@lru_cache(maxsize=2)
def load_llm(model_path: str | Path, n_ctx: int = 4096) -> "Llama":
    """Загружает GGUF-модель через llama-cpp-python и кеширует экземпляр."""

    if Llama is None:  # pragma: no cover - лениво сообщаем об отсутствии зависимости
        raise ImportError(
            "Для генерации ответа установите зависимость llama-cpp-python "
            "(`pip install llama-cpp-python`) в текущее окружение."
        )

    path = Path(model_path)
    if not path.exists():  # pragma: no cover - защита от неверного пути к модели
        raise FileNotFoundError(
            f"GGUF-модель не найдена по пути: {path}. "
            "Убедитесь, что файл существует и путь указан корректно."
        )

    return Llama(model_path=str(path), n_ctx=max(n_ctx, 2048))


def generate_answer(
    question: str,
    chunks: Iterable[dict],
    model_path: str | Path,
    max_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.9,
    n_ctx: int = 4096,
) -> str:
    """Генерирует ответ на основе топ-N чанков и вопроса оператора."""

    context: List[dict] = list(chunks)
    if not context:
        return "Недостаточно контекста для ответа."

    prompt = build_prompt(context=prepare_context(context), question=question)
    model = load_llm(model_path, n_ctx=n_ctx)
    result = model(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=["</s>", "[SYSTEM]", "[INST]"],
    )
    return result["choices"][0]["text"].strip()
