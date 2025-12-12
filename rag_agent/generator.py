"""Генерация ответа на основе найденных чанков с помощью llama-cpp."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence

try:  # pragma: no cover - опциональная зависимость
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - fallback при отсутствующей зависимости
    Llama = None  # type: ignore[assignment]


def format_chunk(chunk: dict) -> str:
    """Форматирует чанк для вставки в промпт."""

    ticket_id = chunk.get("ticket_id")
    chunk_id = chunk.get("chunk_id")
    text = chunk.get("text", "").strip()
    return f"[ticket_id={ticket_id} chunk_id={chunk_id}] {text}"


def build_prompt(question: str, chunks: Sequence[dict]) -> str:
    """Создаёт единый промпт для всех моделей (phi-2, mistral, zephyr, tinyllama)."""

    formatted_chunks = "\n".join(format_chunk(chunk) for chunk in chunks)
    system_instructions = (
        "Ты — помощник оператора. Используй только предоставленные фрагменты тикетов "
        "для ответа на вопрос. Обязательно ссылайся на ticket_id в тексте ответа. "
        "Если информации недостаточно, скажи об этом. Ответ делай лаконичным."
    )
    user_block = (
        f"Вопрос оператора: {question}\n\n"
        f"Контекст (топ-{len(chunks)} чанков):\n{formatted_chunks}\n\n"
        "Дай короткий ответ (не более 1024 токенов), сохраняя ticket_id из контекста."
    )
    return f"<s>[SYSTEM]\n{system_instructions}\n[/SYSTEM]\n{user_block}\n"


@lru_cache(maxsize=2)
def load_llm(model_path: str | Path, n_ctx: int = 2048) -> "Llama":
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

    return Llama(model_path=str(path), n_ctx=n_ctx)


def generate_answer(
    question: str,
    chunks: Iterable[dict],
    model_path: str | Path,
    max_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.9,
    n_ctx: int = 2048,
) -> str:
    """Генерирует ответ на основе топ-N чанков и вопроса оператора."""

    context: List[dict] = list(chunks)
    if not context:
        return "Недостаточно контекста для ответа."

    prompt = build_prompt(question, context)
    model = load_llm(model_path, n_ctx=n_ctx)
    result = model(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=["</s>", "[SYSTEM]", "[INST]"],
    )
    return result["choices"][0]["text"].strip()
