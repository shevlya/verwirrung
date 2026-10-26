"""Симулятор дневника Чарли: трансформация текста на основе метрик."""

import random
import re

# Словарь фонетических искажений (как в первых отчётах Чарли)
TYPO_MAP = {
    "что": "што", "сегодня": "севодня", "почему": "пачему",
    "очень": "очень", "вчера": "вчера", "опять": "апять",
    "хочу": "хочю", "знаю": "знаю", "думаю": "думаю",
    "мне": "мне", "еще": "ещо", "это": "это"
}

# Сложные термины для "гениального" Чарли
SMART_WORDS = [
    "экзистенциальный", "когнитивный", "парадигма", "нейропластичность",
    "деконструкция", "апперцепция", "корреляция", "фрустрация",
    "интроспекция", "идентичность", "детерминизм", "сознание"
]


def apply_typos(text: str, error_ratio: float) -> str:
    """Коверкает текст, если error_ratio высокий (ранний Чарли / регресс)."""
    # Ограничиваем максимальный процент искажений, чтобы текст не стал нечитаемым
    effective_ratio = min(error_ratio * 0.5, 0.3) 
    
    words = text.split()
    result = []
    
    for word in words:
        lower_w = word.lower()
        if lower_w in TYPO_MAP and random.random() < effective_ratio:
            result.append(TYPO_MAP[lower_w])
        elif random.random() < effective_ratio * 0.2: # Заменяем "о" на "а" в некоторых словах
            result.append(word.replace("о", "а").replace("О", "А"))
        else:
            result.append(word)
            
    text = " ".join(result)
    
    # Убираем запятые и делаем буквы строчными, если хаос зашкаливает
    if error_ratio > 0.15:
        text = text.replace(",", "").replace(" - ", " ")
        text = text[0].lower() + text[1:] if text else text
        
    return text


def apply_complexity(text: str, ttr: float) -> str:
    """Усложняет текст, если TTR (словарный запас) высокий."""
    if ttr < 0.5:
        return text  # Слишком простой текст, нечего усложнять
        
    sentences = re.split(r'(?<=[.!?]) +', text)
    result = []
    
    for sent in sentences:
        if not sent: continue
        
        # С вероятностью 30% добавляем философское вступление
        if random.random() < 0.3:
            smart_word = random.choice(SMART_WORDS)
            sent = f"В контексте {smart_word}ого анализа, {sent.lower()}"
            
        # С вероятностью 20% добавляем сложный вывод в конец
        if random.random() < 0.2:
            sent = sent.rstrip(".") + ", что является следствием глубинных процессов."
            
        result.append(sent)
        
    return " ".join(result)


def simulate_entry(real_text: str, unknown_ratio: float, ttr: float, entry_id: int) -> str:
    """Главная функция симуляции."""
    # Фиксируем сид, чтобы при повторном рендере текст не менялся
    random.seed(entry_id)
    
    # 1. Базовые искажения
    simulated = apply_typos(real_text, unknown_ratio)
    
    # 2. Усложнение (только если ошибок мало)
    if unknown_ratio < 0.05:
        simulated = apply_complexity(simulated, ttr)
        
    return simulated
