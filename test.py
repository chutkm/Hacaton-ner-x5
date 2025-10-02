# """
# Генерация файла submission.csv на основе обученной модели NER.

# Использование:
#     python generate_submission.py

# Ожидаемые пути:
#     - Модель: /home/ubuntu/x5-ner-base-label-pro/models/v_grok/final_model/
#     - Входной файл: /home/ubuntu/x5-ner-base-label/data/submission_example.csv
#     - Выходной файл: /home/ubuntu/x5-ner-base-label/data/subm_v_t100_grok.csv
# """

# import os
# import sys
# import json
# import pandas as pd
# import torch
# from typing import List, Tuple, Dict, Any, Optional

# # Добавляем корень проекта в PYTHONPATH для корректного импорта
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # Импортируем необходимые компоненты напрямую
# from transformers import AutoTokenizer, AutoConfig
# from src.utils import LABELS, logits_to_labels_and_confs,fix_bio_spans


# from typing import List, Tuple

# def correct_empty_spans(spans: List[Tuple[int, int, str]], sample: str) -> List[Tuple[int, int, str]]:
#     """
#     Корректирует пустой список спанов, возвращая [(0, len(sample), 'O')] для непустого текста.
#     Если спаны уже есть или текст пустой, возвращает исходные спаны.

#     Args:
#         spans: Список кортежей (start, end, label).
#         sample: Входной текст (запрос).

#     Returns:
#         Список спанов, где для пустого списка и непустого текста возвращается [(0, len(sample), 'O')].
#     """
#     if not spans and sample and sample.strip():
#         return [(0, len(sample), 'O')]
#     return spans


# def load_model_correctly(model_dir: str):
#     """
#     Загружает модель с правильной обработкой CRF слоя
#     """
#     print(f"Загрузка модели из {model_dir} с поддержкой CRF...")
    
#     # Загружаем метаданные
#     meta_path = os.path.join(model_dir, "meta.json")
#     if not os.path.exists(meta_path):
#         raise ValueError(f"Файл meta.json не найден в {model_dir}")
    
#     with open(meta_path, 'r') as f:
#         meta = json.load(f)
    
#     # Определяем, использовался ли CRF
#     use_crf = meta.get("use_crf", False)
#     print(f"Обнаружено: use_crf = {use_crf}")
    
#     # Загружаем токенизатор
#     tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
#     # Загружаем config
#     config = AutoConfig.from_pretrained(model_dir)
    
#     # Добавляем кастомные параметры в config
#     config.use_crf = use_crf
#     config.num_labels = len(LABELS)
    
#     # Загружаем модель с правильной архитектурой
#     from src.model.modular import ModularTokenClassifier
#     model = ModularTokenClassifier.from_pretrained(
#         model_dir,
#         config=config,
#         use_crf=use_crf,
#         num_labels=len(LABELS)
#     )
    
#     # Устанавливаем режим инференса
#     model.eval()
    
#     # Получаем порог из метаданных
#     threshold = meta["best_threshold"]["thr"]
    
#     return tokenizer, model, threshold

# # def fix_bio_spans(spans: List[Tuple[int, int, str]], text: Optional[str] = None) -> List[Tuple[int, int, str]]:
# #     """
# #     Фиксит префиксы B/I в списке char-спанов (Entity).
# #     - Сортирует по start.
# #     - Если consecutive спаны adjacent (start <= prev_end +1) и same entity: первый B-, остальные I-.
# #     - I- без prev -> B-.
# #     - Если spans пустой и передан text, возвращает [(0, len(text), 'O')].
# #     - Возвращает fixed list спанов с обновлёнными labels.
# #     """
# #     if not spans and text:
# #         return [(0, len(text), 'O')]
    
# #     if not spans:
# #         return []
    
# #     sorted_spans = sorted(spans, key=lambda x: x[0])
    
# #     fixed: List[Tuple[int, int, str]] = []
# #     prev_ent: Optional[str] = None
    
# #     for s, e, l in sorted_spans:
# #         if l == "O" or "-" not in l:
# #             fixed.append((s, e, "O"))
# #             prev_ent = None
# #             continue
        
# #         pref, ent = l.split("-", 1)
# #         ent = ent.upper()
        
# #         is_continuation = (prev_ent == ent) and fixed and (s <= fixed[-1][1] + 1)
        
# #         if is_continuation:
# #             new_l = f"I-{ent}"
# #         else:
# #             new_l = f"B-{ent}"
        
# #         fixed.append((s, e, new_l))
# #         prev_ent = ent
    
#     return fixed

# def format_spans(spans: List[Tuple[int, int, str]]) -> str:
#     """
#     Форматирует список спанов в строку в требуемом формате.
#     Пример: [(0, 5, 'B-TYPE'), (6, 9, 'I-TYPE')]
#     """
#     if not spans:
#         return "[]"
    
#     formatted = "["
#     for i, (start, end, label) in enumerate(spans):
#         if i > 0:
#             formatted += ", "
#         # Сохраняем оригинальные B-/I- метки из предсказания
#         formatted += f"({start}, {end}, '{label}')"
#     formatted += "]"
#     return formatted

# def predict_spans(text: str, model, tokenizer, device: str, threshold: float, max_length: int = 128) -> List[Tuple[int, int, str]]:
#     """
#     Предсказывает спаны для текста с использованием модели
#     """
#     # Обработка пустых запросов
#     if not text or not text.strip():
#         return []
    
#     # Токенизация
#     inputs = tokenizer(
#         text, 
#         return_offsets_mapping=True, 
#         return_tensors="pt", 
#         truncation=True, 
#         max_length=max_length
#     )
    
#     # Извлекаем offsets и перемещаем тензоры на устройство
#     offsets = inputs.pop("offset_mapping").squeeze().tolist()
#     inputs = {k: v.to(device) for k, v in inputs.items()}
    
#     # Предсказание
#     with torch.no_grad():
#         outputs = model(**inputs)
    
#     # Обработка логитов
#     if hasattr(outputs, "logits"):
#         logits = outputs.logits[0]
#     elif "logits" in outputs:
#         logits = outputs["logits"][0]
#     else:
#         raise ValueError("Модель не возвращает логиты")
    
#     # Преобразование логитов в метки и confidence
#     labels, confs = logits_to_labels_and_confs(logits, {i: l for i, l in enumerate(LABELS)})
    
#     # Пост-обработка
#     spans = []
#     current_span = None
    
#     for i, (a, b) in enumerate(offsets):
#         # Пропускаем специальные токены
#         if a == b or i >= len(labels) or i >= len(confs):
#             continue
        
#         label = labels[i]
#         conf = confs[i]
        
#         # Обработка O-меток
#         if label == "O":
#             if current_span:
#                 spans.append(current_span)
#                 current_span = None
#             continue
        
#         # Разбираем метку
#         if "-" not in label:
#             continue
        
#         prefix, entity = label.split("-", 1)
        
#         # Начало новой сущности
#         if prefix == "B" or current_span is None or current_span["entity"] != entity:
#             if current_span:
#                 spans.append(current_span)
#             current_span = {
#                 "start_index": a,
#                 "end_index": b,
#                 "entity": entity,
#                 "original_label": label,
#                 "confs": [conf]
#             }
#         # Продолжение текущей сущности
#         elif prefix == "I" and current_span and current_span["entity"] == entity:
#             current_span["end_index"] = b
#             current_span["confs"].append(conf)
    
#     # Добавляем последнюю сущность
#     if current_span:
#         spans.append(current_span)
    
#     # Конвертируем спаны в формат [(start, end, label)]
#     formatted_spans = [(span["start_index"], span["end_index"], span["original_label"]) for span in spans]
    
#     # Исправляем префиксы B-/I- с помощью fix_bio_spans
#     formatted_spans = fix_bio_spans(formatted_spans, text)
    
#     # Фильтрация по порогу
#     filtered_spans = []
#     for span, orig_span in zip(formatted_spans, spans):
#         avg_conf = sum(orig_span["confs"]) / len(orig_span["confs"])
#         if avg_conf >= threshold:
#             filtered_spans.append(span)
    
#     # Корректировка пустых спанов
#     filtered_spans = correct_empty_spans(filtered_spans, text)
    
#     return filtered_spans

# def logits_to_labels_and_confs(logits: torch.Tensor, id2label: Dict[int, str]):
#     """
#     Безопасное преобразование логитов в метки и confidence
#     """
#     # Проверяем форму тензора
#     if logits.dim() == 1:
#         # Добавляем размерность batch
#         logits = logits.unsqueeze(0)
    
#     # Применяем softmax
#     probs = torch.softmax(logits, dim=-1)
    
#     # Получаем предсказания и confidence
#     preds = torch.argmax(probs, dim=-1)
#     max_probs = torch.max(probs, dim=-1)[0]
    
#     # Обрабатываем случай скалярного тензора
#     if preds.dim() == 0:
#         labels = [id2label[preds.item()]]
#         confs = [max_probs.item()]
#     else:
#         labels = [id2label[p.item()] for p in preds]
#         confs = max_probs.tolist()
    
#     return labels, confs

# def main():
#     # Пути к файлам
#     model_dir = "/home/ubuntu/x5-ner-base-label-pro/models/v_prep_up/final_model"
#     input_csv = "/home/ubuntu/x5-ner-base-label/data/submission_example.csv"
#     output_csv = "/home/ubuntu/x5-ner-base-label-pro/data/subm_prep_up.csv"
    
#     print(f"Загрузка модели из {model_dir}...")
#     try:
        
#         tokenizer, model, threshold = load_model_correctly(model_dir)
#         print(f"Модель загружена успешно. Используем порог: {threshold:.2f}")
#     except Exception as e:
#         print(f"Ошибка загрузки модели: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)
    
#     # Определяем устройство
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = model.to(device)
#     print(f"Используем устройство: {device}")
    
#     # Считываем входные данные
#     print(f"Чтение данных из {input_csv}...")
#     try:
#         df = pd.read_csv(input_csv, sep=";")
#         print(f"Загружено {len(df)} запросов для обработки")
#     except Exception as e:
#         print(f"Ошибка чтения CSV: {str(e)}")
#         sys.exit(1)
    
#     # Проверяем структуру входного файла
#     if 'sample' not in df.columns:
#         print("Ошибка: Входной CSV должен содержать столбец 'sample'")
#         sys.exit(1)
    
#     # Подготовка списка для аннотаций
#     annotations = []
    
#     # Обрабатываем каждый запрос
#     print("Начало предсказания...")
#     for idx, sample in enumerate(df['sample']):
#         try:
#             spans = predict_spans(
#                 text=sample,
#                 model=model,
#                 tokenizer=tokenizer,
#                 device=device,
#                 threshold=threshold,
#                 max_length=128
#             )
#             formatted = format_spans(spans)
#             annotations.append(formatted)
            
#             # Выводим прогресс
#             if (idx + 1) % 50 == 0 or idx == len(df) - 1:
#                 print(f"Обработано {idx + 1}/{len(df)} запросов")
                
#         except Exception as e:
#             print(f"Ошибка обработки запроса '{sample}': {str(e)}")
#             annotations.append("[]")  # Пустая аннотация в случае ошибки
    
#     # Создаем результат
#     result_df = pd.DataFrame({
#         'sample': df['sample'],
#         'annotation': annotations
#     })
    
#     # Сохраняем результат
#     try:
#         result_df.to_csv(output_csv, sep=";", index=False)
#         print(f"\nРезультат успешно сохранен в {output_csv}")
#         print(f"Пример первых 3 строк результата:")
#         for i in range(min(3, len(result_df))):
#             print(f"{result_df.iloc[i]['sample']};{result_df.iloc[i]['annotation']}")
#     except Exception as e:
#         print(f"Ошибка сохранения результата: {str(e)}")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()

import os
import sys
import json
import pandas as pd
import torch
from typing import List, Tuple, Dict, Any, Optional
import re

# Добавляем корень проекта в PYTHONPATH для корректного импорта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импортируем необходимые компоненты напрямую
from transformers import AutoTokenizer, AutoConfig
from src.utils import LABELS, logits_to_labels_and_confs, fix_bio_spans
from src.model.modular import ModularTokenClassifier

def correct_empty_spans(spans: List[Tuple[int, int, str]], sample: str) -> List[Tuple[int, int, str]]:
    """
    Корректирует пустой список спанов, возвращая [(0, len(sample), 'O')] для непустого текста.
    Если спаны уже есть или текст пустой, возвращает исходные спаны.
    """
    if not spans and sample and sample.strip():
        return [(0, len(sample), 'O')]
    return spans


def extend_spans_after_prepositions(spans: List[Tuple[int, int, str]], sample: str) -> List[Tuple[int, int, str]]:
    """
    Постобработка спанов: если последний спан не покрывает весь текст, добавляет новые спаны
    для слов после предлогов, используя ту же сущность с префиксом 'I-'. Индексы новых спанов
    сдвигаются на +1 для start и end.
    
    Args:
        spans: Список кортежей (start, end, label).
        sample: Входной текст (запрос).
    
    Returns:
        Обновленный список спанов.
    """
    if not spans or not sample or not sample.strip():
        return spans

    # Проверяем, совпадает ли конец последнего спана с длиной текста
    last_span = spans[-1]
    last_end = last_span[1]
    sample_len = len(sample)

    if last_end >= sample_len or last_span[2] == "O" or "-" not in last_span[2]:
        return spans

    # Извлекаем тип сущности из последнего спана (например, 'TYPE' из 'B-TYPE')
    _, entity = last_span[2].split("-", 1)

    # Список предлогов
    prepositions = ["без", "для", "под", "с", "со", "из", "на", "к", "до", "у"]
    prep_pattern = r'\b(' + '|'.join(prepositions) + r')\b'

    # Находим все предлоги в оставшейся части текста
    remaining_text = sample[last_end:].strip()
    if not remaining_text:
        return spans

    # Разбиваем оставшуюся часть текста на слова, сохраняя пробелы
    new_spans = []
    current_pos = last_end
    words = re.split(r'(\s+)', remaining_text)  # Сохраняем пробелы
    i = 0
    while i < len(words):
        word = words[i]
        if not word.strip():
            # Пропускаем пробелы, но обновляем позицию
            current_pos += len(word)
            i += 1
            continue

        # Проверяем, является ли слово предлогом
        if re.match(prep_pattern, word, re.IGNORECASE):
            # Добавляем предлог как отдельный спан с меткой 'B-<entity>' со сдвигом +1
            start = current_pos + 1
            end = current_pos + len(word) + 1
            if end <= sample_len:  # Проверяем, что не выходим за границы текста
                new_spans.append((start, end, f"B-{entity}"))
            current_pos += len(word)
            i += 1
            # Проверяем, есть ли следующее слово (не пробел)
            if i < len(words) and words[i].strip():
                i += 1  # Пропускаем пробел после предлога
                if i < len(words) and words[i].strip():
                    word = words[i]
                    start = current_pos + 1
                    end = current_pos + len(word) + 1
                    if end <= sample_len:
                        new_spans.append((start, end, f"I-{entity}"))
                    current_pos += len(word)
                    i += 1
        else:
            # Если слово не предлог, добавляем его как 'I-<entity>' со сдвигом +1
            start = current_pos + 1
            end = current_pos + len(word) + 1
            if end <= sample_len:
                new_spans.append((start, end, f"I-{entity}"))
            current_pos += len(word)
            i += 1

    # Объединяем исходные спаны с новыми
    result = spans + new_spans
    # Применяем fix_bio_spans для коррекции префиксов
    result = fix_bio_spans(result, sample)
    return result

def load_model_correctly(model_dir: str):
    """
    Загружает модель с правильной обработкой CRF слоя
    """
    print(f"Загрузка модели из {model_dir} с поддержкой CRF...")
    
    # Загружаем метаданные
    meta_path = os.path.join(model_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise ValueError(f"Файл meta.json не найден в {model_dir}")
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    # Определяем, использовался ли CRF
    use_crf = meta.get("use_crf", False)
    print(f"Обнаружено: use_crf = {use_crf}")
    
    # Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Загружаем config
    config = AutoConfig.from_pretrained(model_dir)
    
    # Добавляем кастомные параметры в config
    config.use_crf = use_crf
    config.num_labels = len(LABELS)
    
    # Загружаем модель с правильной архитектурой
    model = ModularTokenClassifier.from_pretrained(
        model_dir,
        config=config,
        use_crf=use_crf,
        num_labels=len(LABELS)
    )
    
    # Устанавливаем режим инференса
    model.eval()
    
    # Получаем порог из метаданных
    threshold = meta["best_threshold"]["thr"]
    
    return tokenizer, model, threshold

def format_spans(spans: List[Tuple[int, int, str]]) -> str:
    """
    Форматирует список спанов в строку в требуемом формате.
    Пример: [(0, 5, 'B-TYPE'), (6, 9, 'I-TYPE')]
    """
    if not spans:
        return "[]"
    
    formatted = "["
    for i, (start, end, label) in enumerate(spans):
        if i > 0:
            formatted += ", "
        # Сохраняем оригинальные B-/I- метки из предсказания
        formatted += f"({start}, {end}, '{label}')"
    formatted += "]"
    return formatted

def predict_spans(text: str, model, tokenizer, device: str, threshold: float, max_length: int = 128) -> List[Tuple[int, int, str]]:
    """
    Предсказывает спаны для текста с использованием модели
    """
    # Обработка пустых запросов
    if not text or not text.strip():
        return []
    
    # Токенизация
    inputs = tokenizer(
        text, 
        return_offsets_mapping=True, 
        return_tensors="pt", 
        truncation=True, 
        max_length=max_length
    )
    
    # Извлекаем offsets и перемещаем тензоры на устройство
    offsets = inputs.pop("offset_mapping").squeeze().tolist()
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Предсказание
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Обработка логитов
    if hasattr(outputs, "logits"):
        logits = outputs.logits[0]
    elif "logits" in outputs:
        logits = outputs["logits"][0]
    else:
        raise ValueError("Модель не возвращает логиты")
    
    # Преобразование логитов в метки и confidence
    labels, confs = logits_to_labels_and_confs(logits, {i: l for i, l in enumerate(LABELS)})
    
    # Пост-обработка
    spans = []
    current_span = None
    
    for i, (a, b) in enumerate(offsets):
        # Пропускаем специальные токены
        if a == b or i >= len(labels) or i >= len(confs):
            continue
        
        label = labels[i]
        conf = confs[i]
        
        # Обработка O-меток
        if label == "O":
            if current_span:
                spans.append(current_span)
                current_span = None
            continue
        
        # Разбираем метку
        if "-" not in label:
            continue
        
        prefix, entity = label.split("-", 1)
        
        # Начало новой сущности
        if prefix == "B" or current_span is None or current_span["entity"] != entity:
            if current_span:
                spans.append(current_span)
            current_span = {
                "start_index": a,
                "end_index": b,
                "entity": entity,
                "original_label": label,
                "confs": [conf]
            }
        # Продолжение текущей сущности
        elif prefix == "I" and current_span and current_span["entity"] == entity:
            current_span["end_index"] = b
            current_span["confs"].append(conf)
    
    # Добавляем последнюю сущность
    if current_span:
        spans.append(current_span)
    
    # Конвертируем спаны в формат [(start, end, label)]
    formatted_spans = [(span["start_index"], span["end_index"], span["original_label"]) for span in spans]
    
    # Исправляем префиксы B-/I- с помощью fix_bio_spans
    formatted_spans = fix_bio_spans(formatted_spans, text)
    
    # Применяем постобработку для предлогов
    formatted_spans = extend_spans_after_prepositions(formatted_spans, text)
    
    # Фильтрация по порогу
    filtered_spans = []
    for span, orig_span in zip(formatted_spans, spans + [{}] * (len(formatted_spans) - len(spans))):
        # Для новых спанов, добавленных extend_spans_after_prepositions, используем confidence 1.0
        avg_conf = sum(orig_span["confs"]) / len(orig_span["confs"]) if orig_span.get("confs") else 1.0
        if avg_conf >= threshold:
            filtered_spans.append(span)
    
    # Корректировка пустых спанов
    filtered_spans = correct_empty_spans(filtered_spans, text)
    
    return filtered_spans

def logits_to_labels_and_confs(logits: torch.Tensor, id2label: Dict[int, str]):
    """
    Безопасное преобразование логитов в метки и confidence
    """
    # Проверяем форму тензора
    if logits.dim() == 1:
        # Добавляем размерность batch
        logits = logits.unsqueeze(0)
    
    # Применяем softmax
    probs = torch.softmax(logits, dim=-1)
    
    # Получаем предсказания и confidence
    preds = torch.argmax(probs, dim=-1)
    max_probs = torch.max(probs, dim=-1)[0]
    
    # Обрабатываем случай скалярного тензора
    if preds.dim() == 0:
        labels = [id2label[preds.item()]]
        confs = [max_probs.item()]
    else:
        labels = [id2label[p.item()] for p in preds]
        confs = max_probs.tolist()
    
    return labels, confs

def main():
    # Пути к файлам
    model_dir = "/home/ubuntu/x5-ner-base-label-pro/models/v_grok/final_model"
    input_csv = "/home/ubuntu/x5-ner-base-label/data/submission_example.csv"
    output_csv = "/home/ubuntu/x5-ner-base-label-pro/data/subm_grok_prep_pro.csv"
    
    print(f"Загрузка модели из {model_dir}...")
    try:
        tokenizer, model, threshold = load_model_correctly(model_dir)
        print(f"Модель загружена успешно. Используем порог: {threshold:.2f}")
    except Exception as e:
        print(f"Ошибка загрузки модели: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Определяем устройство
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Используем устройство: {device}")
    
    # Считываем входные данные
    print(f"Чтение данных из {input_csv}...")
    try:
        df = pd.read_csv(input_csv, sep=";")
        print(f"Загружено {len(df)} запросов для обработки")
    except Exception as e:
        print(f"Ошибка чтения CSV: {str(e)}")
        sys.exit(1)
    
    # Проверяем структуру входного файла
    if 'sample' not in df.columns:
        print("Ошибка: Входной CSV должен содержать столбец 'sample'")
        sys.exit(1)
    
    # Подготовка списка для аннотаций
    annotations = []
    
    # Обрабатываем каждый запрос
    print("Начало предсказания...")
    for idx, sample in enumerate(df['sample']):
        try:
            spans = predict_spans(
                text=sample,
                model=model,
                tokenizer=tokenizer,
                device=device,
                threshold=threshold,
                max_length=128
            )
            formatted = format_spans(spans)
            annotations.append(formatted)
            
            # Выводим прогресс
            if (idx + 1) % 50 == 0 or idx == len(df) - 1:
                print(f"Обработано {idx + 1}/{len(df)} запросов")
                
        except Exception as e:
            print(f"Ошибка обработки запроса '{sample}': {str(e)}")
            annotations.append("[]")  # Пустая аннотация в случае ошибки
    
    # Создаем результат
    result_df = pd.DataFrame({
        'sample': df['sample'],
        'annotation': annotations
    })
    
    # Сохраняем результат
    try:
        result_df.to_csv(output_csv, sep=";", index=False)
        print(f"\nРезультат успешно сохранен в {output_csv}")
        print(f"Пример первых 3 строк результата:")
        for i in range(min(3, len(result_df))):
            print(f"{result_df.iloc[i]['sample']};{result_df.iloc[i]['annotation']}")
    except Exception as e:
        print(f"Ошибка сохранения результата: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()