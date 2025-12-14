# Классификация видов птиц (CUB-200-2011) - zero-shot CLIP

## Цель
- Использовать локальный CUB-200-2011 (200 классов птиц).
- Собрать единый CSV из официальных метаданных (images.txt, image_class_labels.txt, classes.txt, train_test_split.txt).
- Выполнить zero-shot классификацию CLIP на подвыборке.
- Получить confusion matrix и отчёт о качестве.

## Стек
- Python 3.8+, PyTorch, Transformers
- Pandas, NumPy
- Matplotlib, Seaborn, scikit-learn

## Требования
```
torch>=2.0.0
transformers>=4.25.0
pillow>=9.0.0
matplotlib>=3.5.0
pandas>=1.5.0
numpy>=1.23.0
seaborn
scikit-learn
```

## Структура
```
bird-classification/
├── bird_classification_clip.ipynb   # основной ноутбук
├── bird_data.csv                    # полный список изображений CUB (генерируется)
├── bird_data_eval.csv               # подвыборка для прогона (генерируется)
├── CUB_200_2011/                    # локальный датасет (распакован)
│   ├── images/                      # изображения (200 папок по классам)
│   ├── images.txt                   # image_id -> путь
│   ├── image_class_labels.txt       # image_id -> class_id
│   ├── classes.txt                  # class_id -> название класса
│   ├── train_test_split.txt         # image_id -> train/test
│   └── ... (bounding_boxes, parts, README)
├── requirements.txt
└── .gitignore
```

## Порядок работы (ноутбук)
1. Зависимости: `!pip3 install -r requirements.txt`.
2. Подготовка метаданных: чтение `images.txt`, `image_class_labels.txt`, `classes.txt`, `train_test_split.txt`; формирование `bird_data.csv`.
3. Подвыборка: параметры `N_CLASSES`, `IMAGES_PER_CLASS`; сохранение `bird_data_eval.csv`.
4. Оценка CLIP: загрузка `openai/clip-vit-base-patch32`, текстовые промпты из `class_name`, предсказания, confusion matrix, classification report.

## Данные
Папка `CUB_200_2011` должна быть распакована в корне `bird-classification/`. Ноутбук не скачивает изображения и не использует внешние источники.

## Метрики
- Confusion matrix по подвыборке.
- Precision/Recall/F1 по классам.

## Возможные доработки
- Увеличить подвыборку или использовать полный датасет.
- Добавить train/val/test разбиение и дообучение моделей.
- Сохранение эмбеддингов и поиск (Faiss).

Проект оформлен в строгом академическом стиле, без эмодзи и внешних загрузок; весь конвейер работает с локальным CUB-200-2011.

## Источники данных
- **Датасет:** локальный CUB-200-2011 (200 классов птиц)
- **Метаданные:** `images.txt`, `image_class_labels.txt`, `classes.txt`, `train_test_split.txt`
- **Модель:** OpenAI CLIP (`openai/clip-vit-base-patch32`) из Hugging Face

