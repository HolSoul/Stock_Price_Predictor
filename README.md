# Прогнозирование цен акций с использованием временных рядов

Проект для портфолио ML-инженера, демонстрирующий построение и оценку моделей временных рядов для прогнозирования цен акций.

## Описание

Цель проекта - предсказать цену закрытия акции (или индекса) на следующий торговый день на основе исторических данных OHLCV.

## Структура проекта

```
stock-price-prediction/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── evaluate.py
│   ├── predict.py
│   └── visualization.py
├── models/
└── reports/
    └── figures/
```

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/HolSoul/Stock_Price_Predictor
   cd stock-price-prediction
   ```
2. Создайте виртуальное окружение (рекомендуется):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # venv\Scripts\activate  # Windows
   ```
3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

## Использование

(Будет дополнено...)
