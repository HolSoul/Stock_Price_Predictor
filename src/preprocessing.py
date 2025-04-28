import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Functions for data preprocessing

def handle_missing_values(df: pd.DataFrame, strategy: str = 'ffill') -> pd.DataFrame:
    """Обрабатывает пропущенные значения в DataFrame.

    Args:
        df (pd.DataFrame): Входной DataFrame.
        strategy (str): Стратегия заполнения пропусков ('ffill', 'bfill', 'mean', 'median').
                        По умолчанию 'ffill'.

    Returns:
        pd.DataFrame: DataFrame с заполненными пропусками.
    """
    initial_missing = df.isnull().sum().sum()
    if initial_missing == 0:
        print("Пропущенные значения не найдены.")
        return df

    print(f"Обнаружено {initial_missing} пропущенных значений. Применяется стратегия '{strategy}'.")
    
    df_filled = df.copy()

    if strategy == 'ffill':
        df_filled.ffill(inplace=True)
    elif strategy == 'bfill':
        df_filled.bfill(inplace=True)
    elif strategy == 'mean':
        # Заполняем только числовые колонки
        numeric_cols = df_filled.select_dtypes(include=np.number).columns
        df_filled[numeric_cols] = df_filled[numeric_cols].fillna(df_filled[numeric_cols].mean())
    elif strategy == 'median':
        numeric_cols = df_filled.select_dtypes(include=np.number).columns
        df_filled[numeric_cols] = df_filled[numeric_cols].fillna(df_filled[numeric_cols].median())
    else:
        print(f"Предупреждение: Неизвестная стратегия '{strategy}'. Пропуски не заполнены.")
        return df

    # Проверяем, остались ли пропуски (например, в начале при ffill)
    remaining_missing = df_filled.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"Предупреждение: После стратегии '{strategy}' осталось {remaining_missing} пропусков. Применяется bfill.")
        df_filled.bfill(inplace=True) # Заполняем оставшиеся (например, в начале при ffill)

    final_missing = df_filled.isnull().sum().sum()
    if final_missing == 0:
        print("Все пропуски успешно заполнены.")
    else:
         print(f"Предупреждение: Не удалось заполнить все пропуски. Осталось: {final_missing}")

    return df_filled 

def scale_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame | None = None,
    columns_to_scale: list[str] | None = None,
    scaler_type: str = 'minmax'
) -> tuple:
    """Масштабирует указанные колонки в обучающем и тестовом наборах данных.

    Обучает скейлер только на обучающих данных (df_train).

    Args:
        df_train (pd.DataFrame): Обучающий DataFrame.
        df_test (pd.DataFrame | None): Тестовый DataFrame (опционально).
        columns_to_scale (list[str] | None): Список имен колонок для масштабирования.
                                             Если None, масштабируются все числовые колонки.
                                             По умолчанию None.
        scaler_type (str): Тип скейлера ('minmax' или 'standard').
                           По умолчанию 'minmax'.

    Returns:
        tuple: Кортеж, содержащий:
               - df_train_scaled (pd.DataFrame): Масштабированный обучающий DataFrame.
               - df_test_scaled (pd.DataFrame | None): Масштабированный тестовый DataFrame (или None).
               - scaler (object): Обученный объект скейлера.
    """
    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy() if df_test is not None else None

    if columns_to_scale is None:
        # Выбираем все числовые колонки, если список не предоставлен
        columns_to_scale = df_train.select_dtypes(include=np.number).columns.tolist()
        if not columns_to_scale:
             print("Нет числовых колонок для масштабирования.")
             return df_train_scaled, df_test_scaled, None
        print(f"Масштабируются все числовые колонки: {columns_to_scale}")

    # Проверяем наличие колонок в df_train
    missing_cols_train = [col for col in columns_to_scale if col not in df_train.columns]
    if missing_cols_train:
        raise ValueError(f"Колонки {missing_cols_train} отсутствуют в df_train.")
    
    # Проверяем наличие колонок в df_test, если он есть
    if df_test is not None:
        missing_cols_test = [col for col in columns_to_scale if col not in df_test.columns]
        if missing_cols_test:
             raise ValueError(f"Колонки {missing_cols_test} отсутствуют в df_test.")

    # Выбор и обучение скейлера
    if scaler_type.lower() == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type.lower() == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Неизвестный тип скейлера. Используйте 'minmax' или 'standard'.")

    # Обучаем скейлер ТОЛЬКО на обучающих данных
    scaler.fit(df_train[columns_to_scale])

    # Применяем масштабирование
    df_train_scaled[columns_to_scale] = scaler.transform(df_train[columns_to_scale])

    if df_test_scaled is not None:
        df_test_scaled[columns_to_scale] = scaler.transform(df_test[columns_to_scale])

    print(f"Масштабирование ({scaler_type}) применено к колонкам: {columns_to_scale}")

    return df_train_scaled, df_test_scaled, scaler 