# Functions for loading data 

import os
import pandas as pd
import yfinance as yf
from datetime import datetime

RAW_DATA_DIR = "data/raw"


def load_stock_data(
    ticker: str,
    start_date: str = "2010-01-01",
    end_date: str | None = None,
    save_raw: bool = True,
) -> pd.DataFrame | None:
    """Загружает исторические данные OHLCV для указанного тикера
    с Yahoo Finance и опционально сохраняет их в CSV.

    Args:
        ticker (str): Тикер акции или индекса (например, 'AAPL', '^GSPC').
        start_date (str): Дата начала загрузки в формате 'YYYY-MM-DD'.
                          По умолчанию '2010-01-01'.
        end_date (str | None): Дата окончания загрузки в формате 'YYYY-MM-DD'.
                               Если None, используется текущая дата.
                               По умолчанию None.
        save_raw (bool): Сохранять ли загруженные данные в папку data/raw/.
                         По умолчанию True.

    Returns:
        pd.DataFrame | None: DataFrame с данными OHLCV,
                             индексированный по дате, или None в случае ошибки.
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    try:
        print(f"Загрузка данных для {ticker} с {start_date} по {end_date}...")
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            print(f"Ошибка: Не удалось загрузить данные для тикера {ticker}.")
            return None

        # Проверка и создание директории для сохранения
        if save_raw:
            os.makedirs(RAW_DATA_DIR, exist_ok=True)
            # Используем только дату, без времени, для имени файла
            safe_ticker = ticker.replace("^", "") # Убираем '^' для имени файла
            filename = f"{safe_ticker}_{start_date}_{end_date}_raw.csv"
            filepath = os.path.join(RAW_DATA_DIR, filename)
            data.to_csv(filepath)
            print(f"Данные сохранены в {filepath}")

        return data

    except Exception as e:
        print(f"Произошла ошибка при загрузке данных для {ticker}: {e}")
        return None


# Пример использования (можно раскомментировать для теста)
# if __name__ == "__main__":
#     aapl_data = load_stock_data("AAPL")
#     if aapl_data is not None:
#         print("\nAAPL Data Head:")
#         print(aapl_data.head())
#
#     sp500_data = load_stock_data("^GSPC")
#     if sp500_data is not None:
#         print("\nS&P 500 Data Head:")
#         print(sp500_data.head()) 