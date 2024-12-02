import mysql.connector
import json

# Конфигурация подключения к базе данных
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "1234",
    "database": "stock_prediction_db"
}

# Подключение к базе данных
def connect_to_db(config):
    try:
        connection = mysql.connector.connect(**config)
        if connection.is_connected():
            print("Соединение с базой данных успешно установлено!")
        return connection
    except mysql.connector.Error as err:
        print(f"Ошибка подключения: {err}")
        return None

# Вставка данных в таблицу MySQL
def insert_data_into_db(connection, data):
    try:
        cursor = connection.cursor()

        # SQL-запрос для вставки данных
        insert_query = """
        INSERT INTO time_series_data (symbol, record_date, open_price, high_price, low_price, close_price, volume, data_type)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """

        # Парсинг JSON и подготовка данных
        symbol = data["Meta Data"]["2. Symbol"]
        time_series = data["Time Series (Daily)"]

        for record_date, daily_data in time_series.items():
            open_price = float(daily_data["1. open"])
            high_price = float(daily_data["2. high"])
            low_price = float(daily_data["3. low"])
            close_price = float(daily_data["4. close"])
            volume = int(daily_data["5. volume"])
            data_type = "DAILY"

            # Подготовка к вставке
            values = (symbol, record_date, open_price, high_price, low_price, close_price, volume, data_type)

            # Выполнение запроса
            cursor.execute(insert_query, values)

        # Фиксация изменений
        connection.commit()
        print(f"{cursor.rowcount} записей успешно вставлено.")

    except mysql.connector.Error as err:
        print(f"Ошибка выполнения запроса: {err}")
    finally:
        cursor.close()

# Чтение JSON-файла
def read_json_file(file_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            print("JSON-файл успешно прочитан.")
            return data
    except Exception as e:
        print(f"Ошибка чтения JSON-файла: {e}")
        return None

# Основной процесс
if __name__ == "__main__":
    # Укажите путь к JSON-файлу
    json_file_path = "data.json"  # Замените на ваш путь к файлу

    # Читаем JSON-данные из файла
    json_data = read_json_file(json_file_path)

    if json_data:
        # Подключаемся к базе данных
        connection = connect_to_db(db_config)
        if connection:
            # Вставляем данные в базу
            insert_data_into_db(connection, json_data)
            # Закрываем соединение
            connection.close()
