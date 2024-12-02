from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Получаем данные из запроса
        data = request.get_json()
        df = pd.DataFrame(data['data'])

        # Преобразуем даты из строк в формат datetime
        df['recordDate'] = pd.to_datetime(df['recordDate'], errors='coerce')

        # Проверка на некорректные значения в 'recordDate'
        if df['recordDate'].isnull().any():
            return jsonify({"error": "Некорректный формат даты в поле 'recordDate'"}), 400

        # Сортируем данные по дате
        df = df.sort_values('recordDate')

        # Добавляем столбец с количеством дней с начала (для линейной регрессии)
        df['days_since_start'] = (df['recordDate'] - df['recordDate'].min()).dt.days

        # Добавляем новые параметры
        df['price_difference'] = df['closePrice'] - df['openPrice']  # Разница между ценой закрытия и открытия

        # Нормализация признаков
        scaler = MinMaxScaler()
        df['days_since_start_scaled'] = scaler.fit_transform(df[['days_since_start']])

        # Применяем скользящее среднее для краткосрочных прогнозов
        df['price_difference'] = df['price_difference'].rolling(window=5).mean().fillna(0)  # скользящее среднее за последние 5 дней

        # Подготовка данных для регрессии
        X = df[['days_since_start_scaled', 'price_difference']].values
        y = df['closePrice'].values

        # Создаем и обучаем модель линейной регрессии
        model = LinearRegression()
        model.fit(X, y)

        # Получаем количество предсказаний (dayCount)
        day_count = data.get('dayCount', 1)  # По умолчанию прогнозируем на 1 день

        # Генерация прогнозов
        predictions = []
        for i in range(1, day_count + 1):
            future_day = df['days_since_start'].max() + i
            future_price_difference = df['price_difference'].mean()  # Используем среднюю разницу

            # Нормализуем future_day
            future_day_scaled = scaler.transform([[future_day]])[0][0]

            # Прогнозируем цену
            predicted_price = model.predict([[future_day_scaled, future_price_difference]])[0]

            # Формируем дату прогноза
            prediction_date = df['recordDate'].max() + timedelta(days=i)

            predictions.append({
                "date": prediction_date.strftime('%Y-%m-%d'),
                "predicted_close_price": round(predicted_price, 2)
            })

        # Формируем результат
        result = {
            "symbol": data.get('symbol', 'Unknown'),
            "type": data.get('type', 'Unknown'),
            "day_count": day_count,
            "predictions": predictions
        }

        # Добавляем текстовое описание прогноза
        trend_direction = "растет" if predictions[-1]['predicted_close_price'] > predictions[0]['predicted_close_price'] else "падает"
        result_text = f"Прогноз на {day_count} дня(ей): цена {trend_direction}."

        # Включаем поле result с описанием
        result['result'] = result_text

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
