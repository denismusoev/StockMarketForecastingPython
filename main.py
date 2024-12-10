import keras
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from keras import Sequential, Input
from keras.src.layers import LSTM, Dense
from keras.src.optimizers import Adam
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import tensorflow as tf

app = Flask(__name__)

def prepare_lstm_data(data, look_back=1):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

@app.route('/predict', methods=['POST'])
def predict():
    global historical_data
    try:
        # Получаем данные из запроса
        data = request.get_json()
        model_type = data.get('model', 'linear_regression')  # По умолчанию Linear Regression
        df = pd.DataFrame(data['data'])
        df = df.sort_values(by='recordDate', ascending=True)

        # Преобразуем даты из строк в формат datetime
        df['recordDate'] = pd.to_datetime(df['recordDate'], errors='coerce')

        # Проверка на некорректные значения в 'recordDate'
        if df['recordDate'].isnull().any():
            return jsonify({"error": "Некорректный формат даты в поле 'recordDate'"}), 400

        # Сортируем данные по дате
        df = df.sort_values('recordDate')

        # Нормализация данных
        scaler = MinMaxScaler(feature_range=(0, 1))
        df['closePrice_scaled'] = scaler.fit_transform(df[['closePrice']])

        # Делим данные на обучающую и тестовую выборки (70% на обучение, 30% на тестирование)
        train_size = int(len(df) * 0.7)
        train, test = df[:train_size], df[train_size:]

        predictions = []
        predicted_test_list = []  # Для предсказанных тестовых данных
        look_back = 3
        historical_data = (df[['recordDate', 'closePrice']])
        rmse = None
        mae = None

        if model_type == 'lstm':
            historical_data = historical_data.tail(len(historical_data[train_size:]) - look_back).to_dict(orient='records')  # Исторические данные
            # Подготовка данных для LSTM
            train_X, train_y = prepare_lstm_data(train[['closePrice_scaled']].values, look_back)
            train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))

            # Создаем и обучаем модель LSTM
            lstm_model = Sequential()
            lstm_model.add(Input(shape=(look_back, 1)))
            lstm_model.add(LSTM(50))
            lstm_model.add(Dense(1))
            lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            lstm_model.fit(train_X, train_y, epochs=10, batch_size=1, verbose=0)

            # Прогнозирование на тестовых данных
            test_X, test_y = prepare_lstm_data(test[['closePrice_scaled']].values, look_back)
            test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))
            predicted_test_scaled = lstm_model.predict(test_X)
            predicted_test = scaler.inverse_transform(predicted_test_scaled)

            # Формируем список предсказанных тестовых данных
            predicted_test_list = [
                {"date": str(test['recordDate'].iloc[i + look_back].date()), "predicted": float(predicted_test[i][0])}
                for i in range(len(predicted_test))
            ]

            # Вычисляем точность
            actual_values = test[['closePrice']].values[look_back:]
            rmse = np.sqrt(mean_squared_error(actual_values, predicted_test))
            mae = mean_absolute_error(actual_values, predicted_test)

            # Рассчитываем RMSE и MAE в процентах от среднего значения
            mean_value = np.mean(actual_values)
            rmse_percent = (rmse / mean_value) * 100
            mae_percent = (mae / mean_value) * 100

            # Прогнозирование на будущее
            last_sequence = train[['closePrice_scaled']].values[-look_back:]
            for i in range(data.get('dayCount', 1)):
                last_sequence_scaled = np.reshape(last_sequence, (1, look_back, 1))
                predicted_scaled = lstm_model.predict(last_sequence_scaled)[0][0]
                predicted_price = scaler.inverse_transform([[predicted_scaled]])[0][0]
                predictions.append({
                    "date": (df['recordDate'].max() + timedelta(days=i + 1)).strftime('%Y-%m-%d'),
                    "predicted_close_price": round(predicted_price, 2)
                })
                # Обновляем последовательность для следующего прогноза
                last_sequence = np.append(last_sequence[1:], [[predicted_scaled]], axis=0)

        else:  # Linear Regression
            historical_data = historical_data.tail(len(historical_data[train_size:])).to_dict(orient='records')  # Исторические данные
            # Подготовка данных для линейной регрессии
            df['days_since_start'] = (df['recordDate'] - df['recordDate'].min()).dt.days
            X = df[['days_since_start']].values
            y = df['closePrice'].values

            # Делим данные для обучения и тестирования
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Обучаем линейную регрессию
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)

            # Прогнозирование на тестовых данных
            predicted_test = lr_model.predict(X_test)
            predicted_test_list = [
                {"date": str(test['recordDate'].iloc[i].date()), "predicted": float(predicted_test[i])}
                for i in range(len(predicted_test))
            ]

            actual_values = test[['closePrice']].values
            rmse = np.sqrt(mean_squared_error(actual_values, predicted_test))
            mae = mean_absolute_error(actual_values, predicted_test)

            # Рассчитываем RMSE и MAE в процентах от среднего значения
            mean_value = np.mean(actual_values)
            rmse_percent = (rmse / mean_value) * 100
            mae_percent = (mae / mean_value) * 100

            # Прогнозирование на будущее
            for i in range(1, data.get('dayCount', 1) + 1):
                future_day = df['days_since_start'].max() + i
                predicted_price = lr_model.predict([[future_day]])[0]
                predictions.append({
                    "date": (df['recordDate'].max() + timedelta(days=i)).strftime('%Y-%m-%d'),
                    "predicted_close_price": round(predicted_price, 2)
                })

        # Формируем результат
        result = {
            "symbol": data.get('symbol', 'Unknown'),
            "type": data.get('type', 'Unknown'),
            "model": model_type,
            "day_count": data.get('dayCount', 1),
            "predictions": predictions,
            "predicted_test": predicted_test_list,  # Прогнозы на тестовых данных
            "historical_data": historical_data,  # Исторические данные
            "rmse": round(rmse_percent, 2) if rmse_percent else None,
            "mae": round(mae_percent, 2) if mae_percent else None
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
