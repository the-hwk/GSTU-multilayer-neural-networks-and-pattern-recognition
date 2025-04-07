## Использование модели LSTM для прогнозирования простых временных рядов

Для написания и выполнения кода используем платформу [Kaggle](https://www.kaggle.com)

#### Начало:
- слева нажмите на кнопку *Create* (с плюсом)
- в выпадающем списке выберите *New Notebook*
- добавьте датасет в правой части, секция *Input*, кнопка *Add input*
- в поле ввода вставьте ссылку:

```
https://www.kaggle.com/datasets/sudalairajkumar/daily-temperature-of-major-cities
```

Этот датасет содержит исторические данные средних температур в каждый день в различных городах. Из этого датасета мы выберем историю температур в Москве и, соответственно, обучим модель LSTM прогнозировать температуру на 3 дня вперед.

#### Общий алгоритм процесса подготовки данных и обучения модели LSTM

1. Загрузить набор данных
2. Выполнить обработку данных: очистить от шумов и аномалий, выполнить необходимые преобразования
3. Выполнить нормализацию данных (например, в диапазон от 0 до 1)
4. Подготовить данные для модели LSTM в виде последовательностей
5. Разделить набор данных на обучающую и тестовую выборки
6. Сформировать модель LSTM
7. Обучить модель
8. Протестировать на тестовой выборке
9. Визуализировать полученные результаты

#### Написание кода

Нижеприведенный код является небольшим примером. Внимательно изучите его.

1. Загрузить набор данных
```Python
# Загружаем набор данных
df = pd.read_csv('/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv')
```

```Python
# Выводим информацию о датасете
df.info()
```

```Python
# Выводим первые 5 строк датасета
df.head()
```

2. Выполнить обработку данных: очистить от шумов и аномалий, выполнить необходимые преобразования

```Python
# Берем из набора данных только температуру в Москве
df = df[df['City'] == 'Moscow']
target = 'AvgTemperature'
df = df[[target]]
```

```Python
# Проверяем есть ли отсутствующие значения
df.isna().sum()
```

Т.к. данные представлены в виде градусов Фаренгейта, переведем их в градусы Цельсия.

```Python
# Функция для перевода градусов (F в C)
def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5/9

# Применяем эту функцию к столбцу с температурой
df[target] = df[target].apply(fahrenheit_to_celsius)
```

```Python
# Выводим график для визуальной оценки
plt.plot(range(df[target].size), df[target].values)
plt.show()
```

Т.к. данные содержат аномальные выбросы, выполним очистку.

```Python
# Удаляем аномальные данные на основе Z-оценки
threshold = 3
df['z_score'] = (df[target].values - df[target].mean()) / df[target].std()
df_filtered = df[df['z_score'].abs() < threshold]
```

Также выполним сглаживание, чтобы немного облегчить задачу модели LSTM.

```Python
# Сглаживаем данные (убираем шум)
df_filtered[target] = savgol_filter(df_filtered[target].values, window_length=5, polyorder=1)
```

```Python
# Выводим график для визуальной оценки
plt.plot(range(df_filtered[target].size), df_filtered[target].values)
plt.show()
```

```Python
# Данные подготовлены, заменяем переменную df
df = df_filtered
```

3. Выполнить нормализацию данных (например, в диапазон от 0 до 1)

```Python
# Нормализация данных
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[target].values.reshape(-1, 1))
```

4. Подготовить данные для модели LSTM в виде последовательностей

```Python
# Функция для создания последовательностей
def create_multistep_sequences(X, y, input_len=30, output_len=7):
    X_seq, y_seq = [], []
    for i in range(len(X) - input_len - output_len + 1):
        X_seq.append(X[i:i + input_len])  # input_len: последовательность входных данных
        y_seq.append(y[i + input_len:i + input_len + output_len].flatten())  # output_len: несколько значений вперёд
    return np.array(X_seq), np.array(y_seq)
```

```Python
# Для прогнозирования будем использовать 30 исторических значений
input_len = 30
# А прогнозировать будем на 3 значения вперед
output_len = 3

X_seq, y_seq = create_multistep_sequences(scaled_data, scaled_data, input_len, output_len)
```

5. Разделить набор данных на обучающую и тестовую выборки

```Python
# Разделяем данные на обучающие и тестовые (в пропорции 80/20)
split = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]
```

6. Сформировать модель LSTM

```Python
# Формируем модель LSTM
model = Sequential()

# Входной слой (его размер соответствует размеру последовательности)
model.add(Input(shape=(input_len, 1)))

# Слой LSTM с 64 нейронами
model.add(LSTM(64, activation='tanh', return_sequences=False))

# Выходной слой
model.add(Dense(output_len))

# Компилируем модель
model.compile(optimizer='adam', loss='mae')
```

7. Обучить модель

```Python
# Запускаем обучение модели
history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))
```

8. Протестировать на тестовой выборке (в нашем случае просто подадим тестовые данные модели и получим прогноз)

```Python
# Тестируем обученную модель на тестовых данных
last_sequence = X_test

# Даем этот набор обученной модели и получаем прогноз
predicted_scaled = model.predict(last_sequence)
# Масштабируем полученные данные
predicted = scaler.inverse_transform(predicted_scaled)  # Обратная нормализация
```

9. Визуализировать полученные результаты

```Python
# Будем сравнивать с тестовыми ответами
t_in = scaler.inverse_transform(y_test)[:, 0]
t_pr = predicted[:, 0]

# Диапазон значений
start = 0
end = 100

plt.plot(range(start, end), t_in[start:end], label='Исходные')
plt.plot(range(start, end), t_pr[start:end], label='Прогноз')

plt.legend()
plt.title(f"Данные в диапазоне {start}-{end}")
plt.xlabel("День")
plt.ylabel("Температура")
plt.show()
```

Модель готова к использованию. Можно поэкспериментировать: взять данные по другим городам и странам.