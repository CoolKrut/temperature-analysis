import pandas as pd
import numpy as np
import requests
import aiohttp
import asyncio
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import time


# Загрузка данных
@st.cache_data
def load_data():
    return pd.read_csv('temperature_data.csv', parse_dates=['timestamp'])


df = load_data()


# Функция для вычисления скользящего среднего
def moving_avg(city_df):
    city_df['moving_avg'] = city_df['temperature'].rolling(window=30, min_periods=1).mean()
    return city_df


# Функция для расчета статистики по сезонам
def seasonal_stats(city_df):
    stats = city_df.groupby('season')['temperature'].agg(['mean', 'std']).reset_index()
    stats.rename(columns={
        'season': 'Сезон',
        'mean': 'Средняя температура (°C)',
        'std': 'Стандартное отклонение'
    }, inplace=True)
    return stats


# Функция для выявления аномалий
def detect_anomalies(city_df, stats):
    city_df = city_df.merge(stats, left_on='season', right_on='Сезон', how='left')
    city_df['is_anomaly'] = (np.abs(city_df['temperature'] - city_df['Средняя температура (°C)']) > 2 * city_df[
        'Стандартное отклонение'])
    return city_df


# Синхронная функция получения текущей температуры через OpenWeatherMap
def temp_sync(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['main']['temp']
    else:
        st.error(f"Ошибка для {city}: {response.text}")
        return None


# Асинхронная функция получения текущей температуры через OpenWeatherMap
async def temp_async(session, city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={api_key}"
    async with session.get(url) as response:
        if response.status == 200:
            data = await response.json()
            return data['main']['temp']
        else:
            error_message = await response.text()
            st.error(f"Ошибка для {city}: {error_message}")
            return None


# Обработка данных для каждого города
def process_city(city):
    city_df = df[df['city'] == city].copy()
    city_df = moving_avg(city_df)
    stats = seasonal_stats(city_df)
    city_df = detect_anomalies(city_df, stats)
    return city_df, stats


# Стримлит приложение
st.title("Анализ температурных данных")

uploaded_file = st.file_uploader("Загрузите CSV-файл с погодными данными", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])

cities = df['city'].unique()

city = st.selectbox("Выберите город", cities)
api_key = st.text_input("Введите API-ключ", type="password")

if st.button("Анализировать данные"):
    # Параллельный анализ
    start_time = time.time()
    seq_res = [process_city(city) for city in cities]
    seq_time = time.time() - start_time

    start_time = time.time()
    par_res = Parallel(n_jobs=-1)(delayed(process_city)(city) for city in cities)
    par_time = time.time() - start_time

    st.write(f"### Время выполнения анализа:")
    st.write(f"- Последовательно: {seq_time:.2f} секунд")
    st.write(f"- Параллельно: {par_time:.2f} секунд")

    seq_dfs = [res[0] for res in seq_res]
    stats_dfs = [res[1] for res in seq_res]

    seq_df = pd.concat(seq_dfs)
    stats_df = pd.concat(stats_dfs)

    st.write("### Статистика по сезонам")
    st.write(stats_df)

    city_data = seq_df[seq_df['city'] == city]

    # График временного ряда
    st.write("### Временной ряд температур")
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(city_data['timestamp'], city_data['temperature'], color='blue',
            linewidth=1.5, label='Фактическая температура')

    anomaly_data = city_data[city_data['is_anomaly']]
    ax.scatter(anomaly_data['timestamp'], anomaly_data['temperature'], color='red', s=100, marker='X', label='Аномалии')
    ax.set_title(f"Временной ряд температур в {city}", fontsize=16)
    ax.set_xlabel("Дата", fontsize=14)
    ax.set_ylabel("Температура (°C)", fontsize=14)
    ax.legend(fontsize=12)
    sns.despine()
    st.pyplot(fig)

    # Сезонные профили
    st.write("### Сезонные профили")
    prof_df = stats_df.copy()
    prof_df['Минимальная температура (°C)'] = (
            prof_df['Средняя температура (°C)'] -
            2 * prof_df['Стандартное отклонение'])
    prof_df['Максимальная температура (°C)'] = (
            prof_df['Средняя температура (°C)'] +
            2 * prof_df['Стандартное отклонение'])

    # График сезонных профилей
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_pos = np.arange(len(prof_df))
    bar_w = 0.3

    ax.bar(bar_pos - bar_w, prof_df['Минимальная температура (°C)'], width=bar_w,
           label='Минимальная температура', color='lightblue', edgecolor='black')
    ax.bar(bar_pos, prof_df['Средняя температура (°C)'], width=bar_w,
           label='Средняя температура', color='orange', edgecolor='black')
    ax.bar(bar_pos + bar_w, prof_df['Максимальная температура (°C)'], width=bar_w,
           label='Максимальная температура', color='red', edgecolor='black')

    ax.set_xticks([])
    ax.set_title(f"Сезонные профили температур в {city}", fontsize=16)
    ax.set_xlabel("Сезоны", fontsize=14)
    ax.set_ylabel("Температура (°C)", fontsize=14)
    ax.legend(fontsize=12)
    sns.despine()
    st.pyplot(fig)

    if api_key:

        # Синхронный запрос
        sync_start = time.time()
        curr_temp_sync = temp_sync(city, api_key)
        sync_time = time.time() - sync_start


        # Асинхронный запрос
        async def fetch_async_temp():
            async with aiohttp.ClientSession() as session:
                return await temp_async(session, city, api_key)


        async_start = time.time()
        curr_temp_async = asyncio.run(fetch_async_temp())
        async_time = time.time() - async_start

        st.write(f"### Время выполнения запроса к API:")
        st.write(f"- Синхронно: {sync_time:.2f} секунд")
        st.write(f"- Асинхронно: {async_time:.2f} секунд")

        # Проверка аномальности температуры
        if curr_temp_sync is not None or curr_temp_async is not None:
            curr_temp = curr_temp_sync or curr_temp_async
            season = city_data['season'].mode()[0]
            stats = stats_df[stats_df['Сезон'] == season]
            mean_temp = stats['Средняя температура (°C)'].values[0]
            std_temp = stats['Стандартное отклонение'].values[0]
            is_anomalous = abs(curr_temp - mean_temp) > 2 * std_temp

            st.write(f"### Текущая температура в {city}: {curr_temp}°C")
            if is_anomalous:
                st.error("⚠️Температура аномальная!")
            else:
                st.success("✅Температура в пределах нормы.")