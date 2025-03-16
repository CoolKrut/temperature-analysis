import pandas as pd
import numpy as np
import requests
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
def moving_average(city_df):
    city_df['moving_avg'] = city_df['temperature'].rolling(window=30, min_periods=1).mean()
    return city_df


# Функция для расчета статистики по сезонам
def seasonal_stats(city_df):
    seasonal_stats = city_df.groupby('season')['temperature'].agg(['mean', 'std']).reset_index()
    return seasonal_stats


# Функция для выявления аномалий
def detect_anomalies(city_df, seasonal_stats):
    city_df = city_df.merge(seasonal_stats, on='season', how='left')
    city_df['is_anomaly'] = (np.abs(city_df['temperature'] - city_df['mean']) > 2 * city_df['std'])
    return city_df


# Синхронная функция получения текущей температуры через OpenWeatherMap
def temperature_sync(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['main']['temp']
    else:
        st.error(f"Ошибка при получении данных для {city}: {response.text}")
        return None


# Обработка данных для каждого города
def process_city(city):
    city_df = df[df['city'] == city].copy()
    city_df = moving_average(city_df)
    season_stats_df = seasonal_stats(city_df)
    city_df = detect_anomalies(city_df, season_stats_df)
    return city_df, season_stats_df


# Анализ температурных данных
st.title("Анализ температурных данных")

uploaded_file = st.file_uploader("Загрузите CSV-файл с историческими данными", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])

cities = df['city'].unique()

city = st.selectbox("Выберите город", cities)
api_key = st.text_input("Введите API-ключ OpenWeatherMap", type="password")

if st.button("Анализировать данные"):
    start = time.time()
    seq_res = [process_city(city) for city in cities]
    seq_time = time.time() - start

    start = time.time()
    par_res = Parallel(n_jobs=-1)(delayed(process_city)(city) for city in cities)
    par_time = time.time() - start

    st.write(f"### Время выполнения анализа:")
    st.write(f"- Последовательно: {seq_time:.2f} секунд")
    st.write(f"- Параллельно: {par_time:.2f} секунд")

    seq_dfs = [result[0] for result in seq_res]
    season_stats = [result[1] for result in seq_res]

    seq_df = pd.concat(seq_dfs)
    season_stats_df = pd.concat(season_stats)

    st.write("### Описательная статистика")
    st.write(season_stats_df)

    city_data = seq_df[seq_df['city'] == city]

    # График временного ряда температур
    st.write("### Временной ряд температур")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=city_data, x='timestamp', y='temperature', label='Температура', ax=ax, linewidth=2.5)
    if 'moving_avg' in city_data.columns:
        sns.lineplot(data=city_data, x='timestamp', y='moving_avg', label='Скользящее среднее',
                     ax=ax, linestyle='dashed', linewidth=2.5)
    anomaly_data = city_data[city_data['is_anomaly']]
    sns.scatterplot(data=anomaly_data, x='timestamp', y='temperature', color='red',
                    label='Аномалии', ax=ax, s=100, marker='X')
    ax.set_title(f'Температура в {city}', fontsize=16)
    ax.set_xlabel("Дата", fontsize=14)
    ax.set_ylabel("Температура (°C)", fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=12)
    sns.despine()
    st.pyplot(fig)

    # Сезонные профили
    st.write("### Сезонные профили")
    fig, ax = plt.subplots(figsize=(12, 6))
    seasonal_plot = sns.barplot(data=season_stats_df, x='season', y='mean', ax=ax, palette='viridis')
    ax.errorbar(season_stats_df['season'], season_stats_df['mean'], yerr=season_stats_df['std'],
                fmt='none', c='black', capsize=5)
    ax.set_title(f"Сезонные профили температур в {city}", fontsize=16)
    ax.set_xlabel("Сезон", fontsize=14)
    ax.set_ylabel("Средняя температура (°C)", fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    # Проверка текущей температуры
    if api_key:
        current_temp = temperature_sync(city, api_key)
        if current_temp is not None:
            season = city_data['season'].mode()[0]
            stats = season_stats_df[season_stats_df['season'] == season]
            mean_temp = stats['mean'].values[0]
            std_temp = stats['std'].values[0]
            is_anomalous = abs(current_temp - mean_temp) > 2 * std_temp

            st.write(f"### Текущая температура в {city}: {current_temp}°C")
            if is_anomalous:
                st.error("⚠️ Температура аномальная!")
            else:
                st.success("✅ Температура в пределах нормы.")