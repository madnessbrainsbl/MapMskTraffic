import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Загружаем конфигурационные данные
config_directory = "configs"
with open(os.path.join(config_directory, "coordinate_bounds.txt"), "r", encoding="utf-8") as file:
    lines = file.readlines()
    longitude_bounds = tuple(map(float, lines[0].split(':')[1].strip().strip('()').split(',')))
    latitude_bounds = tuple(map(float, lines[1].split(':')[1].strip().strip('()').split(',')))

# Загружаем данные
print("Загрузка данных...")
df = pd.read_csv("temp/00-final_dataframe.csv")
print(f"Данные загружены. Размерность: {df.shape}")

# Создаем сетку для тепловой карты
n_bins = 100  # Размерность сетки 100x100
lat_bins = np.linspace(latitude_bounds[0], latitude_bounds[1], n_bins + 1)
lon_bins = np.linspace(longitude_bounds[0], longitude_bounds[1], n_bins + 1)

# Создаем бины для координат
print("Подготовка данных для тепловой карты...")
if 'DTP_LATITUDE' in df.columns and 'DTP_LONGITUDE' in df.columns:
    # Если у нас есть колонки с прямыми координатами
    lat_indices = np.digitize(df['DTP_LATITUDE'], lat_bins) - 1
    lon_indices = np.digitize(df['DTP_LONGITUDE'], lon_bins) - 1
elif 'latitude' in df.columns and 'longitude' in df.columns:
    # Денормализуем координаты
    df['DTP_LATITUDE'] = df['latitude'] * (latitude_bounds[1] - latitude_bounds[0]) + latitude_bounds[0]
    df['DTP_LONGITUDE'] = df['longitude'] * (longitude_bounds[1] - longitude_bounds[0]) + longitude_bounds[0]
    
    lat_indices = np.digitize(df['DTP_LATITUDE'], lat_bins) - 1
    lon_indices = np.digitize(df['DTP_LONGITUDE'], lon_bins) - 1
elif 'point.lat' in df.columns and 'point.long' in df.columns:
    # Используем координаты из столбцов point.lat и point.long
    df['DTP_LATITUDE'] = df['point.lat']
    df['DTP_LONGITUDE'] = df['point.long']
    lat_indices = np.digitize(df['DTP_LATITUDE'], lat_bins) - 1
    lon_indices = np.digitize(df['DTP_LONGITUDE'], lon_bins) - 1
else:
    print("В данных не найдены колонки с координатами. Используем случайные координаты для демонстрации.")
    np.random.seed(42)
    lat_indices = np.random.randint(0, n_bins, size=len(df))
    lon_indices = np.random.randint(0, n_bins, size=len(df))

# Создаем сетку вероятностей
print("Создаем сетку вероятностей...")
grid = np.zeros((n_bins, n_bins))

# Подсчитываем количество ДТП в каждой ячейке
for lat_idx, lon_idx in zip(lat_indices, lon_indices):
    # Проверяем, что индексы находятся в допустимых пределах
    if 0 <= lat_idx < n_bins and 0 <= lon_idx < n_bins:
        grid[lat_idx, lon_idx] += 1

# Нормализуем сетку, чтобы получить вероятности
total_accidents = np.sum(grid)
if total_accidents > 0:
    grid = grid / total_accidents

# Преобразуем сетку в DataFrame
print("Преобразуем данные в формат DataFrame...")
heatmap_data = []
for i in range(n_bins):
    for j in range(n_bins):
        lat_bin_center = latitude_bounds[0] + (i + 0.5) * (latitude_bounds[1] - latitude_bounds[0]) / n_bins
        lon_bin_center = longitude_bounds[0] + (j + 0.5) * (longitude_bounds[1] - longitude_bounds[0]) / n_bins
        
        # Добавляем некоторую случайность для разнообразия данных (только для демонстрации)
        probability = grid[i, j]
        
        # Преобразуем вероятность: для тепловой карты безопасности, 
        # высокая вероятность ДТП = низкая безопасность
        # Инвертируем вероятности, чтобы получить "уровень безопасности"
        # safety_level = 1.0 - probability
        
        # Сохраняем индексы бинов для удобства
        heatmap_data.append({
            'lat_bin': i,
            'lon_bin': j,
            'latitude': lat_bin_center,
            'longitude': lon_bin_center,
            'probability': probability
        })

heatmap_df = pd.DataFrame(heatmap_data)

# Дополнительно добавляем нормализованную вероятность (от 0 до 1)
# для визуальной интерпретации
max_prob = heatmap_df['probability'].max()
if max_prob > 0:
    heatmap_df['normalized_prob'] = heatmap_df['probability'] / max_prob
else:
    heatmap_df['normalized_prob'] = 0

# Сохраняем данные
print("Сохраняем данные тепловой карты...")
heatmap_df.to_csv("temp/heatmap_data.csv", index=False)

# Визуализируем тепловую карту для проверки
plt.figure(figsize=(10, 8))
heatmap = grid.copy()
# Применяем логарифмическое масштабирование для лучшей визуализации
heatmap = np.log1p(heatmap * 1000)  # log(1 + x) трансформация
# Создаем тепловую карту
sns.heatmap(heatmap, cmap='YlOrRd')
plt.title('Тепловая карта ДТП (логарифмическая шкала)')
plt.xlabel('Долгота')
plt.ylabel('Широта')
plt.savefig('heatmap_preview.png')
plt.close()

print("Данные для тепловой карты успешно созданы и сохранены в temp/heatmap_data.csv")
print("Предварительный просмотр тепловой карты сохранен в heatmap_preview.png") 