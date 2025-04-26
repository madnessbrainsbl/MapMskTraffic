import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import requests
from scipy.spatial import cKDTree
import branca.colormap as cm
import argparse

# Конфигурация
CONFIG = {
    "data_path": "temp/00-final_dataframe.csv",  # Файл с исходными данными (должен содержать координаты в колонках 'point.lat' и 'point.long' или 'DTP_LATITUDE' и 'DTP_LONGITUDE')
    "output_heatmap": "temp/heatmap_data.csv",
    "map_center": [55.7558, 37.6176],  # Центр Москвы
    "grid_size": 100,
    "yandex_api_key": "ВАШ_API_КЛЮЧ",  # Получите API-ключ на developer.tech.yandex.ru
    # Здесь можно добавить весовые коэффициенты для маршрутизации
    "risk_weight": 0.5,
    "time_weight": 0.3,
    "dist_weight": 0.2,
}

# Загрузка координат (границ области) из конфигурационного файла
def load_coordinate_bounds():
    config_dir = "configs"
    with open(os.path.join(config_dir, "coordinate_bounds.txt"), "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        # Пример строк в файле:
        # Долгота: (37.4,37.95)
        # Широта: (55.5,55.9)
        lon_part = lines[0].split(':')[1].strip().replace('(', '').replace(')', '')
        lon_bounds = tuple(map(float, lon_part.split(',')))
        lat_part = lines[1].split(':')[1].strip().replace('(', '').replace(')', '')
        lat_bounds = tuple(map(float, lat_part.split(',')))
    return lat_bounds, lon_bounds

# Обработка данных ДТП: агрегация на регулярную сетку
def process_accident_data(df, lat_bounds, lon_bounds, n_bins=100):
    # Определяем, какие колонки с координатами использовать
    if 'DTP_LATITUDE' in df.columns and 'DTP_LONGITUDE' in df.columns:
        lat = df['DTP_LATITUDE']
        lon = df['DTP_LONGITUDE']
    elif 'latitude' in df.columns and 'longitude' in df.columns:
        # Денормализуем координаты, если они нормализованы
        lat = df['latitude'] * (lat_bounds[1] - lat_bounds[0]) + lat_bounds[0]
        lon = df['longitude'] * (lon_bounds[1] - lon_bounds[0]) + lon_bounds[0]
    elif 'point.lat' in df.columns and 'point.long' in df.columns:
        lat = df['point.lat']
        lon = df['point.long']
    else:
        raise KeyError("В данных отсутствуют колонки с координатами. Найдены колонки: " + str(df.columns.tolist()))
    lat_bins = np.linspace(lat_bounds[0], lat_bounds[1], n_bins + 1)
    lon_bins = np.linspace(lon_bounds[0], lon_bounds[1], n_bins + 1)
    grid, _, _ = np.histogram2d(lat, lon, bins=[lat_bins, lon_bins])
    return grid

# Создание статичной тепловой карты (PNG)
def create_heatmap(grid):
    plt.figure(figsize=(12, 10))
    sns.heatmap(np.log1p(grid), cmap='YlOrRd', cbar_kws={'label': 'Логарифм вероятности ДТП'})
    plt.gca().invert_yaxis()
    plt.title('Тепловая карта рисков ДТП')
    plt.savefig('heatmap_static.png')
    plt.close()

# Создание интерактивной карты с несколькими слоями (результат – HTML файл)
def create_interactive_map(heatmap_df):
    # Создаем карту с базовым OpenStreetMap тайлом
    m = folium.Map(
        location=CONFIG['map_center'],
        zoom_start=11,
        tiles='OpenStreetMap',
        name='OpenStreetMap'
    )
    # Дополнительный слой Stamen Toner
    folium.TileLayer(
        tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/toner/{z}/{x}/{y}.png',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under ODbL.',
        name='Stamen Toner',
        control=True
    ).add_to(m)
    HeatMap(
        data=heatmap_df[['latitude', 'longitude', 'probability']].values,
        radius=15,
        blur=20,
        gradient={'0.4': 'blue', '0.6': 'lime', '0.8': 'red'},
        min_opacity=0.3,
        max_opacity=0.7
    ).add_to(folium.FeatureGroup(name='Тепловая карта рисков').add_to(m))
    folium.LayerControl().add_to(m)
    colormap = cm.LinearColormap(
        colors=['green', 'yellow', 'red'],
        vmin=heatmap_df['probability'].min(),
        vmax=heatmap_df['probability'].max(),
        caption='Уровень риска'
    )
    colormap.add_to(m)
    m.save('final_map.html')
    return m

# Добавление маршрута на карту (используется результат от Яндекс API)
def add_route_to_map(m, route_data):
    # Извлекаем координаты пути (переключая порядок, чтобы соответствовать (lat, lon))
    coordinates = [
        (point[1], point[0])
        for feature in route_data.get('features', [])
        for point in feature['geometry']['coordinates']
    ]
    # Добавляем линию маршрута
    folium.PolyLine(
        locations=coordinates,
        color='#FF0000',
        weight=3,
        opacity=0.8,
        tooltip=f"Общий риск: {route_data['metadata']['risk_analysis']['total_risk']:.2f}"
    ).add_to(m)
    # Добавляем маркеры начала и конца маршрута
    folium.Marker(
        location=coordinates[0],
        icon=folium.Icon(icon='play', color='green')
    ).add_to(m)
    folium.Marker(
        location=coordinates[-1],
        icon=folium.Icon(icon='stop', color='red')
    ).add_to(m)
    return m

# Вызов маршрутизации через Яндекс API с учетом весов
def calculate_route(start_coord, end_coord, risk_weight, time_weight, dist_weight):
    # Запрос к OSRM для получения реального трека маршрута вдоль дорог
    # https://project-osrm.org/docs/v5.5.1/api/#route-service
    lon1, lat1 = start_coord[1], start_coord[0]
    lon2, lat2 = end_coord[1], end_coord[0]
    osrm_url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
    try:
        resp = requests.get(osrm_url)
        data = resp.json()
        # Координаты трека: список [lon, lat]
        coords = data['routes'][0]['geometry']['coordinates']
        # Собираем в GeoJSON-подобную структуру
        features = [{ 'geometry': { 'coordinates': coords } }]
        route_data = { 'features': features, 'metadata': {} }
        # Анализ риска вдоль пути
        risk_info = analyze_route_risk(route_data)
        # Сохраняем результаты анализа и дополнительные метрики
        route_data['metadata']['risk_analysis'] = risk_info
        route_data['metadata']['distance'] = data['routes'][0]['distance']  # в метрах
        route_data['metadata']['duration'] = data['routes'][0]['duration']  # в секундах
        return route_data
    except Exception as e:
        print(f"Ошибка OSRM маршрутизации: {e}")
        return None

# Анализ риска маршрута с использованием тепловой карты
def analyze_route_risk(route_data):
    heatmap_df = pd.read_csv(CONFIG['output_heatmap'])
    points = heatmap_df[['latitude', 'longitude']].values
    tree = cKDTree(points)
    risks = []
    for feature in route_data.get('features', []):
        for coord in feature['geometry']['coordinates']:
            query_point = [coord[1], coord[0]]
            distances, indices = tree.query(query_point, k=3)
            avg_risk = heatmap_df.iloc[indices]['probability'].mean()
            risks.append(avg_risk)
    if len(risks) == 0:
        risks = [0]
    return {
        'total_risk': float(np.mean(risks)),
        'max_risk': float(np.max(risks)),
        'risk_points': int(sum(1 for r in risks if r > 0.1))
    }

def geocode(address):
    """Геокодирует адрес в координаты (lat, lon) с помощью Nominatim"""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json"}
    resp = requests.get(url, params=params, headers={"User-Agent": "risk_mapper"})
    data = resp.json()
    if not data:
        raise ValueError(f"Не удалось геокодировать адрес: {address}")
    return float(data[0]['lat']), float(data[0]['lon'])

# Основной пайплайн
def main():
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Создание карты и построение маршрута с учётом рисков")
    parser.add_argument("--start", help="Адрес начала маршрута")
    parser.add_argument("--end", help="Адрес конца маршрута")
    parser.add_argument("--risk_weight", type=float, default=CONFIG['risk_weight'], help="Вес риска в оптимизации маршрута")
    parser.add_argument("--time_weight", type=float, default=CONFIG['time_weight'], help="Вес времени в оптимизации маршрута")
    parser.add_argument("--dist_weight", type=float, default=CONFIG['dist_weight'], help="Вес расстояния в оптимизации маршрута")
    args = parser.parse_args()

    # Интерактивный ввод, если адреса не переданы через CLI
    if not args.start:
        args.start = input("Введите адрес начала маршрута: ")
    if not args.end:
        args.end = input("Введите адрес конца маршрута: ")
    # Геокодируем адреса
    start_coord = geocode(args.start)
    end_coord = geocode(args.end)
    print(f"Адрес начала: {args.start} -> {start_coord}")
    print(f"Адрес конца : {args.end} -> {end_coord}")

    # Загрузка и агрегация ДТП
    print("Загрузка данных...")
    lat_bounds, lon_bounds = load_coordinate_bounds()
    df = pd.read_csv(CONFIG['data_path'])
    print("Обработка данных...")
    grid = process_accident_data(df, lat_bounds, lon_bounds, CONFIG['grid_size'])
    grid_normalized = grid / grid.sum() if grid.sum() > 0 else grid

    # Сохранение данных тепловой карты
    print("Сохранение результатов...")
    lon_grid, lat_grid = np.meshgrid(
        np.linspace(lon_bounds[0], lon_bounds[1], CONFIG['grid_size']),
        np.linspace(lat_bounds[0], lat_bounds[1], CONFIG['grid_size'])
    )
    heatmap_df = pd.DataFrame({
        'latitude': lat_grid.flatten(),
        'longitude': lon_grid.flatten(),
        'probability': grid_normalized.flatten()
    })
    heatmap_df.to_csv(CONFIG['output_heatmap'], index=False)

    # Генерация карт
    create_heatmap(grid_normalized)
    m = create_interactive_map(heatmap_df)

    # Построение маршрута по весам
    route = calculate_route(start_coord, end_coord, args.risk_weight, args.time_weight, args.dist_weight)
    if route:
        m = add_route_to_map(m, route)
        m.save('final_map.html')
        print("Карта с маршрутом сохранена: final_map.html")
    else:
        print("Не удалось построить маршрут.")

    print("""
Результаты:
- heatmap_static.png: Статичная тепловая карта
- final_map.html: Интерактивная карта с наложенной тепловой картой и маршрутом
- temp/heatmap_data.csv: Данные для анализа
Дополнительные идеи:
- Добавить фильтрацию по времени, погоде и типу транспорта для более точного расчета рисков.
- Реализовать веб-интерфейс (например, с помощью Flask или Streamlit) для динамического выбора параметров.
- Использовать несколько вариантов маршрутов с последующим сравнением по весовым коэффициентам.
    """)

if __name__ == "__main__":
    main() 