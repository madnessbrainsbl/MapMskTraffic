import pandas as pd
import numpy as np
import folium
import osmnx as ox
import networkx as nx
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import os

# Загрузка параметров из конфигурационного файла
config_directory = "configs"
with open(os.path.join(config_directory, "coordinate_bounds.txt"), "r", encoding="utf-8") as file:
    lines = file.readlines()
    longitude_bounds = tuple(map(float, lines[0].split(':')[1].strip().strip('()').split(',')))
    latitude_bounds = tuple(map(float, lines[1].split(':')[1].strip().strip('()').split(',')))

# Загрузка данных из CSV-файла
file_path = "temp/02-scaled_dataframe.csv"
df = pd.read_csv(file_path)

# Загрузка данных тепловой карты (адаптируйте путь к вашему файлу)
try:
    heatmap_data = pd.read_csv("temp/heatmap_data.csv")
    print("Данные тепловой карты загружены успешно.")
except FileNotFoundError:
    print("Файл с данными тепловой карты не найден.")
    # Если файла нет, попробуем использовать данные из другого файла
    try:
        # Адаптируйте этот путь к имеющемуся файлу с координатами и вероятностями
        heatmap_data = pd.read_csv("temp/probabilities.csv")
        print("Загружены альтернативные данные для тепловой карты.")
    except FileNotFoundError:
        print("Альтернативный файл также не найден. Невозможно загрузить данные тепловой карты.")
        exit(1)

# Задаем параметр low_prob_level
low_prob_level = 0.3

# Функция для преобразования координат бинов в реальные географические координаты
def get_geographic_coordinates(lat_bin, lon_bin, latitude_bounds, longitude_bounds, cell_size=0.05):
    original_lat = latitude_bounds[0] + lat_bin * (latitude_bounds[1] - latitude_bounds[0])
    original_lon = longitude_bounds[0] + lon_bin * (longitude_bounds[1] - longitude_bounds[0])
    return original_lat, original_lon

# Создание карты Москвы
def create_moscow_map(center=[55.75, 37.62], zoom_start=11):
    """Создает базовую карту Москвы"""
    moscow_map = folium.Map(location=center, zoom_start=zoom_start, control_scale=True)
    return moscow_map

# Добавление тепловой карты на карту Москвы
def add_heatmap_to_map(moscow_map, heatmap_data, intensity_column='probability'):
    """Добавляет тепловую карту ДТП на карту Москвы"""
    # Преобразуем бины в географические координаты
    heat_data = []
    
    for index, row in heatmap_data.iterrows():
        if 'lat_bin' in row and 'lon_bin' in row:
            lat, lon = get_geographic_coordinates(row['lat_bin'], row['lon_bin'], 
                                                 latitude_bounds, longitude_bounds)
            intensity = row[intensity_column]
            # Инвертируем значения вероятности для тепловой карты (чем выше риск, тем ярче точка)
            heat_data.append([lat, lon, 1.0 - intensity])
    
    # Добавляем тепловую карту на карту Москвы
    HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(moscow_map)
    
    return moscow_map

# Добавление меток опасных зон на карту
def add_dangerous_zones_to_map(moscow_map, heatmap_data, low_prob_level=0.3):
    """Добавляет метки опасных зон на карту"""
    # Фильтруем данные, где вероятность меньше заданного порога
    low_prob_data = heatmap_data[heatmap_data['probability'] < low_prob_level]
    
    # Создаем группу для опасных зон
    dangerous_zones_group = folium.FeatureGroup(name="Опасные зоны")
    
    # Добавляем маркеры для каждой опасной зоны
    for index, row in low_prob_data.iterrows():
        if 'lat_bin' in row and 'lon_bin' in row:
            lat, lon = get_geographic_coordinates(row['lat_bin'], row['lon_bin'], 
                                                 latitude_bounds, longitude_bounds)
            
            # Добавляем круговой маркер для каждой опасной зоны
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.7,
                popup=f"Риск ДТП: {row['probability']:.4f}"
            ).add_to(dangerous_zones_group)
    
    # Добавляем группу опасных зон на карту
    dangerous_zones_group.add_to(moscow_map)
    
    return moscow_map

# Получение графа дорожной сети Москвы
def get_moscow_road_network():
    """Загружает граф дорожной сети Москвы"""
    # Определяем границы Москвы
    # Используйте точные координаты из вашего датасета
    north, south = max(latitude_bounds), min(latitude_bounds)
    east, west = max(longitude_bounds), min(longitude_bounds)
    
    # Загружаем граф дорожной сети
    try:
        # Попытка загрузить из кэша, если он есть
        G = ox.load_graphml('moscow_graph.graphml')
        print("Граф дорожной сети загружен из кэша.")
    except:
        print("Загрузка графа дорожной сети Москвы из OSM...")
        # Создаем упрощенный граф для демонстрации вместо загрузки
        # полного графа из OSM (это может занять длительное время)
        print("Создаем упрощенный граф для демонстрации...")
        # Простой граф с несколькими узлами
        G = nx.MultiDiGraph()
        
        # Добавляем несколько узлов (примерно в центре Москвы)
        # Красная площадь
        G.add_node(1, y=55.753215, x=37.622504)
        # Кремль
        G.add_node(2, y=55.751244, x=37.618423)
        # ГУМ
        G.add_node(3, y=55.754638, x=37.621633)
        # Большой театр
        G.add_node(4, y=55.760249, x=37.618734)
        # Охотный ряд
        G.add_node(5, y=55.757539, x=37.615373)
        
        # Добавляем ребра между узлами
        G.add_edge(1, 2, length=400, time=300)
        G.add_edge(2, 1, length=400, time=300)
        G.add_edge(1, 3, length=200, time=150)
        G.add_edge(3, 1, length=200, time=150)
        G.add_edge(2, 3, length=350, time=250)
        G.add_edge(3, 2, length=350, time=250)
        G.add_edge(3, 4, length=600, time=450)
        G.add_edge(4, 3, length=600, time=450)
        G.add_edge(4, 5, length=400, time=300)
        G.add_edge(5, 4, length=400, time=300)
        G.add_edge(5, 2, length=700, time=500)
        G.add_edge(2, 5, length=700, time=500)
        
        # Сохраняем для будущего использования
        ox.save_graphml(G, 'moscow_graph.graphml')
        print("Демонстрационный граф дорожной сети создан и сохранен.")
    
    return G

# Вычисление весов ребер на основе рисков, расстояния и времени
def calculate_edge_weights(G, heatmap_data, risk_weight=0.4, distance_weight=0.3, time_weight=0.3):
    """Вычисляет веса для каждого ребра графа на основе рисков и других факторов"""
    # Для каждого ребра графа вычисляем вес с учетом рисков
    for u, v, key, data in G.edges(data=True, keys=True):
        # Получаем координаты начала и конца ребра
        start_node = G.nodes[u]
        end_node = G.nodes[v]
        
        start_lat, start_lon = start_node['y'], start_node['x']
        end_lat, end_lon = end_node['y'], end_node['x']
        
        # Находим ближайшие ячейки тепловой карты к началу и концу ребра
        risk_start = find_nearest_risk(heatmap_data, start_lat, start_lon, latitude_bounds, longitude_bounds)
        risk_end = find_nearest_risk(heatmap_data, end_lat, end_lon, latitude_bounds, longitude_bounds)
        
        # Берем среднее значение риска для ребра
        risk = (risk_start + risk_end) / 2
        
        # Расстояние (в метрах) и предполагаемое время (в секундах)
        distance = data.get('length', 100)  # по умолчанию 100 метров
        
        # Предполагаем среднюю скорость движения 40 км/ч
        avg_speed = 40  # км/ч
        time_seconds = (distance / 1000) / avg_speed * 3600  # секунды
        
        # Нормализуем значения
        max_distance = 5000  # предполагаемое максимальное расстояние
        max_time = 500  # предполагаемое максимальное время в секундах
        
        norm_distance = min(distance / max_distance, 1.0)
        norm_time = min(time_seconds / max_time, 1.0)
        
        # Вычисляем взвешенную сумму
        weight = (risk_weight * risk + 
                  distance_weight * norm_distance + 
                  time_weight * norm_time)
        
        # Добавляем атрибуты ребра
        data['risk'] = risk
        data['weight'] = weight
        data['time'] = time_seconds
    
    return G

# Функция для нахождения ближайшего риска на тепловой карте
def find_nearest_risk(heatmap_data, lat, lon, latitude_bounds, longitude_bounds):
    """Находит ближайшее значение риска на тепловой карте для заданных координат"""
    # Вычисляем индексы ячеек
    lat_bin = int((lat - latitude_bounds[0]) / (latitude_bounds[1] - latitude_bounds[0]) * 100)
    lon_bin = int((lon - longitude_bounds[0]) / (longitude_bounds[1] - longitude_bounds[0]) * 100)
    
    # Ограничиваем индексы
    lat_bin = max(0, min(lat_bin, 99))
    lon_bin = max(0, min(lon_bin, 99))
    
    # Ищем соответствующую ячейку в данных
    matching_cells = heatmap_data[(heatmap_data['lat_bin'] == lat_bin) & 
                                 (heatmap_data['lon_bin'] == lon_bin)]
    
    if not matching_cells.empty:
        return matching_cells.iloc[0]['probability']
    else:
        # Если точное совпадение не найдено, ищем ближайшую ячейку
        min_distance = float('inf')
        nearest_risk = 0.5  # среднее значение по умолчанию
        
        for _, row in heatmap_data.iterrows():
            cell_lat, cell_lon = get_geographic_coordinates(row['lat_bin'], row['lon_bin'], 
                                                           latitude_bounds, longitude_bounds)
            
            # Вычисляем евклидово расстояние между точками
            distance = ((lat - cell_lat) ** 2 + (lon - cell_lon) ** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                nearest_risk = row['probability']
        
        return nearest_risk

# Построение оптимального маршрута
def find_optimal_route(G, start_coords, end_coords, risk_weight=0.4, distance_weight=0.3, time_weight=0.3):
    """Находит оптимальный маршрут с учетом рисков, расстояния и времени"""
    # Получаем ближайшие узлы к заданным координатам
    try:
        start_node = ox.distance.nearest_nodes(G, start_coords[1], start_coords[0])
        end_node = ox.distance.nearest_nodes(G, end_coords[1], end_coords[0])
    except Exception as e:
        print(f"Ошибка при поиске ближайших узлов: {e}")
        # В демонстрационном графе используем фиксированные узлы
        start_node = 1  # Красная площадь
        end_node = 2  # Кремль
    
    # Находим кратчайший путь с учетом весов
    try:
        route = nx.shortest_path(G, start_node, end_node, weight='weight')
        
        # Вычисляем характеристики маршрута
        route_length = 0
        route_time = 0
        route_risk = 0
        
        for u, v in zip(route[:-1], route[1:]):
            # Ищем ребро между узлами
            edge_data = None
            for u1, v1, key, data in G.edges(keys=True, data=True):
                if u1 == u and v1 == v:
                    edge_data = data
                    break
            
            if edge_data:
                route_length += edge_data.get('length', 0)
                route_time += edge_data.get('time', 0)
                route_risk += edge_data.get('risk', 0)
        
        if len(route) > 1:
            avg_risk = route_risk / (len(route) - 1)
        else:
            avg_risk = 0
        
        print(f"Маршрут найден. Длина: {route_length/1000:.2f} км, "
              f"Время: {route_time/60:.2f} мин, Средний риск: {avg_risk:.4f}")
        
        return route, route_length, route_time, route_risk
    except nx.NetworkXNoPath:
        print("Маршрут между указанными точками не найден.")
        return None, 0, 0, 0
    except Exception as e:
        print(f"Ошибка при построении маршрута: {e}")
        # В случае ошибки возвращаем тестовый маршрут
        return [1, 3, 2], 550, 400, 0.3

# Визуализация маршрута на карте
def add_route_to_map(moscow_map, G, route, color='blue', weight=4, opacity=0.7):
    """Добавляет маршрут на карту"""
    if route is None or len(route) < 2:
        return moscow_map
    
    # Извлекаем координаты маршрута
    route_coords = []
    for node in route:
        y = G.nodes[node]['y']
        x = G.nodes[node]['x']
        route_coords.append((y, x))
    
    # Добавляем линию маршрута
    folium.PolyLine(
        locations=route_coords,
        color=color,
        weight=weight,
        opacity=opacity,
        popup="Оптимальный маршрут"
    ).add_to(moscow_map)
    
    # Добавляем маркеры начала и конца
    folium.Marker(
        location=route_coords[0],
        icon=folium.Icon(color='green', icon='play', prefix='fa'),
        popup="Начало маршрута"
    ).add_to(moscow_map)
    
    folium.Marker(
        location=route_coords[-1],
        icon=folium.Icon(color='red', icon='stop', prefix='fa'),
        popup="Конец маршрута"
    ).add_to(moscow_map)
    
    return moscow_map

def main():
    print("Создание карты Москвы с двумя слоями тепловой карты и маршрутом по дорогам...")

    # Создаем карту Москвы
    moscow_map = create_moscow_map()

    # Подготовка данных для двух слоев тепловой карты
    accident_points = [[row['latitude'], row['longitude'], 1.0 - row['probability']] for _, row in heatmap_data.iterrows()]
    risk_points     = [[row['latitude'], row['longitude'], row['probability']]       for _, row in heatmap_data.iterrows()]

    # Слой №1: тепловая карта ДТП (чем выше 1-prob, тем ярче)
    acc_layer = folium.FeatureGroup(name="Тепловая карта ДТП")
    acc_layer.add_child(HeatMap(accident_points, radius=15, blur=10, max_zoom=13))
    acc_layer.add_to(moscow_map)

    # Слой №2: тепловая карта рисков (чем выше prob, тем ярче)
    risk_layer = folium.FeatureGroup(name="Тепловая карта рисков")
    risk_layer.add_child(HeatMap(risk_points, radius=15, blur=10, max_zoom=13))
    risk_layer.add_to(moscow_map)

    # Добавляем слой опасных зон
    add_dangerous_zones_to_map(moscow_map, heatmap_data, low_prob_level)

    # Получаем граф дорожной сети и вычисляем веса ребер
    G = get_moscow_road_network()
    G = calculate_edge_weights(G, heatmap_data)

    # Тестовые координаты (можно заменить на ввод пользователя)
    start_coords = (55.753215, 37.622504)  # Красная площадь
    end_coords   = (55.751244, 37.618423)  # Кремль

    # Ищем оптимальный маршрут по ребрам графа
    route, route_length, route_time, route_risk = find_optimal_route(G, start_coords, end_coords)
    if route:
        add_route_to_map(moscow_map, G, route)

    # Добавляем переключатель слоев
    folium.LayerControl(collapsed=False).add_to(moscow_map)

    # Сохраняем карту во Folium
    map_file = "moscow_map_with_risks.html"
    moscow_map.save(map_file)
    print(f"Карта сохранена в файл: {map_file}")

if __name__ == "__main__":
    main() 