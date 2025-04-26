import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import requests
from scipy.spatial import cKDTree

# Конфигурация
CONFIG = {
    'data_path': 'temp/00-final_dataframe.csv',
    'heatmap_path': 'temp/heatmap_data.csv',
    'yandex_api_key': None  # не нужен, используем OSRM и Nominatim
}

# Загружаем тепловые данные при старте
heat_df = pd.read_csv(CONFIG['heatmap_path'])
# Добавляем сырые координаты ДТП
acc_df = pd.read_csv(CONFIG['data_path'])
if 'DTP_LATITUDE' in acc_df.columns and 'DTP_LONGITUDE' in acc_df.columns:
    accident_points = acc_df[['DTP_LATITUDE','DTP_LONGITUDE']].dropna().values.tolist()
elif 'point.lat' in acc_df.columns and 'point.long' in acc_df.columns:
    accident_points = acc_df[['point.lat','point.long']].dropna().values.tolist()
else:
    accident_points = []
# Используем нормализованную вероятность (0-1) для визуализации и расчёта риска
if 'normalized_prob' in heat_df.columns:
    heat_key = 'normalized_prob'
else:
    heat_key = 'probability'
heat_points = heat_df[['latitude', 'longitude', heat_key]].values.tolist()
# Максимум нормализованной вероятности равен 1
HEAT_MAX = heat_df[heat_key].max()
tree = cKDTree(heat_df[['latitude', 'longitude']].values)

app = Flask(__name__)


def load_moscow_bounds():
    # Здесь используем жестко заданные границы Москвы
    return (55.5, 55.9), (37.3, 37.95)

# Загрузка координат области Москвы
lat_bounds, lon_bounds = load_moscow_bounds()


def geocode(address):
    url = 'https://nominatim.openstreetmap.org/search'
    params = {
        'q': address, 'format': 'json',
        'viewbox': f"{lon_bounds[0]},{lat_bounds[1]},{lon_bounds[1]},{lat_bounds[0]}",
        'bounded': 1,
        'limit': 5
    }
    resp = requests.get(url, params=params, headers={'User-Agent': 'risk_mapper'})
    data = resp.json()
    if not data:
        raise ValueError(f'Не удалось геокодировать адрес: {address}')
    return float(data[0]['lat']), float(data[0]['lon'])


def analyze_route_risk(coords):
    risks = []
    for lon, lat in coords:
        dist, idx = tree.query([lat, lon], k=3)
        # средний риск для ближайших ячеек (normalized_prob)
        probs = heat_df.iloc[idx][heat_key].values
        risks.append(probs.mean())
    return {
        'total_risk': float(np.mean(risks)),
        'max_risk': float(np.max(risks)),
        'risk_points': int((np.array(risks) > 0.1).sum())
    }


def get_route(start, end):
    # OSRM-запрос
    lon1, lat1 = start[1], start[0]
    lon2, lat2 = end[1], end[0]
    url = f'http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson'
    resp = requests.get(url)
    data = resp.json()
    coords = data['routes'][0]['geometry']['coordinates']
    meta = analyze_route_risk(coords)
    meta['distance'] = data['routes'][0]['distance']
    meta['duration'] = data['routes'][0]['duration']
    return coords, meta


@app.route('/')
def index():
    # Передаем точки тепловой карты и исходные точки ДТП в шаблон
    return render_template('index.html', heat_points=heat_points, accident_points=accident_points, heat_max=HEAT_MAX)


@app.route('/api/route', methods=['POST'])
def api_route():
    data = request.json
    start = data.get('start')
    end = data.get('end')
    rw = float(data.get('risk_weight', 1.0))
    tw = float(data.get('time_weight', 1.0))
    dw = float(data.get('dist_weight', 1.0))
    if not start or not end:
        return jsonify({'error': 'Укажите start и end адреса'}), 400
    try:
        start_coord = geocode(start)
        end_coord = geocode(end)
        # Получаем альтернативные маршруты
        routes, metas = get_alternative_routes(start_coord, end_coord)
        # Нормализация метрик: риск по глобальному максимуму, время/дистанция по альтернативам
        max_risk = HEAT_MAX or 1
        max_time = max(m['duration'] for m in metas) or 1
        max_dist = max(m['distance'] for m in metas) or 1
        # Выбираем оптимальный маршрут по взвешенной сумме
        best_idx = 0
        best_score = float('inf')
        for i, m in enumerate(metas):
            nr = m['risk_analysis']['total_risk'] / max_risk
            nt = m['duration'] / (max_time or 1)
            nd = m['distance'] / (max_dist or 1)
            score = rw * nr + tw * nt + dw * nd
            if score < best_score:
                best_score, best_idx = score, i
        coords = routes[best_idx]
        meta = metas[best_idx]
        meta['score'] = best_score
        return jsonify({'coordinates': coords, 'metadata': meta})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def get_alternative_routes(start, end, max_alts=3):
    """Возвращает список маршрутов с OSRM и метрики по каждому"""
    lon1, lat1 = start[1], start[0]
    lon2, lat2 = end[1], end[0]
    url = f'http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?' \
          f'alternatives=true&overview=full&geometries=geojson&annotations=duration,distance&steps=false'
    resp = requests.get(url)
    data = resp.json()
    routes, metas = [], []
    for route in data.get('routes', [])[:max_alts]:
        coords = route['geometry']['coordinates']
        # Пересчет риска по маршруту
        risk_info = analyze_route_risk(coords)
        meta = {
            'distance': route['distance'],
            'duration': route['duration'],
            'risk_analysis': risk_info
        }
        routes.append(coords)
        metas.append(meta)
    return routes, metas


@app.route('/api/suggest', methods=['GET'])
def api_suggest():
    """Возвращает подсказки по частичному вводу адреса"""
    q = request.args.get('q', '')
    limit = request.args.get('limit', 5)
    if not q:
        return jsonify([])
    url = 'https://nominatim.openstreetmap.org/search'
    params = {
        'q': q, 'format': 'json', 'limit': limit,
        'viewbox': f"{lon_bounds[0]},{lat_bounds[1]},{lon_bounds[1]},{lat_bounds[0]}",
        'bounded': 1
    }
    try:
        resp = requests.get(url, params=params, headers={'User-Agent': 'risk_mapper'})
        results = resp.json()
        suggestions = [item.get('display_name') for item in results]
        return jsonify(suggestions)
    except Exception:
        return jsonify([])


if __name__ == '__main__':
    app.run(debug=True) 