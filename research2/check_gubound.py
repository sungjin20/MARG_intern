import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from PIL import Image

IMAGE_SIZE = (256, 256)

# (1) 엑셀 파일 불러오기
df = pd.read_excel("종합.xlsx")

# (2) 월 타임스탬프 정의
def generate_months(start):
    end = '201711'
    months = []
    start_date = datetime.strptime(str(start), "%Y%m")
    end_date = datetime.strptime(str(end), "%Y%m")
    while start_date <= end_date:
        months.append(start_date.strftime("%Y_%m"))
        if start_date.month == 12:
            start_date = datetime(start_date.year + 1, 1, 1)
        else:
            start_date = datetime(start_date.year, start_date.month + 1, 1)
    return months

# (3) JSON 캐시 파일 로드
with open("geocoding_cache.json", "r", encoding="utf-8") as f:
    geocoding_cache = json.load(f)

# (4) 캐시 기반 주소 → 좌표 함수
def get_coords(address):
    if address in geocoding_cache:
        lat = geocoding_cache[address].get('lat')
        lon = geocoding_cache[address].get('lon')
        if lat is not None and lon is not None:
            return lat, lon
    return None, None

# (5) 좌표 → 이미지 위치 매핑 함수
def coord_to_pixel(lat, lon, bounds, image_size):
    lat_min, lat_max, lon_min, lon_max = bounds
    h, w = image_size
    x = int((lon - lon_min) / (lon_max - lon_min) * w)
    y = int((lat_max - lat) / (lat_max - lat_min) * h)  # 위쪽이 y=0이 되도록
    return np.clip(x, 0, w-1), np.clip(y, 0, h-1)

# (6) 구별 경계 정의
gu_bounds = {
    '강서구': (34.989093, 35.236081, 128.761718, 128.998224),
    '금정구': (35.209541, 35.306584, 129.036564, 129.144606),
    '기장군': (35.181620, 35.389037, 129.108562, 129.305605),
    '남구': (35.090908, 35.161318, 129.063001, 129.128720),
    '동구': (35.108664, 35.146174, 129.024309, 129.066219),
    '동래구': (35.183326, 35.226273, 129.044129, 129.116370),
    '부산진구': (35.135508, 35.200079, 129.010503, 129.082832),
    '북구': (35.187475, 35.276895, 128.987736, 129.061767),
    '사상구': (35.116925, 35.201455, 128.957445, 129.021085),
    '사하구': (34.884428, 35.136845, 128.927046, 129.014365),
    '서구': (35.047976, 35.140989, 128.996015, 129.031134),
    '수영구': (35.135752, 35.186500, 129.091552, 129.135035),
    '연제구': (35.158174, 35.199534, 129.047321, 129.115177),
    '영도구': (35.036699, 35.103648, 129.030992, 129.105243),
    '중구': (35.095675, 35.116605, 129.021177, 129.044443),
    '해운대구': (35.151846, 35.250968, 129.110240, 129.208410)
}

# (7) 각 구의 위도/경도 범위 차이 중 최대값 계산 → L
max_diff = 0
for bounds in gu_bounds.values():
    lat_min, lat_max, lon_min, lon_max = bounds
    lat_diff = lat_max - lat_min
    lon_diff = lon_max - lon_min
    max_diff = max(max_diff, lat_diff, lon_diff)
L = max_diff  # 가장 큰 범위를 모든 구의 이미지 범위로 설정

# (8) 각 구의 중심 좌표 계산 및 이미지 범위 설정
gu_center_bounds = {}
for gu, (lat_min, lat_max, lon_min, lon_max) in gu_bounds.items():
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    lat_min_new = center_lat - L / 2
    lat_max_new = center_lat + L / 2
    lon_min_new = center_lon - L / 2
    lon_max_new = center_lon + L / 2
    gu_center_bounds[gu] = (lat_min_new, lat_max_new, lon_min_new, lon_max_new)

print(gu_center_bounds['영도구'])