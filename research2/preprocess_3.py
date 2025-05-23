import pandas as pd
import requests
import json
import time
import os

# 카카오 API 키
KAKAO_API_KEY = '87bf0a99b34b38ee61803ed44cdf6742'

# 캐시 파일 경로
CACHE_FILE = 'geocoding_cache.json'
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        cache = json.load(f)
else:
    cache = {}

# 캐시 저장 함수
def save_cache():
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

# 주소 → 위경도 변환 함수 (캐시 포함)
def kakao_geocode_cached(address):
    if address in cache:
        return cache[address]

    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    params = {"query": address}
    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    lat, lon = None, None  # 기본값

    try:
        if 'documents' in data and len(data['documents']) > 0:
            doc = data['documents'][0]
            if 'address' in doc and doc['address'] is not None:
                lat = float(doc['address']['y'])
                lon = float(doc['address']['x'])
            elif 'road_address' in doc and doc['road_address'] is not None:
                lat = float(doc['road_address']['y'])
                lon = float(doc['road_address']['x'])
    except Exception as e:
        print(f"❌ 예외 발생: {e} @ 주소: {address}")

    cache[address] = {'lat': lat, 'lon': lon}
    save_cache()
    time.sleep(0.1)

    return {'lat': lat, 'lon': lon}

# ✅ 엑셀 데이터 불러오기
input_file = '종합.xlsx'  # 엑셀 파일명
df = pd.read_excel(input_file)

# ✅ 주소 열에서 위경도 변환
latitudes = []
longitudes = []

for i, row in df.iterrows():
    raw_address = row['Address']  # 또는 '지번', '도로명주소' 등 원하는 주소열
    full_address = f"부산광역시 {row['Gu']} {raw_address}"

    coords = kakao_geocode_cached(full_address)
    latitudes.append(coords['lat'])
    longitudes.append(coords['lon'])

    print(f"{i+1}/{len(df)} - {full_address} → {coords}")

# ✅ 위도/경도 열 추가
df['Latitude'] = latitudes
df['Longitude'] = longitudes

# ✅ 결과 저장
output_file = '주소_위경도_매핑결과.xlsx'
df.to_excel(output_file, index=False)

print(f"✅ 변환 완료! → {output_file}")
