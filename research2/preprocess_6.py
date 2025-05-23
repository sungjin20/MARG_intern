import pandas as pd
import json

# 파일 경로
csv_file = '위도경도_결과_2.csv'
json_file = 'geocoding_cache.json'

# CSV 파일 로드
df = pd.read_csv(csv_file)

# JSON 캐시 로드
with open(json_file, 'r', encoding='utf-8') as f:
    cache = json.load(f)

# 업데이트 여부 플래그
updated = False

# "_GC_TYPE"이 "정"인 행만 필터링
filtered_df = df[df['_GC_TYPE'] == '대']

# 주소 기준으로 cache 업데이트
for _, row in filtered_df.iterrows():
    address = row['field1']
    new_lat = row['Latitude']
    new_lon = row['Longitude']
    
    if address in cache and (cache[address]['lat'] is None or cache[address]['lon'] is None):
        print(f"🛠 업데이트: {address} → lat: {new_lat}, lon: {new_lon}")
        cache[address]['lat'] = new_lat
        cache[address]['lon'] = new_lon
        updated = True

# 변경사항 저장
if updated:
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print("✅ 캐시가 성공적으로 업데이트되었습니다.")
else:
    print("ℹ️ 업데이트할 항목이 없습니다.")
