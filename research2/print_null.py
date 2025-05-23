import json

# 캐시 파일 경로
cache_file = 'geocoding_cache.json'

# 파일 읽기
with open(cache_file, 'r', encoding='utf-8') as f:
    cache = json.load(f)

# lat 또는 lon이 None인 항목 필터링 및 출력
missing_count = 0
print("⚠️ 위도/경도 정보가 없는 주소 목록:")

for address, coords in cache.items():
    if coords['lat'] is None or coords['lon'] is None:
        print(f"- {address} → {coords}")
        missing_count += 1

print(f"\n❗ 위도/경도 정보가 없는 주소 개수: {missing_count}")
