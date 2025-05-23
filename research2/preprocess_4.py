import json
import pandas as pd

# 파일 경로
CACHE_FILE = 'geocoding_cache.json'
EXCEL_FILE = '종합.xlsx'
OUTPUT_CSV = '위경도_null_주소목록.csv'

# 캐시 로드
with open(CACHE_FILE, 'r', encoding='utf-8') as f:
    cache = json.load(f)

# 위경도 정보가 없는 주소 목록 추출
null_full_addresses = [addr for addr, coord in cache.items() if coord['lat'] is None or coord['lon'] is None]

# 도로명주소만 추출 (부산광역시 이후 제거)
null_road_addresses = [addr.replace("부산광역시", "").strip().split(" ", 1)[1] for addr in null_full_addresses if addr.startswith("부산광역시")]

# 엑셀 데이터 불러오기
df = pd.read_excel(EXCEL_FILE)

# 주소 매칭을 위해 Address 열에서 공백 제거
df['Address_stripped'] = df['Address'].str.strip()

# 위경도 null 주소 포함된 행 필터링
filtered_df = df[df['Address_stripped'].isin(null_road_addresses)]

# 새로운 열 생성: 도로명 주소 및 지번주소 (전체주소 형태)
filtered_df['도로명주소'] = "부산광역시 " + filtered_df['Gu'].astype(str).str.strip() + " " + filtered_df['Address'].astype(str).str.strip()
filtered_df['지번주소'] = "부산광역시 " + filtered_df['Gu'].astype(str).str.strip() + " " + filtered_df['Jibun'].astype(str).str.strip()

# 필요한 열만 선택하여 저장
filtered_df[['도로명주소', '지번주소']].to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

print(f"✅ 저장 완료: {OUTPUT_CSV}")
