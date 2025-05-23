import geopandas as gpd
import pandas as pd

# 1. Shapefile 읽기 (인코딩 명시)
gdf = gpd.read_file("sig.shp", encoding="cp949")  # 파일 경로에 맞게 수정

# 2. 원본 좌표계 설정 (예: EPSG:5179 - UTM-K)
gdf.set_crs(epsg=5179, inplace=True)

# 3. 위경도 좌표계(WGS84)로 변환
gdf = gdf.to_crs(epsg=4326)

# 4. 부산광역시에 해당하는 행정코드는 '26'으로 시작 → 해당 데이터만 필터링
gdf_busan = gdf[gdf["SIG_CD"].str.startswith("26")]

# 5. 각 구의 위경도 범위 계산
results = []
for _, row in gdf_busan.iterrows():
    name = row["SIG_KOR_NM"]
    bounds = row.geometry.bounds  # (minx, miny, maxx, maxy) = (min_lon, min_lat, max_lon, max_lat)
    min_lon, min_lat, max_lon, max_lat = bounds
    results.append({
        "구": name,
        "min_lat": min_lat,
        "max_lat": max_lat,
        "min_lon": min_lon,
        "max_lon": max_lon,
    })

# 6. 데이터프레임 생성 및 16개 구만 필터링
df_bounds = pd.DataFrame(results)

# 부산광역시 16개 구 이름 목록
busan_gu_list = [
    "강서구", "금정구", "기장군", "남구", "동구", "동래구", "부산진구", "북구",
    "사상구", "사하구", "서구", "수영구", "연제구", "영도구", "중구", "해운대구"
]

df_busan = df_bounds[df_bounds["구"].isin(busan_gu_list)]

# 7. 결과 출력 (정렬 포함)
print(df_busan.sort_values("구"))
