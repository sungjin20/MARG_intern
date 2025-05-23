import geopandas as gpd

# 모든 구성 파일(.shp, .shx, .dbf 등)이 동일한 폴더에 있어야 함
gdf = gpd.read_file("a.shp")

# 좌표계 확인 (예: EPSG:4326은 위도/경도)
print(gdf.crs)

# 위도, 경도 추출 (만약 좌표계가 위경도 형태일 경우)
gdf["Longitude"] = gdf.geometry.x
gdf["Latitude"] = gdf.geometry.y

# 또는 위도/경도가 아닌 경우 변환 (예: EPSG:4326으로)
if gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs("EPSG:4326")
    gdf["Longitude"] = gdf.geometry.x
    gdf["Latitude"] = gdf.geometry.y

# 결과 확인
print(gdf[["Longitude", "Latitude"]].head())

# CSV로 저장
gdf.to_csv("위도경도_결과_2.csv", index=False)
