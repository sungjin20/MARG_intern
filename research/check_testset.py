region = 'Jeju'

# 파일 경로 설정
trainset_path = f"/data2/personal/sungjin/korean_dialects/modified/{region}/{region}_classification_trainset.txt"
testset_path = f"/data2/personal/sungjin/korean_dialects/modified/{region}/{region}_classification2_testset.txt"

# 파일 읽기
with open(trainset_path, "r", encoding="utf-8") as f:
    trainset = set(f.read().splitlines())

with open(testset_path, "r", encoding="utf-8") as f:
    testset = set(f.read().splitlines())

# 겹치는 경로 찾기
overlapping_paths = trainset.intersection(testset)

# 결과 출력
if overlapping_paths:
    print(f"총 {len(overlapping_paths)}개의 중복된 경로가 있습니다.")
    for path in overlapping_paths:
        print(path)
else:
    print("겹치는 경로가 없습니다.")
