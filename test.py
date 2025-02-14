import torch
from diffusers import DiffusionPipeline

# 사용할 GPU 지정 (예: GPU 1번 사용)
torch.cuda.set_device(0)  # GPU 1번 선택

# 모델을 불러오고, 지정한 GPU로 이동
pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda:0")  # GPU 1번으로 모델 이동

# 텍스트 입력으로 이미지를 생성
pipeline("An image of a squirrel in Picasso style").images[0]
