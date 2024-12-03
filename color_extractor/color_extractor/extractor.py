from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

def extract_color_palette(image_path, n_colors=5):
    """
    이미지에서 주요 색상 팔레트를 추출합니다.
    
    Args:
        image_path (str): 입력 이미지 파일 경로.
        n_colors (int): 추출할 주요 색상의 개수.
        
    Returns:
        List[Tuple[int, int, int]]: RGB 형식으로 주요 색상 리스트 반환.
    """
    # 이미지를 열고 RGB 형식으로 변환
    image = Image.open(image_path).convert('RGB')
    
    # 이미지를 200x200 크기로 리사이즈 (처리 속도 향상을 위해)
    image = image.resize((200, 200))
    
    # 이미지를 numpy 배열로 변환
    image_array = np.array(image)
    
    # 배열을 (픽셀 수, RGB 채널) 형식으로 재구조화
    pixels = image_array.reshape(-1, 3)
    
    # KMeans 알고리즘으로 주요 색상 클러스터링
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    
    # 클러스터 중심(주요 색상)을 정수 값으로 변환
    palette = kmeans.cluster_centers_.astype(int)
    
    # RGB 형식의 튜플로 반환
    return [tuple(color) for color in palette]

