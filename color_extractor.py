from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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

def plot_palette(palette):
    """
    추출된 주요 색상을 시각적으로 표시합니다.
    
    Args:
        palette (List[Tuple[int, int, int]]): RGB 형식의 주요 색상 리스트.
    """
    # 그래프 크기 설정 및 축 제거
    plt.figure(figsize=(8, 2))
    plt.axis('off')
    plt.title("추출된 색상 팔레트")
    
    # 색상을 시각적으로 나타내는 가로 막대 생성
    bar = np.zeros((50, len(palette) * 50, 3), dtype=np.uint8)
    for i, color in enumerate(palette):
        # 각 색상을 막대의 일정 부분에 채워 넣음
        bar[:, i*50:(i+1)*50, :] = color
    
    # 생성된 팔레트를 화면에 출력
    plt.imshow(bar)
    plt.show()

# 사용 예제
image_path = "lake.jpg"  # 처리할 이미지 경로
n_colors = 5  # 추출할 색상 개수

# 색상 팔레트 추출 및 시각화
palette = extract_color_palette(image_path, n_colors)
print("추출된 색상 팔레트:", palette)  # RGB 값 출력
plot_palette(palette)  # 팔레트 표시
