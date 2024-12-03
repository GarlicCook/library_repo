import matplotlib.pyplot as plt
import numpy as np

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
