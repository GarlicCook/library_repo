import cv2
import numpy as np

class ColorFilter():
    #이미지 로드
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)

    #회색 이미지 색성(명도만 남기는 함수)
    def make_gray_image(self):
        if len(self.image.shape) == 3:  # 컬러 이미지인 경우
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # 흑백 변환
        else:
            gray = self.image # 이미 흑백인 경우 그대로 사용
        self.image = gray
        return gray

    #사용자 정의 색상 필터 color: (B, G, R)
    def custom_colorfilter(self, color):
        gray = self.make_gray_image().astype(np.float32)
        height, width = gray.shape
        colored_image = np.zeros((height, width, 3), dtype=np.float32)
        # 각 채널에 사용자 색상 배합
        colored_image[:, :, 0] = (gray * color[0] / 255.0)  # Blue
        colored_image[:, :, 1] = (gray * color[1] / 255.0)  # Green
        colored_image[:, :, 2] = (gray * color[2] / 255.0)  # Red
        
        #이미지에 필터 적용
        self.image = np.clip(colored_image, 0, 255).astype(np.uint8)

        return self
    
    def silver_colorfilter(self):
        color = (192, 192, 192)
        gray = self.make_gray_image().astype(np.float32)
        height, width = gray.shape
        colored_image = np.zeros((height, width, 3), dtype=np.float32)
        # 각 채널에 사용자 색상 배합
        colored_image[:, :, 0] = (gray * color[0] / 255.0)  # Blue
        colored_image[:, :, 1] = (gray * color[1] / 255.0)  # Green
        colored_image[:, :, 2] = (gray * color[2] / 255.0)  # Red
        
        #이미지에 필터 적용
        self.image = np.clip(colored_image, 0, 255).astype(np.uint8)

        return self
    def deepskyblue_colorfilter(self):
        color = (255, 191, 0)
        gray = self.make_gray_image().astype(np.float32)
        height, width = gray.shape
        colored_image = np.zeros((height, width, 3), dtype=np.float32)
        # 각 채널에 사용자 색상 배합
        colored_image[:, :, 0] = (gray * color[0] / 255.0)  # Blue
        colored_image[:, :, 1] = (gray * color[1] / 255.0)  # Green
        colored_image[:, :, 2] = (gray * color[2] / 255.0)  # Red
        
        #이미지에 필터 적용
        self.image = np.clip(colored_image, 0, 255).astype(np.uint8)

        return self
    def magenta_colorfilter(self):
        color = (255, 0, 255)
        gray = self.make_gray_image().astype(np.float32)
        height, width = gray.shape
        colored_image = np.zeros((height, width, 3), dtype=np.float32)
        # 각 채널에 사용자 색상 배합
        colored_image[:, :, 0] = (gray * color[0] / 255.0)  # Blue
        colored_image[:, :, 1] = (gray * color[1] / 255.0)  # Green
        colored_image[:, :, 2] = (gray * color[2] / 255.0)  # Red
        
        #이미지에 필터 적용
        self.image = np.clip(colored_image, 0, 255).astype(np.uint8)

        return self

    
    #이미지 화면에 표시
    def show(self, window_name="Image"):
        cv2.imshow(window_name, self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return self
    

#실행 예제
if __name__ == "__main__":
    #이미지 경로 설정
    image_path = './imagePreprocessor/picture/100.jpg'

    #이미지전처리기 실행
    processor = ColorFilter(image_path)
    user_color = (201, 251, 206)
    processor.custom_colorfilter(user_color)
    processor.show()
