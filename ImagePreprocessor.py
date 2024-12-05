import cv2
import numpy as np

class ImagePreprocessor():
    #이미지 로드
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)

    #return colored_image
    def make_gray_image(self):
        if len(self.image.shape) == 3:  # 컬러 이미지인 경우
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # 흑백 변환
        else:
            gray = self.image  # 이미 흑백인 경우 그대로 사용
        return gray.astype(np.float32)

    #사용자 정의 색상 필터 color: (B, G, R)
    def custom_colorfilter(self, color):
        gray = self.make_gray_image()
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
        gray = self.make_gray_image()
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
        gray = self.make_gray_image()
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
        gray = self.make_gray_image()
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
    
    """
    여기다가 각자 코드 추가하셔서 merge하시면 될 것 같습니다.
    """
    def count_faces(image_path):
        # Haar Cascade 모델 로드
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print("이미지를 로드할 수 없습니다. 경로를 확인하세요.")
            return 0
    
        # 그레이스케일 변환 (얼굴 탐지에 효과적)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        # 얼굴 탐지
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
        # 얼굴 개수 출력
        print(f"탐지된 얼굴 수: {len(faces)}")
    
        # 탐지된 얼굴에 사각형 그리기
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
        # 결과 이미지 표시
        cv2.imshow('Detected Faces', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
        return len(faces)
    
    
    # 로컬 이미지 경로 설정
    image_path = 'sample_image/gang.jpg'  # 이미지 파일 경로
    number_of_faces = count_faces(image_path)
    print(f"이미지에서 발견된 얼굴의 수: {number_of_faces}")
    def feat3(self):
      pass


#실행 예제
if __name__ == "__main__":
    #이미지 경로 설정
    image_path = './data/4.jpg'

    #이미지전처리기 실행
    processor = ImagePreprocessor(image_path)
    user_color = (201, 251, 206)
    processor.custom_colorfilter(user_color)
    #processor.deepskyblue_colorfilter()
    processor.show()
