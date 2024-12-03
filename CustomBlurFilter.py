import cv2
import numpy as np

class CustomBlurFilter:
    """
    이미지에 enhance_and_blur 기능을 적용하고 결과를 화면에 표시하는 클래스.
    """
    
    def __init__(self, image_path):
        """
        클래스 초기화. 이미지 경로를 받아 이미지를 로드합니다.
        :param image_path: str - 입력 이미지 경로
        """
        self.image_path = image_path
        self.image = self.load_image()
    
    def load_image(self):
        """
        이미지를 경로에서 읽어옵니다.
        :return: np.ndarray - 로드된 이미지
        """
        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError(f"이미지를 찾을 수 없습니다: {self.image_path}")
        return image
    
    @staticmethod
    def enhance_and_blur(image, kernel_size=(5, 5)):
        """
        히스토그램 평활화를 적용한 후 이미지를 블러 처리합니다.
        
        :param image: np.ndarray - 입력 이미지
        :param kernel_size: tuple - Gaussian 블러 커널 크기 (기본값: (5, 5))
        :return: np.ndarray - 대비가 개선되고 블러가 적용된 이미지
        """
        # Step 1: 컬러 이미지를 흑백으로 변환 (필요한 경우)
        if len(image.shape) == 3 and image.shape[2] == 3:  # 컬러 이미지인지 확인
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            grayscale = image  # 이미 흑백 이미지라면 그대로 사용
        
        # Step 2: 히스토그램 평활화 적용
        enhanced = cv2.equalizeHist(grayscale)
        
        # Step 3: Gaussian 블러 처리
        blurred = cv2.GaussianBlur(enhanced, kernel_size, 0)
        
        return blurred
    
    def apply_filter(self, kernel_size=(5, 5)):
        """
        enhance_and_blur 필터를 현재 이미지에 적용합니다.
        :param kernel_size: tuple - Gaussian 블러 커널 크기
        :return: np.ndarray - 필터가 적용된 이미지
        """
        return self.enhance_and_blur(self.image, kernel_size)
    
    def display_images(self, processed_image):
        """
        원본 이미지와 처리된 이미지를 화면에 표시합니다.
        :param processed_image: np.ndarray - 필터가 적용된 이미지
        """
        cv2.imshow("Original Image", self.image)
        cv2.imshow("Processed Image (Enhanced & Blurred)", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 클래스 사용 예시
if __name__ == "__main__":
    # 이미지 파일 경로를 지정하세요
    image_path = "C:/Users/hong0/vscode.py/steak.jpg"  # "example.jpg"를 처리할 이미지 파일 경로로 교체
    
    try:
        # CustomBlurFilter 클래스 초기화
        filter_instance = CustomBlurFilter(image_path)
        
        # 필터 적용
        processed_image = filter_instance.apply_filter(kernel_size=(7, 7))  # 커널 크기를 변경 가능
        
        # 이미지 화면에 표시
        filter_instance.display_images(processed_image)
    
    except Exception as e:
        print(f"오류 발생: {e}")
