import cv2
import numpy as np
import os

class CustomBlurFilter:
    def __init__(self, image_path):
        self.image_path = os.path.join(image_path)  # 파일 경로 설정
        self.image = None
        self.enhanced_blurred_image = None

    def load_image(self):
        # 한글 경로 처리
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"File not found: {self.image_path}")
        
        self.image = cv2.imdecode(np.fromfile(self.image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if self.image is None:
            raise ValueError("Failed to load image. Check the file format or path.")
        return self.image

    def enhance_and_blur(self, clip_limit=2.0, tile_grid_size=(8, 8), blur_ksize=(15, 15)):
        if self.image is None:
            raise ValueError("Image not loaded. Call load_image() first.")
        
        # 이미지 밝기 및 대비 향상
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 가우시안 블러 적용
        self.enhanced_blurred_image = cv2.GaussianBlur(enhanced_image, blur_ksize, 0)
        return self.enhanced_blurred_image

    def display_images(self):
        if self.image is None or self.enhanced_blurred_image is None:
            raise ValueError("Images not processed. Ensure load_image() and enhance_and_blur() are called.")
        
        cv2.imshow("Original Image", self.image)
        cv2.imshow("Enhanced and Blurred Image", self.enhanced_blurred_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# 실행 예제
if __name__ == "__main__":
    # 이미지 경로 설정
    image_path = os.path.join("./library_repo/Example Image", "steak.jpg")  # 경로에 맞게 수정하세요
    
    try:
        # CustomBlurFilter 사용
        filter_instance = CustomBlurFilter(image_path)
        filter_instance.load_image()
        filter_instance.enhance_and_blur()
        filter_instance.display_images()
    except Exception as e:
        print(f"Error: {e}")
