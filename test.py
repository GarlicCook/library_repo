import cv2
import numpy as np


class CustomColorFilter:
    def __init__(self, image_path):
        """
        이미지 로드
        :param image_path: 이미지 파일 경로
        """
        self.image = cv2.imread(image_path)

    def apply_custom_color(self, color):
        """
        사용자 정의 색상 필터 적용
        :param color: 사용자 정의 색상 (B, G, R) 튜플
        """
        if len(self.image.shape) == 3:  # 컬러 이미지인 경우
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # 흑백 변환
        else:
            gray = self.image  # 이미 흑백인 경우 그대로 사용
        gray = gray.astype(np.float32)
        # 사용자 정의 색상 적용
        height, width = gray.shape
        colored_image = np.zeros((height, width, 3), dtype=np.float32)

        # 각 채널에 사용자 색상 배합
        colored_image[:, :, 0] = (gray * color[0] / 255.0)  # Blue
        colored_image[:, :, 1] = (gray * color[1] / 255.0)  # Green
        colored_image[:, :, 2] = (gray * color[2] / 255.0)  # Red

        #테스트용 코드
        print(colored_image)
        print("Min:", colored_image.min(), "Max:", colored_image.max())
        self.image = np.clip(colored_image, 0, 255).astype(np.uint8)
        print(self.image)
        print("Min:", self.image.min(), "Max:", self.image.max())
        return self

    def show(self, window_name="Image"):
        """
        이미지를 화면에 표시
        :param window_name: 창 이름
        """
        cv2.imshow(window_name, self.image)
        cv2.waitKey(0)
        return self
    


input_image = './library_repo/data/4.jpg'
# img = cv2.imread(input_image)
# print(img.shape)
# cv2.imshow("test", img)
# cv2.waitKey()



user_color = (0, 69, 255) #B G R tuple

filter = CustomColorFilter(input_image)

filter.apply_custom_color(user_color)

filter.show()
