import cv2

class CountFace():
    def __init__(self, image_path=None):
        self.image_path = image_path  # 이미지 경로 초기화
    
    def set_image_path(self, image_path):
        """이미지 경로를 설정하는 메서드"""
        self.image_path = image_path
    
    def count_faces(self):
        """이미지에서 얼굴을 탐지하고 개수를 반환하는 메서드"""
        if not self.image_path:
            print("이미지 경로가 설정되지 않았습니다. set_image_path()를 사용하세요.")
            return 0
        
        # Haar Cascade 모델 로드
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # 이미지 로드
        image = cv2.imread(self.image_path)
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

if __name__ == "__main__":
# 사용 예시
    image_path = 'data/gang.jpg'  # 이미지 파일 경로
    preprocessor = CountFace(image_path)  # 클래스 인스턴스 생성 및 경로 초기화

# 얼굴 탐지 실행
    number_of_faces = preprocessor.count_faces()
    print(f"이미지에서 발견된 얼굴의 수: {number_of_faces}")

    
