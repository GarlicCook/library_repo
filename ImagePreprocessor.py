import cv2
class ImagePreprocessor():
    def __init__(self):
        pass
    def feat1(self):
        pass
    def count_faces(self,image_path):
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
    
    
    
    # 로컬 이미지 경로 설정
        self.image_path = 'sample_image/gang.jpg'  # 이미지 파일 경로
        number_of_faces = count_faces(image_path)
        print(f"이미지에서 발견된 얼굴의 수: {number_of_faces}")
    
    def feat2(self):
        pass
    def feat3(self):
        pass
    def feat4(self):
        pass
    def feat5(self):
        pass



