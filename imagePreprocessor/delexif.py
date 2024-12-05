import piexif



def modexif(image_path, output_path, make = b"", model = b"", datetime = b""):
    exif_dict = piexif.load(image_path)
    #이미지 메타데이터 읽기


    exif_dict['0th'][271] = (make if make == b"" else make.decode("utf-8"))
    #제조사 삭제 or 수정
    exif_dict['0th'][272] = (model if model == b"" else model.decode("utf-8"))
    #모델이름 삭제 or 수정



    exif_dict['GPS']= {}
    exif_dict['Exif'] = {}
    del(exif_dict['0th'][34665])
    del(exif_dict['0th'][34853])
    #exif, gps 삭제



    exif_dict['Exif'][36867] = (datetime if datetime == b"" else datetime.decode("utf-8"))
    exif_dict['Exif'][36868] = (datetime if datetime == b"" else datetime.decode("utf-8"))
    #찍은시간 삭제 or 수정



    exif_b=piexif.dump(exif_dict)
    piexif.insert(exif_b, image_path, output_path)
    #새 파일 생성



