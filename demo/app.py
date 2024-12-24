import streamlit as st
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import numpy as np
import cv2

st.set_page_config(layout="wide", page_title="말벌 탐지")

# 컬럼 설정
title, logo = st.columns([4, 1])

with title:
    st.title(":bee: 말벌 탐지 서비스")
with logo:
    st.image("assets/img/logo.png", width=160)
        
# st.write("## 말벌 탐지 서비스 :bee:")
st.markdown(
    """
    이 인공지능 서비스는 
    <b>꿀벌<span style="color:blue">(honeybee)</span></b>과 
    <b>말벌<span style="color:red">(hornet)</span></b>을 구분지어 탐지할 수 있어, 양봉업계에서 유용하게 활용될 수 있습니다.
    """,
    unsafe_allow_html=True,
)
st.write(" 자세한 코드는 [GitHub 저장소](https://github.com/DongwooYoon/Hornet-Detection-Using-YOLO)에서 확인하고 다운로드할 수 있습니다.")
st.sidebar.write("## 이미지 업로드 / 다운로드 :gear:")

MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB

# YOLO 모델 로드
model = YOLO("assets/model/best_16_4.pt")

# 모델의 가중치 확인
state_dict = model.model.state_dict()

# 처음 5개의 레이어 가중치만 출력
for idx, (layer_name, weights) in enumerate(state_dict.items()):
    if idx < 5:  # 처음 5개까지만 출력
        print(f"레이어 이름: {layer_name}, 가중치 크기: {weights}")
    else:
        break
    
# Download image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def predict_objects(upload):
    # 업로드된 이미지 로드
    image = Image.open(upload)
    col1.write(":camera: 원본 이미지")
    col1.image(image)
    
    # 이미지 배열로 변환
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_array = np.array(image)
    
    # YOLO 모델로 예측 수행
    results = model.predict(source=img_array, imgsz=640, save=False)

    # 모델의 가중치 확인
    state_dict = model.model.state_dict()

    # 처음 5개의 레이어 가중치만 출력
    for idx, (layer_name, weights) in enumerate(state_dict.items()):
        if idx < 5:  # 처음 5개까지만 출력
            print(f"레이어 이름: {layer_name}, 가중치 크기: {weights}")
        else:
            break
        
    # 탐지 결과 그리기
    detected_image = img_array.copy()
    for result in results[0].boxes.data.tolist():
        x_min, y_min, x_max, y_max, confidence, class_id = result
        class_name = model.names[int(class_id)]
        label = f"{model.names[int(class_id)]} ({confidence:.2f})"
        
        # 클래스별 색상 설정
        color = (255, 0, 0) if class_name == "hornet" else (0, 0, 255)  # honeybee=파랑 hornet=빨강
        # Bounding Box 그리기
        cv2.rectangle(
            detected_image,
            (int(x_min), int(y_min)), (int(x_max), int(y_max)),
            color,
            2
        )
        cv2.putText(
            detected_image,
            label,
            (int(x_min), int(y_min) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            1,
            cv2.LINE_AA,
        )
                
    # OpenCV 이미지를 PIL 이미지로 변환
    detected_image = Image.fromarray(detected_image)

    # 탐지 이미지 표시
    col2.write(":mag: 탐지 이미지")
    col2.image(detected_image)
    for result in results[0].boxes.data.tolist():
        x_min, y_min, x_max, y_max, confidence, class_id = result
        class_name = model.names[int(class_id)]
        col2.write(f"- **{class_name}**: 신뢰도 {confidence:.2f}, 위치: ({x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f})")
    
    # 다운로드 버튼
    st.sidebar.download_button(
        "탐지 이미지 다운로드",
        convert_image(detected_image),
        "detected_image.png",
        "image/png",
    )

col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("말벌과 관련된 이미지를 업로드하세요.", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("업로드된 이미지가 너무 큽니다. 200MB보다 작은 이미지를 업로드해주세요.")
    else:
        predict_objects(my_upload)
else:
    predict_objects("assets/img/bees.jpg")