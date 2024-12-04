import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pyperclip
from streamlit_cropper import st_cropper


import process_img as dip
import ocr_app as ocr

def init_session_var():
    if 'MODEL_OPTION' not in st.session_state:
        st.session_state.MODEL_OPTION = None

    if 'MODEL_INPUT' not in st.session_state:
        st.session_state.MODEL_INPUT = None

    if 'PREDICTION_STR' not in st.session_state:
        st.session_state.PREDICTION_STR = None

    if 'PREDICTION_MUL' not in st.session_state:
        st.session_state.PREDICTION_MUL = None

    if 'OPENCV_IMAGE' not in st.session_state:
        st.session_state.OPENCV_IMAGE = None

    if 'OLD_DIL' not in st.session_state:
        st.session_state.OLD_DIL = None

    if 'OLD_ERO' not in st.session_state:
        st.session_state.OLD_ERO = None

    if 'RESIZE_ENABLE' not in st.session_state:
        st.session_state.RESIZE_ENABLE = None

    if 'SIZE_PREDICT' not in st.session_state:
        st.session_state.SIZE_PREDICT = None

    if 'MULTILINE' not in st.session_state:
        st.session_state.MULTILINE = None

    if 'SEGMENTS_IMG' not in st.session_state:
        st.session_state.SEGMENTS_IMG = None

    if 'SEGMENTS_ARR' not in st.session_state:
        st.session_state.SEGMENTS_ARR = None

    if 'SEGMENTS_PRS' not in st.session_state:
        st.session_state.SEGMENTS_PRS = None

def reset():
    st.session_state.MODEL_OPTION = None
    st.session_state.MODEL_INPUT = None
    st.session_state.PREDICTION_STR = None
    st.session_state.PREDICTION_MUL = None
    st.session_state.OPENCV_IMAGE = None
    st.session_state.OLD_DIL = None
    st.session_state.OLD_ERO = None
    st.session_state.RESIZE_ENABLE = None
    st.session_state.SIZE_PREDICT = None
    st.session_state.MULTILINE = None
    st.session_state.SEGMENTS_IMG = None
    st.session_state.SEGMENTS_ARR = None
    st.session_state.SEGMENTS_PRS = None
    st.rerun()


def main():
    # Khởi tạo các biến session
    init_session_var()

    st.set_page_config(
        'Xử lý ảnh PTIT 2024', '/home/hungnm/Desktop/OCR_XLA_PTIT/imgs/Logo_PTIT_University.png')
    
    message_container = st.empty()

    st.title("Bài tập lớn xử lý ảnh - PTIT 2024")
    st.markdown('<div style="margin-top:2rem"></div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: justify; font-size:20px;">Nhận dạng chữ viết tay Tiếng Việt sử dụng mô hình CRNN + LTSM + CTC. ' +
                'Ứng dụng cho phép nhận diện văn bản trên ảnh, ảnh có thể được xử lý trước để tăng độ chính xác của mô hình.', unsafe_allow_html=True)
   

    IMAGE_UPLOAD = st.file_uploader(
        "Định dạng cho phép: JPG, PNG, JPEG", type=['png', 'jpg', 'jpeg'])
    
    # Hiển thị ảnh đã tải lên
    if IMAGE_UPLOAD is not None:
        st.image(IMAGE_UPLOAD, caption='Ảnh đã tải lên', use_container_width=True) # hiển thị ảnh

        # Hiển thị nút để resize ảnh
        if st.session_state.RESIZE_ENABLE == True:
            img = Image.open(IMAGE_UPLOAD)
            cropped_img = st_cropper(
                img, realtime_update=True, box_color='#0000FF', aspect_ratio=None)
            np_image = np.asarray(cropped_img)
            
            # Chuyển đổi định dạng hình ảnh từ RGB sang BGR
            st.session_state.OPENCV_IMAGE = cv2.cvtColor(
                np_image, cv2.COLOR_RGB2BGR)

            if st.button("Hủy"):
                st.session_state.RESIZE_ENABLE = False
                st.rerun()
        else:
            if st.button("Resize"):
                st.session_state.MODEL_INPUT = None
                st.session_state.RESIZE_ENABLE = True
                st.rerun()

    processed_image_container = st.empty()
    if st.session_state.MODEL_INPUT is not None and st.session_state.SEGMENTS_IMG is None:
        processed_image_container.image(st.session_state.MODEL_INPUT,
                                        caption='Ảnh đã xử lý')

    if st.session_state.MODEL_INPUT is None and st.session_state.SEGMENTS_IMG is not None:
        processed_image_container.image(st.session_state.SEGMENTS_IMG,
                                        caption='Ảnh đã xử lý')

    st.markdown('<div style="margin-top:2rem"></div>', unsafe_allow_html=True)
    st.header("Kết quả")
    st.markdown('<div style="margin-top:1rem"></div>', unsafe_allow_html=True)

    if st.session_state.PREDICTION_STR is not None:
        st.text(
            "Click [Copy and reset] để sao chép và reset ứng dụng")
        st.session_state.PREDICTION_STR = st.text_input(
            "Kết quả dự đoán", st.session_state.PREDICTION_STR)
        if st.button("Copy and reset", type="secondary", key=123):
            pyperclip.copy(st.session_state.PREDICTION_STR)
            st.session_state.IMG_DATA = None
            st.session_state.MODEL_INPUT = None
            st.session_state.PREDICTION_STR = None
            st.rerun()

    # Sidebar
    with st.sidebar:
        st.title("GV: Đào Thị Thúy Quỳnh")

        st.title("Thành viên nhóm")
        st.text("B21DCCN412 - Nguyễn Mạnh Hùng")
        st.text("B21DCCN339 - Nguyễn Minh Hiển")
        st.text("B21DCCN771 - Nguyễn Thanh Tùng")
        

        st.title(" ")
        st.title("Tiền xử lý ảnh")
        col1, col2 = st.sidebar.columns([2,1])

        with col1:
            if st.button("Xử lý ảnh đầu vào", type="primary"):
                # Xử lý khi nút [Xử lý ảnh đầu vào] được nhấn
                if IMAGE_UPLOAD is not None:
                    if st.session_state.RESIZE_ENABLE is None or st.session_state.RESIZE_ENABLE == False:
                        img_array = np.frombuffer(IMAGE_UPLOAD.read(), np.uint8)
                        st.session_state.OPENCV_IMAGE = cv2.imdecode(
                            img_array, cv2.IMREAD_COLOR)

                    st.session_state.MODEL_INPUT = dip.process_image(st.session_state.OPENCV_IMAGE)
                    processed_image_container.image(dip.process_image(st.session_state.OPENCV_IMAGE), caption='Ảnh đã xử lý')
                else:
                    message_container.error('Vui lòng upload ảnh cần xử lý')

        with col2:
            if st.button("Reset"):
                reset()
                st.rerun()

         # Giá trị của slider 1
        param1 = st.sidebar.slider(
            "Co đối tượng trong ảnh", 1, 8, 1, help="Tăng hoặc giảm kenel", key=1)
        if (st.session_state.MODEL_INPUT is not None
                and st.session_state.OLD_ERO != param1):
            st.session_state.OLD_ERO = param1
            st.session_state.MODEL_INPUT = dip.erosion_dilation_image(st.session_state.MODEL_INPUT,
                                                                      param1, True)
            processed_image_container.image(st.session_state.MODEL_INPUT)

        # Giá trị của slider 2
        param2 = st.sidebar.slider(
            "Giãn đối tượng trong ảnh", 1, 8, 1, help="Tăng hoặc giảm kenel", key=2)
        if (st.session_state.MODEL_INPUT is not None
                and st.session_state.OLD_DIL != param2):
            st.session_state.OLD_DIL = param2
            st.session_state.MODEL_INPUT = dip.erosion_dilation_image(st.session_state.MODEL_INPUT,
                                                                      param2, False)
            processed_image_container.image(st.session_state.MODEL_INPUT)

        
        if st.sidebar.button("Nhận diện văn bản", type="primary"):
            if st.session_state.MODEL_INPUT is not None:

                    if st.session_state.MULTILINE:
                        processed_image_container.image(st.session_state.SEGMENTS_IMG,
                                                        caption='Ảnh đã xử lý')
                        with processed_image_container.container():
                            st.image(
                                st.session_state.SEGMENTS_IMG, caption='Ảnh đã xử lý')
                            i = 0
                            for img_prs in st.session_state.SEGMENTS_PRS:
                                st.image(img_prs, 'Segments: {}'.format(i))
                                i += 1
                        st.session_state.PREDICTION_MUL = ocr.prediction_multiline(st.session_state.MODEL_INPUT,
                                                                                   st.session_state.SIZE_PREDICT)
                    else:
                        processed_image_container.image(
                            st.session_state.MODEL_INPUT, caption='Ảnh đã xử lý')
                        # Dự đoán chuỗi
                        st.session_state.PREDICTION_STR = ocr.prediction_ocr_crnn_ctc(
                            dip.convert_img_to_input(st.session_state.MODEL_INPUT))
                    st.rerun()
            
if __name__ == "__main__":
    main()
