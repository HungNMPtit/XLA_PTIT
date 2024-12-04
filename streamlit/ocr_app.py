import ocr as ocr

def prediction_ocr_crnn_ctc(input_img):
    str_pred = ocr.prediction_ocr(input_img)
    print('Prediction:')
    print(str_pred)
    return str_pred