import cv2
import numpy as np

def process_image(img_file):
    img = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)
    height = 118
    width = 2122
    img = cv2.resize(img, (int(118/height*width), 118))
    img = np.pad(img, ((0, 0), (0, 2167-width)), 'median')
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    img = np.expand_dims(img, axis=2)
    img = img/255.
    return img

def convert_img_to_input(img_file):
    valid_img = []
    valid_img.append(img_file)
    valid_img = np.array(valid_img)
    return valid_img

def erosion_dilation_image(image, kernel_size, isErosion):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if (isErosion == True):
        img = cv2.erode(image, kernel, iterations=1)
    else:
        img = cv2.dilate(image, kernel, iterations=1)
    return img
