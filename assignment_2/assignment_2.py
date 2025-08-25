"""
Author: Douwe Berkeij
Date: 25-08-2025
"""

import cv2
import numpy as np

def padding(img, borderWidth):
    return cv2.copyMakeBorder(img, borderWidth, borderWidth, borderWidth, borderWidth, cv2.BORDER_REFLECT)

def crop(img, x_0, x_1,  y_0, y_1):
    height, width, _ = img.shape
    modifiedX_1 = [height - x_1 if 0 < x_1 else height]
    modifiedY_1 = [width - y_1 if 0 < y_1 else width]
    return img[x_0:modifiedX_1[0], y_0:modifiedY_1[0]]

def resize(img, new_width, new_height):
    return cv2.resize(img, (new_width, new_height))

def copy(img, emptyPictureArray):
    emptyPictureArray = np.copy(img)
    return emptyPictureArray

def grey_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def hue_shift(img, emptyPictureArray, hue):
    shifted = (img.astype(np.int16) + hue) % 256
    result = shifted.astype(np.uint8)
    return result

def smoothing(img):
    return cv2.GaussianBlur(img, ksize=(15, 15), sigmaX=0, borderType=cv2.BORDER_DEFAULT)

def rotation(img, rotation_angle):
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    return cv2.warpAffine(img, rotation_matrix, (width, height))

if __name__ == "__main__":
    img = cv2.imread("assignment_2/lena-2.png")
    padded_img = padding(img, 100)
    cropped_img = crop(img, 80, 130, 80, 130)
    resized_img = resize(img, 200, 200)
    height, width, _ = img.shape
    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)
    copied_img = copy(img, emptyPictureArray)
    grey_scaled_img = grey_scale(img)
    hsv_img = hsv(img)
    hue_shifted_img = hue_shift(img, emptyPictureArray, 50)
    smoothed_img = smoothing(img)
    rotated_img_90 = rotation(img, 90)
    rotated_img_180 = rotation(img, 180)

    cv2.imwrite("assignment_2/solutions/padded.png", padded_img)
    cv2.imwrite("assignment_2/solutions/cropped.png", cropped_img)
    cv2.imwrite("assignment_2/solutions/resized.png", resized_img)
    cv2.imwrite("assignment_2/solutions/copied.png", copied_img)
    cv2.imwrite("assignment_2/solutions/grey_scaled.png", grey_scaled_img)
    cv2.imwrite("assignment_2/solutions/hsv.png", hsv_img)
    cv2.imwrite("assignment_2/solutions/hue_shifted.png", hue_shifted_img)
    cv2.imwrite("assignment_2/solutions/smoothed.png", smoothed_img)
    cv2.imwrite("assignment_2/solutions/rotated_90.png", rotated_img_90)
    cv2.imwrite("assignment_2/solutions/rotated_180.png", rotated_img_180)