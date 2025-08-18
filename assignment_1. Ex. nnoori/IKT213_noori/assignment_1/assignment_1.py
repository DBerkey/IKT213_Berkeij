import numpy as np
import cv2

"""
Author: Douwe Berkeij
Date: 18-08-2025
"""

def print_image_information(image):
    """
    Print the information of the given image.
    """
    print("Image Information:")
    print(f"\tHeight: {image.shape[0]}")
    print(f"\tWidth: {image.shape[1]}")
    print(f"\tChannels: {image.shape[2]}")
    print(f"\tImage Size: {image.size} bytes")
    print(f"\tImage Data Type: {image.dtype}")

if __name__ == "__main__":
    image = cv2.imread("assignment_1. Ex. nnoori/IKT213_noori/assignment_1/lena-1.png")
    print_image_information(image)