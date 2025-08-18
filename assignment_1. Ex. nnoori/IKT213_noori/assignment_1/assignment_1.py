"""
Author: Douwe Berkeij
Date: 18-08-2025
"""

import cv2

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

def get_camera_outputs():
    """
    saves the fps, height, and width from the webcam
    """
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    with open("assignment_1. Ex. nnoori/IKT213_noori/assignment_1/solutions/camera_outputs.txt", "w+", encoding="utf-8") as f:
        f.write(f"FPS: {fps}\n")
        f.write(f"Width: {width}\n")
        f.write(f"Height: {height}\n")

if __name__ == "__main__":
    image = cv2.imread("assignment_1. Ex. nnoori/IKT213_noori/assignment_1/lena-1.png")
    print_image_information(image)
    get_camera_outputs()