"""
Author: Douwe Berkeij
Date: 15-09-2025
"""

import cv2
import numpy as np


def detect_edges_sobel(image):
    """
    Detect edges in an image using the Sobel operator.

    Parameters:
    - image: np.ndarray, input image.

    Returns:
    - edges: np.ndarray, binary image with detected edges.
    """
    bimg_blur = cv2.GaussianBlur(image, (3, 3), 0)

    sobelxy = cv2.Sobel(bimg_blur, cv2.CV_64F, 1, 1, ksize=5)

    return sobelxy


def detect_edge_canny(image):
    """
    Detect edges in an image using the Canny edge detector.

    Parameters:
    - image: np.ndarray, input image.

    Returns:
    - edges: np.ndarray, binary image with detected edges.
    """
    bimg_blur = cv2.GaussianBlur(image, (3, 3), 0)

    edges = cv2.Canny(bimg_blur, 50, 50)

    return edges


def match_template(image, template):
    """
    Match a template in an image using normalized cross-correlation.

    Parameters:
    - image: np.ndarray, input image.
    - template: np.ndarray, template to match.

    Returns:
    - new_image: np.ndarray, image with red rectangles drawn around 
                    matched templates.
    """
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(result >= threshold)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + template.shape[1], pt[1] +
                                  template.shape[0]), (0, 0, 255), 2)
    new_image = image
    return new_image


def resize_image(image, scale_factor: int, up_or_down: str):
    """
    Resize an image by a given scale percentage.

    Parameters:
    - image: np.ndarray, input image.
    - scale_factor: int, percentage to scale the image.
    - up_or_down: str, 'up' to enlarge, 'down' to reduce.

    Returns:
    - resized_image: np.ndarray, resized image.
    """

    rows, cols = image.shape[:2]

    if up_or_down == 'up':
        resized_image = cv2.pyrUp(image, dstsize=(scale_factor * cols,
                                                  scale_factor * rows))
    elif up_or_down == 'down':
        resized_image = cv2.pyrDown(image, dstsize=(
            cols // scale_factor, rows // scale_factor))
    else:
        raise ValueError("up_or_down must be 'up' or 'down'")

    return resized_image


if __name__ == "__main__":
    IMAGE_PATH_LAMBO = 'assignment_3/lambo.png'
    image_lambo = cv2.imread(IMAGE_PATH_LAMBO, cv2.IMREAD_GRAYSCALE)
    edges_sobel = detect_edges_sobel(image_lambo)
    edges_canny = detect_edge_canny(image_lambo)

    IMAGE_PATH_SHAPES = 'assignment_3/shapes-1.png'
    IMAGE_PATH_SHAPES_TEMPLATE = 'assignment_3/shapes_template.jpg'

    image_shapes = cv2.imread(IMAGE_PATH_SHAPES, cv2.IMREAD_GRAYSCALE)
    image_shapes_template = cv2.imread(IMAGE_PATH_SHAPES_TEMPLATE,
                                       cv2.IMREAD_GRAYSCALE)
    matched_image = match_template(image_shapes, image_shapes_template)

    resized_image_up = resize_image(image_lambo, 2, 'up')
    resized_image_down = resize_image(image_lambo, 2, 'down')
    # save images
    cv2.imwrite('assignment_3/solutions/edges_sobel.png',
                edges_sobel)
    cv2.imwrite('assignment_3/solutions/edges_canny.png',
                edges_canny)
    cv2.imwrite('assignment_3/solutions/matched_image.png',
                matched_image)
    cv2.imwrite('assignment_3/solutions/resized_image_up.png',
                resized_image_up)
    cv2.imwrite('assignment_3/solutions/resized_image_down.png',
                resized_image_down)
