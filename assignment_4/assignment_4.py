"""
Author: Douwe Berkeij
Date: 07-10-2025
"""
from reportlab.pdfgen import canvas
from PIL import Image
import cv2
import numpy as np

def reference_image(image: np.ndarray) -> np.ndarray:
    """
    find the edge with harris save the image at the first page of the pdf as a reference image
    :param image: image as numpy array
    """
    # Find the edges in the image using the Harris corner detection method
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # Result is dilated for marking the corners
    dst = cv2.dilate(dst, None)

    # Thresholding the corners
    image[dst > 0.01 * dst.max()] = [0, 0, 255]

    return image

def save_images_to_multipage_pdf(images, pdf_path: str) -> None:
    """
    Save multiple images to a multi-page PDF file.
    :param images: List of images as numpy arrays.
    :param pdf_path: Path to save the PDF file.
    """
    # Create canvas for multi-page PDF
    c = canvas.Canvas(pdf_path)
    
    for i, image in enumerate(images):
        # Convert the image from BGR to RGB format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert the numpy array to a PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Get image dimensions
        img_width, img_height = pil_image.size
        
        # Create new page with image dimensions
        c.setPageSize((img_width, img_height))
        
        # Draw the image on the current page
        c.drawInlineImage(pil_image, 0, 0, width=img_width, height=img_height)
        
        # Start new page (except for last image)
        if i < len(images) - 1:
            c.showPage()
    
    # Save the PDF
    c.save()

def feature_based_alignment_SIFT(image_to_align: np.ndarray, reference_image: np.ndarray, max_features: int, good_match_percent: float):
    """
    Aligns the image_to_align to the reference image using SIFT feature matching.
    
    image_to_align: The image to be aligned (trainImage).
    reference_image: The reference image (queryImage).
    max_features: Maximum number of features (not used in SIFT, but kept for consistency).
    good_match_percent: Threshold for good matches (Lowe's ratio test).
    return: Tuple of (aligned_image, matches_image)
    """
    MIN_MATCH_COUNT = max_features
    
    # Convert to grayscale if images are colored
    if len(reference_image.shape) == 3:
        img1 = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    else:
        img1 = reference_image
        
    if len(image_to_align.shape) == 3:
        img2 = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    else:
        img2 = image_to_align

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Store all the good matches as per Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < good_match_percent * n.distance:
            good.append(m)

    # Create matches image
    matches_img = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Set a condition that at least MIN_MATCH_COUNT matches are to be there to find the object
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w = img1.shape
        # Warp the image_to_align to align with the reference image
        aligned_image = cv2.warpPerspective(image_to_align, M, (w, h))
        
        return aligned_image, matches_img

    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        return image_to_align, matches_img

if __name__ == "__main__":
    reference_image_path = "assignment_4/reference_img.png"
    image_to_align_path = "assignment_4/align_this.jpg"
    pdf_path = "assignment_4/solutions/solutions_document.pdf"
    
    # Load images
    reference_img = cv2.imread(reference_image_path)
    image_to_align = cv2.imread(image_to_align_path)
    
    # Page 1: Reference image with Harris corners
    processed_reference = reference_image(reference_img)
    
    # Page 2 & 3: SIFT alignment (max_features=10, good_match_percent=0.7)
    aligned_sift, matches_sift = feature_based_alignment_SIFT(
        image_to_align, processed_reference, max_features=10, good_match_percent=0.7
    )
    
    # Save individual images
    cv2.imwrite("assignment_4/solutions/reference_image.png", processed_reference)
    cv2.imwrite("assignment_4/solutions/aligned_sift.png", aligned_sift)
    cv2.imwrite("assignment_4/solutions/matches_sift.png", matches_sift)
    
    # Create multi-page PDF
    images_for_pdf = [
        processed_reference,  # Page 1: Reference with Harris corners
        aligned_sift,        # Page 2: SIFT aligned image
        matches_sift,        # Page 3: SIFT matches
    ]
    
    save_images_to_multipage_pdf(images_for_pdf, pdf_path)
    print(f"Multi-page PDF saved to: {pdf_path}")
    print("Individual images saved to assignment_4/solutions/")
    

