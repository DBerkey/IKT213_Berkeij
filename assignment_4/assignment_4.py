"""
Author: Douwe Berkeij
Date: 07-10-2025
"""
from reportlab.pdfgen import canvas
from PIL import Image
import cv2
import numpy as np

def reference_image(image: np.ndarray, pdf_path: str) -> None:
    """
    find the edge with harris save the image at the first page of the pdf as a reference image
    :param image: image as numpy array
    :param pdf_path: path to the pdf
    """
    # Find the edges in the image using the Harris corner detection method
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # Result is dilated for marking the corners
    dst = cv2.dilate(dst, None)

    # Thresholding the corners
    image[dst > 0.01 * dst.max()] = [0, 0, 255]
    # Save the image with corners marked
    cv2.imwrite("assignment_4/solutions/reference_image.png", image)
    
    # Convert OpenCV image (BGR) to PIL Image (RGB)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    
    # Create PDF using ReportLab
    # Get image dimensions
    img_width, img_height = pil_image.size
    
    # Create canvas with page size based on image dimensions
    # Convert pixels to points (1 point = 1/72 inch, assuming 72 DPI)
    page_width = img_width
    page_height = img_height
    
    c = canvas.Canvas(pdf_path, pagesize=(page_width, page_height))
    
    # Draw the PIL image directly on the PDF
    # ReportLab can handle PIL images directly
    c.drawInlineImage(pil_image, 0, 0, width=page_width, height=page_height)
    
    # Save the PDF
    c.save()

if __name__ == "__main__":
    reference_image_path = "assignment_4/reference_img.png"
    pdf_path = "assignment_4/solutions/solutions_document.pdf"
    
    image = cv2.imread(reference_image_path)
    reference_image(image, pdf_path)

