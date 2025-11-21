import cv2
import numpy as np
import fitz  # PyMuPDF

# Install with: pip3 install PyMuPDF

# Load PDF and convert to image
doc = fitz.open('images/Slant-Edge Target.pdf')
page = doc[0]
pix = page.get_pixmap()
pix.save('target.png')

# Process image
img = cv2.imread('target.png', cv2.IMREAD_GRAYSCALE)
sigmas = [0.5, 1.0, 1.5, 2.0, 3.0]

for sigma in sigmas:
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    cv2.imwrite(f'target_blur_{sigma}.png', blurred)