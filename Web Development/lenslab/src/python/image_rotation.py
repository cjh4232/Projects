import cv2
import numpy as np
import fitz

# Convert PDF to image
pdf = fitz.open('./images/Slant-Edge Target.pdf')
page = pdf[0]
pix = page.get_pixmap()
pix.save('temp.png')

# Load and rotate
img = cv2.imread('temp.png', cv2.IMREAD_GRAYSCALE)
angle = 8
height, width = img.shape
rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
rotated = cv2.warpAffine(img, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=255)  # White border

# Create circular mask
center = (width//2, height//2)
radius = min(width, height)//2
mask = np.zeros((height, width), dtype=np.uint8)
cv2.circle(mask, center, radius, 255, -1)

# Apply mask
result = cv2.bitwise_and(rotated, mask)
result[mask == 0] = 255  # Set outside to white

cv2.imwrite('target_rotated_2.png', result)