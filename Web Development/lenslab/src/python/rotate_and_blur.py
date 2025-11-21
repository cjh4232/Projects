import cv2
import numpy as np
import argparse
import os
import tempfile
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image  # For additional image format support


def read_file(input_path):
    """
    Read various file formats and convert to a numpy array.
    Supports PDF, PNG, JPEG, TIFF, and other common image formats.

    Parameters:
    input_path (str): Path to input file

    Returns:
    numpy.ndarray: Grayscale image as numpy array
    """
    file_ext = Path(input_path).suffix.lower()

    if file_ext == '.pdf':
        # Handle PDF files
        try:
            # Open PDF and get first page
            pdf_doc = fitz.open(input_path)
            if pdf_doc.page_count == 0:
                raise ValueError("PDF file is empty")

            # Get first page
            page = pdf_doc[0]

            # Convert to image
            pix = page.get_pixmap()

            # Create a temporary file to store the image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                pix.save(tmp_file.name)
                # Read the temporary file with OpenCV
                img = cv2.imread(tmp_file.name, cv2.IMREAD_GRAYSCALE)
                # Clean up
                os.unlink(tmp_file.name)

            pdf_doc.close()
            return img

        except Exception as e:
            raise ValueError(f"Error processing PDF: {str(e)}")

    else:
        try:
            # Try reading with OpenCV first
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                # If OpenCV fails, try with PIL
                with Image.open(input_path) as pil_img:
                    # Convert to grayscale
                    pil_img = pil_img.convert('L')
                    # Convert to numpy array
                    img = np.array(pil_img)

            return img

        except Exception as e:
            raise ValueError(f"Error reading image file: {str(e)}")


def process_image(input_path, output_dir, angle=8, sigma=1.0):
    """
    Process an image by rotating it and applying Gaussian blur.

    Parameters:
    input_path (str): Path to input file (PDF or image)
    output_dir (str): Directory to save processed images
    angle (float): Rotation angle in degrees
    sigma (float): Gaussian blur sigma value
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the input file
    print(f"Reading input file: {input_path}")
    img = read_file(input_path)

    if img is None:
        raise ValueError("Failed to read input file")

    # Get image dimensions
    height, width = img.shape
    print(f"Image dimensions: {width}x{height}")

    # Rotate image
    print(f"Rotating image by {angle} degrees")
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
    rotated = cv2.warpAffine(img, rotation_matrix, (width, height),
                             flags=cv2.INTER_LINEAR, borderValue=255)

    # Create circular mask
    center = (width//2, height//2)
    radius = min(width, height)//2
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    # Apply mask to rotated image
    masked = cv2.bitwise_and(rotated, mask)
    masked[mask == 0] = 255  # Set outside to white

    # Apply Gaussian blur
    print(f"Applying Gaussian blur with sigma={sigma}")
    blurred = cv2.GaussianBlur(masked, (0, 0), sigma)

    # Save results
    base_name = Path(input_path).stem

    # Save rotated image
    rotated_path = os.path.join(output_dir, f"{base_name}_rotated.png")
    cv2.imwrite(rotated_path, masked)

    # Save blurred image
    blurred_path = os.path.join(output_dir, f"{base_name}_rotated_blurred.png")
    cv2.imwrite(blurred_path, blurred)

    return rotated_path, blurred_path


def main():
    parser = argparse.ArgumentParser(
        description='Process image/PDF with rotation and blur')
    parser.add_argument('input_path', help='Path to input file (PDF or image)')
    parser.add_argument('--output_dir', default='output',
                        help='Output directory')
    parser.add_argument('--angle', type=float, default=8.0,
                        help='Rotation angle in degrees')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Gaussian blur sigma')

    args = parser.parse_args()

    try:
        rotated_path, blurred_path = process_image(
            args.input_path,
            args.output_dir,
            args.angle,
            args.sigma
        )
        print(f"Saved rotated image to: {rotated_path}")
        print(f"Saved blurred image to: {blurred_path}")
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
