import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import math
from scipy.ndimage import gaussian_filter


def create_quadrant_circle(size=1024,
                           rotation_angle=0, edge_transition=3,
                           slant_angle=3):
    """
    Create a circle divided into quadrants with alternating black
    and white colors.

    Parameters:
    size (int): Width and height of the output image in pixels
    rotation_angle (float): Rotation angle of the entire pattern in degrees
    edge_transition (float): Width of the transition at edges (0 for hard edge)
    slant_angle (float): Angle of slant for the dividing lines in degrees

    Returns:
    PIL.Image: The generated test pattern
    """
    # Create a white background
    img = Image.new('L', (size, size), 255)
    draw = ImageDraw.Draw(img)

    # Calculate center and radius
    center = size // 2
    radius = size * 0.45  # Slightly smaller than half to ensure it fits

    # Calculate the slanted dividing lines
    theta = math.radians(slant_angle)
    offset_x = math.tan(theta) * center
    offset_y = math.tan(theta) * center

    # Create a mask for each quadrant
    mask = np.zeros((size, size), dtype=np.uint8)

    # Draw the circle
    draw.ellipse((center - radius, center - radius,
                 center + radius, center + radius), fill=0)

    # Convert to numpy array for easier manipulation
    np_img = np.array(img)

    # Create dividing lines with slant
    for y in range(size):
        for x in range(size):
            # Calculate position relative to center
            rel_x = x - center
            rel_y = y - center

            # Apply rotation if specified
            if rotation_angle != 0:
                angle = math.radians(rotation_angle)
                rot_x = rel_x * math.cos(angle) - rel_y * math.sin(angle)
                rot_y = rel_x * math.sin(angle) + rel_y * math.cos(angle)
                rel_x, rel_y = rot_x, rot_y

            # Adjusting quadrant borders with slant
            if (rel_x > -offset_x and rel_y < offset_y) or (rel_x < offset_x and rel_y > -offset_y):
                # If in top-right or bottom-left quadrant, set to white
                distance = math.sqrt(rel_x**2 + rel_y**2)
                if distance <= radius:
                    np_img[y, x] = 255

    # Apply edge transition if specified
    if edge_transition > 0:
        np_img = smooth_edges(np_img, edge_transition)

    # Convert back to PIL Image
    result = Image.fromarray(np_img)
    return result


def smooth_edges(img_array, transition_width):
    """
    Apply a smooth transition at the edges between black and white regions.

    Parameters:
    img_array (numpy.ndarray): Input image as numpy array
    transition_width (float): Width of the transition in pixels

    Returns:
    numpy.ndarray: Image with smoothed edges
    """
    # Apply Gaussian blur for smooth transitions
    smoothed = gaussian_filter(img_array.astype(float), sigma=transition_width)

    # Normalize and convert back to uint8
    smoothed = ((smoothed - smoothed.min()) *
                (255.0 / (smoothed.max() - smoothed.min()))).astype(np.uint8)

    return smoothed


def create_slant_edge_test_pattern(size=1024, slant_angle=5):
    """
    Create a simpler slant-edge test pattern (half black, half white with a slanted edge)

    Parameters:
    size (int): Width and height of the output image in pixels
    slant_angle (float): Angle of slant for the dividing line in degrees

    Returns:
    PIL.Image: The generated test pattern
    """
    # Create a white background
    img = np.ones((size, size), dtype=np.uint8) * 255

    # Calculate the slope based on the slant angle
    slope = math.tan(math.radians(slant_angle))

    # Draw the slanted dividing line
    for y in range(size):
        # Calculate the x position for this y value
        x_position = int(size/2 + (y - size/2) * slope)

        # Ensure we're within bounds
        if x_position < 0:
            x_position = 0
        elif x_position >= size:
            x_position = size - 1

        # Fill everything to the left of this position with black
        img[y, 0:x_position] = 0

    return Image.fromarray(img)


def create_siemens_star(size=1024, num_sectors=16, inner_radius_ratio=0.1, anti_aliasing=True):
    """
    Create a Siemens star pattern which is useful for MTF analysis across multiple angles.

    Parameters:
    size (int): Width and height of the output image in pixels
    num_sectors (int): Number of black/white sector pairs
    inner_radius_ratio (float): Ratio of inner circle radius to image size
    anti_aliasing (bool): Whether to apply anti-aliasing for smoother edges

    Returns:
    PIL.Image: The generated Siemens star pattern
    """
    # Create a white background
    img = np.ones((size, size), dtype=np.float32)

    # Calculate center and outer radius
    center = size // 2
    outer_radius = size // 2 * 0.95  # Slightly smaller to ensure it fits
    inner_radius = size * inner_radius_ratio

    # Generate the pattern
    y, x = np.ogrid[-center:size-center, -center:size-center]
    r = np.sqrt(x*x + y*y)
    theta = np.arctan2(y, x) * num_sectors / (2 * np.pi)

    # Create the alternating pattern
    sector_mask = np.mod(theta, 1) >= 0.5

    # Apply radial limits (inner and outer circles)
    radial_mask = (r >= inner_radius) & (r <= outer_radius)

    # Combine masks and set colors
    img[radial_mask & sector_mask] = 0  # Black sectors

    # Apply anti-aliasing if requested
    if anti_aliasing:
        img = gaussian_filter(img, sigma=0.5)

    # Normalize and convert to uint8
    img = (img * 255).astype(np.uint8)

    return Image.fromarray(img)


# Example usage
if __name__ == "__main__":
    # Generate quadrant circle pattern
    qc_pattern = create_quadrant_circle(
        size=512, rotation_angle=0, edge_transition=0, slant_angle=0)
    qc_pattern.save("quadrant_circle_pattern.png")

    # Generate simple slant edge pattern
    slant_pattern = create_slant_edge_test_pattern(size=512, slant_angle=5)
    slant_pattern.save("slant_edge_pattern.png")

    # Generate Siemens star pattern
    star_pattern = create_siemens_star(size=512, num_sectors=16)
    star_pattern.save("siemens_star_pattern.png")

    # Display the patterns
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(qc_pattern, cmap='gray')
    plt.title("Quadrant Circle Pattern")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(slant_pattern, cmap='gray')
    plt.title("Slant Edge Pattern")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(star_pattern, cmap='gray')
    plt.title("Siemens Star Pattern")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
