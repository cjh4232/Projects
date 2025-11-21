import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import math
import os
from scipy.ndimage import gaussian_filter
import random

# Constants from ISO 12233:2014 standard
ISO_RECOMMENDED_CONTRAST = 4.0  # 4:1 contrast ratio
ISO_RECOMMENDED_ANGLE = 5.0     # 5 degrees from horizontal/vertical
# Minimum angle from horizontal/vertical (degrees)
MIN_ANGLE_OFFSET = 3.0
# Maximum angle from horizontal/vertical (degrees)
MAX_ANGLE_OFFSET = 15.0


def create_quadrant_circle(size=1024, rotation_angle=0, edge_transition=3, slant_angle=3,
                           contrast_ratio=ISO_RECOMMENDED_CONTRAST):
    """
    Create a circle divided into quadrants with alternating black and white colors,
    with fixed 4:1 contrast ratio (ISO standard).

    Parameters:
    size (int): Width and height of the output image in pixels
    rotation_angle (float): Rotation angle of the entire pattern in degrees
    edge_transition (float): Width of the transition at edges (0 for hard edge)
    slant_angle (float): Angle of slant for the dividing lines in degrees
    contrast_ratio (float): Parameter kept for compatibility but always uses ISO standard 4:1

    Returns:
    PIL.Image: The generated test pattern
    """
    # Always use ISO recommended contrast ratio regardless of input parameter
    contrast_ratio = ISO_RECOMMENDED_CONTRAST

    # Calculate pixel values for the fixed contrast ratio
    middle_gray = 128
    dark_value = int(2 * middle_gray / (contrast_ratio + 1))
    light_value = int(dark_value * contrast_ratio)

    # Clamp values to valid range
    dark_value = max(0, min(255, dark_value))
    light_value = max(0, min(255, light_value))

    print(
        f"Creating target with fixed contrast ratio {contrast_ratio}:1 (light={light_value}, dark={dark_value})")

    # Create a gray background
    img = Image.new('RGB', (size, size),
                    (middle_gray, middle_gray, middle_gray))
    draw = ImageDraw.Draw(img)

    # Calculate center and radius
    center = size // 2
    radius = size * 0.45  # Slightly smaller than half to ensure it fits

    # Calculate the slanted dividing lines
    theta = math.radians(slant_angle)
    offset_x = math.tan(theta) * center
    offset_y = math.tan(theta) * center

    # Draw the circle
    draw.ellipse((center - radius, center - radius, center + radius, center + radius),
                 fill=(dark_value, dark_value, dark_value))

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
                    np_img[y, x] = [light_value, light_value, light_value]

    # Apply edge transition if specified
    if edge_transition > 0:
        np_img = smooth_edges(np_img, edge_transition)

    # Convert back to PIL Image
    result = Image.fromarray(np_img)
    return result


def smooth_edges(img_array, transition_width):
    """Apply a smooth transition at the edges between black and white regions."""
    # Convert to grayscale for processing
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.copy()

    # Apply Gaussian blur for smooth transitions
    smoothed = gaussian_filter(gray.astype(float), sigma=transition_width)

    # Normalize
    smoothed = ((smoothed - smoothed.min()) *
                (255.0 / (smoothed.max() - smoothed.min()))).astype(np.uint8)

    # If original was RGB, convert back
    if len(img_array.shape) == 3:
        result = np.zeros_like(img_array)
        for i in range(3):
            result[:, :, i] = smoothed
        return result

    return smoothed


def is_angle_valid_for_mtf(angle_degrees):
    """
    Check if angle is in valid range for MTF analysis per ISO 12233:2014.

    The angle should be between MIN_ANGLE_OFFSET and MAX_ANGLE_OFFSET degrees
    from horizontal (0°, 180°) or vertical (90°, 270°).

    Additionally, angles very close to 45° should be avoided as they can cause
    aliasing issues in the MTF calculation.
    """
    # Normalize angle to 0-180 range (treat angles that differ by 180° as the same)
    norm_angle = angle_degrees % 180

    # Calculate minimum distance from horizontal (0° or 180°)
    h_dist = min(norm_angle, 180 - norm_angle)

    # Calculate minimum distance from vertical (90°)
    v_dist = abs(90 - norm_angle)

    # Take the smaller of the two distances
    min_dist = min(h_dist, v_dist)

    # Check if within critical 45° region (avoid n*π/4 angles)
    dist_from_45 = abs((norm_angle % 90) - 45)
    critical_45deg_region = dist_from_45 < 2  # Within 2° of 45°

    # Valid if within proper range from horizontal/vertical AND not too close to 45°
    return (MIN_ANGLE_OFFSET <= min_dist <= MAX_ANGLE_OFFSET) and not critical_45deg_region


def generate_random_mtf_target(size=512, output_dir="mtf_test_targets", save=True):
    """
    Generate a random MTF test target with varying angles that may or may not be valid
    according to ISO standards. Contrast ratio is fixed to 4:1 per ISO standard.

    Returns a dictionary with the generated parameters and validity status.
    """
    # Generate random parameters with fixed contrast ratio
    rotation_angle = random.uniform(0, 90)  # 0-90 degrees rotation
    edge_transition = random.uniform(1, 5)  # 1-5 pixel edge transition
    contrast_ratio = ISO_RECOMMENDED_CONTRAST  # Fixed at 4:1

    # Generate the target
    pattern = create_quadrant_circle(
        size=size,
        rotation_angle=rotation_angle,
        edge_transition=edge_transition,
        slant_angle=3,  # Fixed slant angle for dividing lines
        contrast_ratio=contrast_ratio
    )

    # Check if the angle is valid according to ISO standards
    valid_angle = is_angle_valid_for_mtf(rotation_angle)

    # Save the image if requested
    if save:
        os.makedirs(output_dir, exist_ok=True)
        validity_tag = "valid" if valid_angle else "invalid"
        filename = f"mtf_target_angle{rotation_angle:.1f}_{validity_tag}.png"
        pattern.save(os.path.join(output_dir, filename))

    return {
        "image": pattern,
        "rotation_angle": rotation_angle,
        "edge_transition": edge_transition,
        "contrast_ratio": contrast_ratio,
        "valid_angle": valid_angle,
        "is_valid": valid_angle  # Only consider angle validity
    }


def generate_iso_compliant_target(size=512, output_dir="mtf_test_targets", save=True):
    """
    Generate an MTF test target that strictly follows ISO 12233:2014 recommendations.
    """
    # Use ISO recommended values
    rotation_angle = ISO_RECOMMENDED_ANGLE  # 5 degrees
    edge_transition = 2  # Moderate edge transition
    contrast_ratio = ISO_RECOMMENDED_CONTRAST  # 4:1 contrast

    # Generate the target
    pattern = create_quadrant_circle(
        size=size,
        rotation_angle=rotation_angle,
        edge_transition=edge_transition,
        slant_angle=3,  # Fixed slant angle for dividing lines
        contrast_ratio=contrast_ratio
    )

    # Save the image if requested
    if save:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"mtf_target_ISO_compliant.png"
        pattern.save(os.path.join(output_dir, filename))

    return {
        "image": pattern,
        "rotation_angle": rotation_angle,
        "edge_transition": edge_transition,
        "contrast_ratio": contrast_ratio,
        "valid_angle": True,
        "is_valid": True
    }


def generate_test_set(num_targets=10, output_dir="mtf_test_targets"):
    """
    Generate a set of test targets with varying angles for robustness testing.
    Half will be valid according to ISO standards, half will be invalid.
    Contrast ratio is fixed to 4:1 as per ISO 12233:2014.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate one perfect ISO-compliant target
    iso_target = generate_iso_compliant_target(output_dir=output_dir)

    # Generate targets that are within valid angle ranges
    valid_targets = []
    for i in range(num_targets // 2):
        # Random angle within valid range
        if i % 2 == 0:  # Half near horizontal
            angle = random.uniform(MIN_ANGLE_OFFSET, MAX_ANGLE_OFFSET)
        else:  # Half near vertical
            angle = random.uniform(90 - MAX_ANGLE_OFFSET,
                                   90 - MIN_ANGLE_OFFSET)

        # Fixed contrast with ISO standard
        contrast = ISO_RECOMMENDED_CONTRAST

        # Edge transition
        edge_blur = random.uniform(1.5, 3.0)

        target = create_quadrant_circle(
            rotation_angle=angle,
            edge_transition=edge_blur,
            contrast_ratio=contrast
        )

        validity_tag = "valid"
        filename = f"mtf_target_{i:02d}_angle{angle:.1f}_{validity_tag}.png"
        target.save(os.path.join(output_dir, filename))
        valid_targets.append({
            "filename": filename,
            "rotation_angle": angle,
            "contrast_ratio": contrast,
            "edge_transition": edge_blur,
            "valid": True
        })

    # Generate targets that are outside valid angle ranges
    invalid_targets = []
    for i in range(num_targets // 2):
        # Generate invalid angle (either too close to 0/90 or at 45 degrees)
        if random.random() < 0.5:
            angle = random.choice([
                # Too close to horizontal
                random.uniform(0, MIN_ANGLE_OFFSET - 0.5),
                random.uniform(90 - MIN_ANGLE_OFFSET + 0.5,
                               90)  # Too close to vertical
            ])
        else:
            angle = random.uniform(40, 50)  # Near 45 degrees

        # Fixed contrast with ISO standard
        contrast = ISO_RECOMMENDED_CONTRAST

        edge_blur = random.uniform(1.5, 3.0)

        target = create_quadrant_circle(
            rotation_angle=angle,
            edge_transition=edge_blur,
            contrast_ratio=contrast
        )

        validity_tag = "invalid"
        filename = f"mtf_target_{i+num_targets//2:02d}_angle{angle:.1f}_{validity_tag}.png"
        target.save(os.path.join(output_dir, filename))
        invalid_targets.append({
            "filename": filename,
            "rotation_angle": angle,
            "contrast_ratio": contrast,
            "edge_transition": edge_blur,
            "valid": False,
            "invalid_reason": "angle"
        })

    # Return summary of generated targets
    return {
        "iso_target": iso_target,
        "valid_targets": valid_targets,
        "invalid_targets": invalid_targets
    }


if __name__ == "__main__":
    # Generate a set of test targets
    generate_test_set(num_targets=20, output_dir="mtf_test_targets")

    # Generate one perfect ISO-compliant target
    iso_target = generate_iso_compliant_target()

    print("Target generation complete. Check the 'mtf_test_targets' directory for the generated images.")
