import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import math
import os
from scipy.ndimage import gaussian_filter

# Constants based on ISO 12233:2014 standard
ISO_RECOMMENDED_ANGLE = 5.0     # 5 degrees from horizontal/vertical
# Minimum angle from horizontal/vertical (degrees)
MIN_ANGLE_OFFSET = 3.0
# Maximum angle from horizontal/vertical (degrees)
MAX_ANGLE_OFFSET = 15.0
ISO_RECOMMENDED_CONTRAST = 4.0  # 4:1 contrast ratio
MIN_CONTRAST_RATIO = 3.0        # Minimum acceptable contrast ratio
MAX_CONTRAST_RATIO = 5.0        # Maximum acceptable contrast ratio
ROI_LENGTH_FACTOR = 0.6         # How much of the line length to use for ROI


class EdgeROI:
    """Class to store edge detection results with ISO standard validation"""

    def __init__(self):
        self.line = None  # [x1, y1, x2, y2]
        self.angle = 0.0
        self.roi = None   # [x, y, width, height]
        self.roi_image = None
        self.valid_angle_for_mtf = False  # Whether the angle is in valid range per ISO
        # Still calculated but only for informational purposes
        self.contrast_ratio = None
        self.valid_contrast_for_mtf = False  # Not used for validation decisions
        self.mtf_validity_reason = ""     # Reason for angle validity/invalidity


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


def get_angle_validity_reason(angle_degrees):
    """Return a human-readable explanation for why an angle is valid or invalid for MTF"""
    norm_angle = angle_degrees % 180
    h_dist = min(norm_angle, 180 - norm_angle)
    v_dist = abs(90 - norm_angle)
    min_dist = min(h_dist, v_dist)
    dist_from_45 = abs((norm_angle % 90) - 45)

    if dist_from_45 < 2:
        return f"Too close to 45° (or n·π/4): {dist_from_45:.1f}° from 45°"
    elif min_dist < MIN_ANGLE_OFFSET:
        reference = "horizontal" if h_dist <= v_dist else "vertical"
        return f"Too close to {reference}: only {min_dist:.1f}° from {reference} (min {MIN_ANGLE_OFFSET}° required)"
    elif min_dist > MAX_ANGLE_OFFSET:
        reference = "horizontal" if h_dist <= v_dist else "vertical"
        return f"Too far from {reference}: {min_dist:.1f}° from {reference} (max {MAX_ANGLE_OFFSET}° allowed)"
    else:
        reference = "horizontal" if h_dist <= v_dist else "vertical"
        return f"Good angle: {min_dist:.1f}° from {reference} (optimal is ~5°)"


def estimate_contrast_ratio(roi_image):
    """
    Estimate the contrast ratio of an edge ROI using a more robust method.
    Attempts to find the two sides of the edge and measure their intensity difference.

    Returns the contrast ratio and whether it meets ISO standards.
    """
    if roi_image is None:
        return None, False, "No ROI image available"

    # Convert to grayscale if needed
    if len(roi_image.shape) == 3:
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_image

    # Apply median blur to reduce noise
    gray_blurred = cv2.medianBlur(gray, 5)

    # Use Otsu's method to get a good threshold and segment the edge sides
    _, binary = cv2.threshold(gray_blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find light and dark regions
    light_mask = binary == 255
    dark_mask = binary == 0

    # Make sure we have enough pixels in each region (at least 10% of the ROI)
    min_pixels = roi_image.shape[0] * roi_image.shape[1] * 0.1

    if np.sum(light_mask) < min_pixels or np.sum(dark_mask) < min_pixels:
        # Fallback to percentile method if segmentation failed
        sorted_pixels = np.sort(gray.flatten())
        total_pixels = len(sorted_pixels)

        # Use 10th and 90th percentiles to avoid outliers
        dark_value = np.mean(sorted_pixels[:int(total_pixels * 0.1)])
        light_value = np.mean(sorted_pixels[int(total_pixels * 0.9):])
    else:
        # Use the segmented regions
        light_value = np.mean(gray[light_mask])
        dark_value = np.mean(gray[dark_mask])

    # Ensure dark value is not too close to zero to avoid division issues
    dark_value = max(dark_value, 1.0)

    # Calculate contrast ratio
    contrast_ratio = light_value / dark_value

    # Cap excessively high contrast ratios to avoid display issues
    if contrast_ratio > 25:
        contrast_ratio = 25.0

    # Check if within ISO standards
    is_valid = MIN_CONTRAST_RATIO <= contrast_ratio <= MAX_CONTRAST_RATIO

    # Generate reason
    if contrast_ratio < MIN_CONTRAST_RATIO:
        reason = f"Contrast too low: {contrast_ratio:.1f}:1 (min {MIN_CONTRAST_RATIO}:1 recommended)"
    elif contrast_ratio > MAX_CONTRAST_RATIO:
        reason = f"Contrast too high: {contrast_ratio:.1f}:1 (max {MAX_CONTRAST_RATIO}:1 recommended)"
    else:
        reason = f"Good contrast: {contrast_ratio:.1f}:1 (optimal is ~4:1)"

    return contrast_ratio, is_valid, reason


def add_debug_visualization(image, roi, contrast_data=None):
    """Add visualization of contrast regions to help with debugging"""
    debug_img = image.copy()

    # Draw ROI
    x, y, w, h = roi
    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 255), 2)

    # If we have contrast data, draw it
    if contrast_data is not None:
        # Extract image region
        roi_img = image[y:y+h, x:x+w].copy()

        # Convert to grayscale if needed
        if len(roi_img.shape) == 3:
            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_img.copy()

        # Apply median blur
        gray_blurred = cv2.medianBlur(gray, 5)

        # Threshold to get binary image
        _, binary = cv2.threshold(
            gray_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Colorize binary image for visualization
        colored_binary = np.zeros((h, w, 3), dtype=np.uint8)
        colored_binary[binary == 255] = [0, 255, 0]  # Light areas in green
        colored_binary[binary == 0] = [0, 0, 255]    # Dark areas in red

        # Draw on the ROI part of the debug image
        alpha = 0.5  # Transparency
        for c in range(3):
            debug_img[y:y+h, x:x+w, c] = (alpha * colored_binary[:, :, c] +
                                          (1-alpha) * debug_img[y:y+h, x:x+w, c])

        # Add contrast value text
        text = f"CR: {contrast_data[0]:.1f}:1"
        cv2.putText(debug_img, text, (x, y-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return debug_img


def detect_edges_and_create_rois(input_image, expected_angle=None, debug=False):
    """
    Enhanced edge detection that includes ISO standard validation for MTF analysis.

    Parameters:
    input_image: Input image for edge detection
    expected_angle: Expected angle of the edges (optional)
    debug: Whether to output debug information and images

    Returns:
    List of EdgeROI objects with ISO validation information
    """
    detected_edges = []

    # Create a copy of the input image
    image = np.array(input_image)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Convert image to OpenCV format if needed
    if len(image.shape) == 2:
        # Grayscale to BGR
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 3:
        # Ensure BGR format (OpenCV default)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Create a mask for the central region
    masked = image.copy()
    radius = min(image.shape[0], image.shape[1]) // 20
    cv2.circle(masked, (image.shape[1] // 2, image.shape[0] //
               2), radius, (0, 0, 0), -1, cv2.LINE_AA)

    if debug:
        cv2.imwrite("debug_1_masked_input.png", masked)

    # Convert to grayscale
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)

    # Apply Canny edge detection with adaptive parameters
    canny_low = max(20, int(mean_intensity * 0.2))
    canny_high = min(200, int(mean_intensity * 0.6))
    edges = cv2.Canny(gray, canny_low, canny_high)

    if debug:
        cv2.imwrite("debug_2_detected_edges.png", edges)

    # Apply Hough Line detection with optimized parameters
    min_line_length = min(image.shape[0], image.shape[1]) // 8
    max_line_gap = min(image.shape[0], image.shape[1]) // 16
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, 50, minLineLength=min_line_length, maxLineGap=max_line_gap)

    # If no lines were detected, return empty
    if lines is None:
        if debug:
            print("No lines detected in the image")
        return []

    # Group lines by angle
    image_center = (image.shape[1] / 2, image.shape[0] / 2)

    # Store all line info for analysis
    line_info = []

    for line in lines:
        l = line[0]
        pt1 = (l[0], l[1])
        pt2 = (l[2], l[3])
        line_vector = (pt2[0] - pt1[0], pt2[1] - pt1[1])

        # Calculate angle - note the negative sign to match original code
        angle = -math.degrees(math.atan2(line_vector[1], line_vector[0]))
        while angle < 0:
            angle += 360.0

        # Calculate line center and distance from image center
        line_center = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)
        distance_from_center = math.sqrt((line_center[0] - image_center[0])**2 +
                                         (line_center[1] - image_center[1])**2)

        # Calculate line length
        line_length = math.sqrt(line_vector[0]**2 + line_vector[1]**2)

        # Store for analysis
        line_info.append({
            'line': l,
            'angle': angle,
            'length': line_length,
            'center': line_center,
            'distance': distance_from_center
        })

    if debug:
        print(f"Found {len(lines)} total lines")
        print("Line angles detected:")
        for info in line_info:
            print(f"Angle: {info['angle']:.2f}°, Length: {info['length']:.1f}")

    # Find the main edges - the quadrant pattern should have two main angles at 90° from each other
    # If we know the expected angle, we can calculate both target angles
    if expected_angle is not None:
        target_angle1 = expected_angle % 180
        target_angle2 = (target_angle1 + 90) % 180

        # Add angle normalization (handle angles near 0/180)
        def normalize_angle_diff(a1, a2):
            diff = abs(a1 - a2) % 180
            return min(diff, 180 - diff)

        # Find lines close to the target angles
        tolerance = 10  # degrees
        target1_lines = [l for l in line_info if normalize_angle_diff(
            l['angle'], target_angle1) < tolerance]
        target2_lines = [l for l in line_info if normalize_angle_diff(
            l['angle'], target_angle2) < tolerance]

        # Sort by length to get the most prominent lines
        target1_lines.sort(key=lambda x: x['length'], reverse=True)
        target2_lines.sort(key=lambda x: x['length'], reverse=True)

        # Use the longest lines from each group
        candidate_lines = target1_lines[:2] + target2_lines[:2]
    else:
        # Without expected angle, identify prominent angles using a histogram approach
        angles = [info['angle'] for info in line_info]

        # Normalize angles to 0-180 range (lines at 180° difference are the same)
        norm_angles = [angle % 180 for angle in angles]

        # Create histogram bins for angles
        hist, bins = np.histogram(norm_angles, bins=36, range=(0, 180))

        # Find the two most common angle bins
        top_bins = np.argsort(hist)[-2:]
        bin_centers = [(bins[i] + bins[i+1])/2 for i in top_bins]

        # Extract the longest lines close to these angles
        candidate_lines = []
        for center in bin_centers:
            matches = [l for l in line_info if abs(
                (l['angle'] % 180) - center) < 10]
            matches.sort(key=lambda x: x['length'], reverse=True)
            # Get top 2 from each angle cluster
            candidate_lines.extend(matches[:2])

    # Create debug image with all detected lines
    if debug and len(lines) > 0:
        all_lines_image = masked.copy()
        for line in lines:
            l = line[0]
            cv2.line(all_lines_image, (l[0], l[1]),
                     (l[2], l[3]), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite("debug_all_detected_lines.png", all_lines_image)

    # For debug - create contrast visualization
    debug_rois = []

    # Create ROIs for the candidate lines
    for line_data in candidate_lines:
        roi_data = EdgeROI()
        roi_data.line = line_data['line']
        roi_data.angle = line_data['angle']

        # Check if angle is valid for MTF per ISO standards
        roi_data.valid_angle_for_mtf = is_angle_valid_for_mtf(roi_data.angle)
        roi_data.mtf_validity_reason = get_angle_validity_reason(
            roi_data.angle)

        # Calculate ROI dimensions
        line_start = (roi_data.line[0], roi_data.line[1])
        line_end = (roi_data.line[2], roi_data.line[3])
        line_center = line_data['center']

        length = line_data['length']
        roi_length = length * ROI_LENGTH_FACTOR

        narrow_dim = min(image.shape[0], image.shape[1]) * 0.125

        # Create consistent ROI dimensions
        roi_width = roi_length
        roi_height = narrow_dim

        # Convert angle to radians for transformation
        angle_rad = np.radians(roi_data.angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        # Define corner points of ROI rectangle
        half_width = roi_width / 2
        half_height = roi_height / 2

        # Rotate corner points
        corners = [
            (line_center[0] - half_width * cos_angle + half_height * sin_angle,
             line_center[1] - half_width * sin_angle - half_height * cos_angle),
            (line_center[0] + half_width * cos_angle + half_height * sin_angle,
             line_center[1] + half_width * sin_angle - half_height * cos_angle),
            (line_center[0] + half_width * cos_angle - half_height * sin_angle,
             line_center[1] + half_width * sin_angle + half_height * cos_angle),
            (line_center[0] - half_width * cos_angle - half_height * sin_angle,
             line_center[1] - half_width * sin_angle + half_height * cos_angle)
        ]

        # Find the bounding rectangle
        min_x = max(0, int(min(c[0] for c in corners)))
        min_y = max(0, int(min(c[1] for c in corners)))
        max_x = min(image.shape[1], int(max(c[0] for c in corners)))
        max_y = min(image.shape[0], int(max(c[1] for c in corners)))

        # Create the ROI
        roi_data.roi = (min_x, min_y, max_x - min_x, max_y - min_y)

        # Extract ROI image if the dimensions are valid
        if max_x > min_x and max_y > min_y:
            roi_data.roi_image = image[min_y:max_y, min_x:max_x].copy()

            # Still estimate contrast ratio for the ROI (for informational purposes)
            roi_data.contrast_ratio, roi_data.valid_contrast_for_mtf, contrast_reason = estimate_contrast_ratio(
                roi_data.roi_image)

            # Add contrast reason but don't use it for validation
            if roi_data.mtf_validity_reason:
                roi_data.mtf_validity_reason += " | " + \
                    contrast_reason + " (informational only)"
            else:
                roi_data.mtf_validity_reason = contrast_reason + \
                    " (informational only)"

            detected_edges.append(roi_data)

            # Add debug visualization for contrast estimation
            if debug:
                debug_rois.append({
                    'roi': roi_data.roi,
                    'contrast_data': (roi_data.contrast_ratio, roi_data.valid_contrast_for_mtf, contrast_reason)
                })

    # Create debug image with selected lines and ROIs
    if debug and detected_edges:
        rois_image = image.copy()

        # Add contrast visualization
        for debug_roi in debug_rois:
            rois_image = add_debug_visualization(rois_image,
                                                 debug_roi['roi'],
                                                 debug_roi['contrast_data'])

        # Draw lines and ROIs
        for edge in detected_edges:
            # Draw the line
            cv2.line(rois_image,
                     (edge.line[0], edge.line[1]),
                     (edge.line[2], edge.line[3]),
                     (0, 255, 0), 2, cv2.LINE_AA)

            # Draw the ROI box
            x, y, w, h = edge.roi

            # Use different colors based on validity for MTF - only consider angle
            is_valid = edge.valid_angle_for_mtf
            box_color = (0, 255, 0) if is_valid else (0, 0, 255)

            cv2.rectangle(rois_image, (x, y), (x+w, y+h),
                          box_color, 2, cv2.LINE_AA)

            # Add label with detected angle and validity
            angle_text = f"{edge.angle:.1f}°"
            cv2.putText(rois_image, angle_text,
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        cv2.imwrite("debug_selected_edges.png", rois_image)
        print(f"Found {len(detected_edges)} target edges")

        # Print detailed information about each edge
        for i, edge in enumerate(detected_edges):
            angle_status = "VALID" if edge.valid_angle_for_mtf else "INVALID"
            contrast_status = "VALID" if edge.valid_contrast_for_mtf else "INVALID"
            print(f"Edge {i+1}: Detected angle = {edge.angle:.2f}° ({angle_status}) | " +
                  f"Contrast = {edge.contrast_ratio:.1f}:1 ({contrast_status}, informational only)")
            print(f"  Reason: {edge.mtf_validity_reason}")

    return detected_edges


def add_angle_validation_overlay(image, detected_edges, rotation_angle=None, detailed=True, debug=False):
    """
    Add an overlay showing whether the edges are valid for MTF analysis
    according to ISO 12233:2014 standards, considering only angle requirements.

    Parameters:
    image: Input image
    detected_edges: List of EdgeROI objects
    rotation_angle: Expected rotation angle (optional)
    detailed: Whether to include detailed information in the overlay
    debug: Whether to add debug information

    Returns:
    Image with overlay
    """
    # Convert to PIL Image if it's not already
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        pil_img = image

    draw = ImageDraw.Draw(pil_img)
    width, height = pil_img.size

    # Check if any edges meet angle requirements
    any_valid_angle = any(edge.valid_angle_for_mtf for edge in detected_edges)

    # Try to load a font (fall back to default if not available)
    try:
        font = ImageFont.truetype("Arial.ttf", 24)
        small_font = ImageFont.truetype("Arial.ttf", 18)
    except:
        try:
            # Try another common font name
            font = ImageFont.truetype("DejaVuSans.ttf", 24)
            small_font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except:
            # Last resort - default font
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

    # Create semi-transparent overlay
    overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    if any_valid_angle:
        # Draw green approval banner
        banner_y = 20
        banner_height = 40
        overlay_draw.rectangle([(0, banner_y), (width, banner_y + banner_height)],
                               fill=(0, 128, 0, 180))

        message = "Valid Angle for MTF Analysis Detected"
        # Use textbbox if available (newer Pillow), fallback to getsize
        try:
            bbox = overlay_draw.textbbox((0, 0), message, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            text_width, text_height = font.getsize(message)

        overlay_draw.text(
            ((width - text_width) // 2, banner_y + 10),
            message,
            font=font,
            fill=(255, 255, 255, 255)
        )
    else:
        # Draw warning banner
        banner_y = height // 2 - 75
        banner_height = 150
        overlay_draw.rectangle([(0, banner_y), (width, banner_y + banner_height)],
                               fill=(255, 0, 0, 180))

        # Draw warning text
        message = "Warning: Invalid Angle for MTF Analysis"

        # Use textbbox if available (newer Pillow), fallback to getsize
        try:
            bbox = overlay_draw.textbbox((0, 0), message, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            text_width, text_height = font.getsize(message)

        overlay_draw.text(
            ((width - text_width) // 2, banner_y + 15),
            message,
            font=font,
            fill=(255, 255, 255, 255)
        )

        # Add specific issues about angles
        issue = "- Edge angle should be 3-15° from horizontal/vertical and not close to 45°"

        y_offset = banner_y + 50
        # Use textbbox if available (newer Pillow), fallback to getsize
        try:
            bbox = overlay_draw.textbbox((0, 0), issue, font=small_font)
            issue_width, issue_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            issue_width, issue_height = small_font.getsize(issue)

        overlay_draw.text(
            ((width - issue_width) // 2, y_offset),
            issue,
            font=small_font,
            fill=(255, 255, 255, 255)
        )
        y_offset += issue_height + 10

        # Add ISO reference
        iso_ref = "Per ISO 12233:2014 standard recommendations"
        # Use textbbox if available (newer Pillow), fallback to getsize
        try:
            bbox = overlay_draw.textbbox((0, 0), iso_ref, font=small_font)
            ref_width, ref_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            ref_width, ref_height = small_font.getsize(iso_ref)

        overlay_draw.text(
            ((width - ref_width) // 2, banner_y + banner_height - ref_height - 15),
            iso_ref,
            font=small_font,
            fill=(255, 255, 255, 200)
        )

    # Composite the overlay onto the image
    pil_img = Image.alpha_composite(
        pil_img.convert('RGBA'), overlay).convert('RGB')

    # Add detailed edge information if requested
    if detailed and detected_edges:
        draw = ImageDraw.Draw(pil_img)
        y_offset = height - 30 * len(detected_edges) - 10

        for i, edge in enumerate(detected_edges):
            # Format text with angle information
            angle_status = "✓" if edge.valid_angle_for_mtf else "✗"

            text = f"Edge {i+1}: {edge.angle:.1f}° [{angle_status}]"
            if edge.contrast_ratio:
                # Still show contrast info, but don't factor it into validation
                text += f" | Contrast {edge.contrast_ratio:.1f}:1 (informational only)"

            draw.text((10, y_offset + i*30), text, font=small_font,
                      fill=(255, 0, 0) if not edge.valid_angle_for_mtf else (0, 128, 0))

    # Convert back to numpy array for OpenCV
    return np.array(pil_img)


def test_edge_detection(input_image_path, expected_angle=None):
    """
    Test edge detection on an input image and validate angles against ISO standards.
    """
    # Load the image
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Failed to load image: {input_image_path}")
        return None

    # Detect edges and create ROIs
    edges = detect_edges_and_create_rois(
        img, expected_angle=expected_angle, debug=True)

    if not edges:
        print("No edges detected")
        return None

    # Create result image with ISO validation overlay
    result_img = img.copy()

    # Draw each detected edge and its ROI
    for i, edge in enumerate(edges):
        # Draw the original line
        cv2.line(result_img,
                 (edge.line[0], edge.line[1]),
                 (edge.line[2], edge.line[3]),
                 (0, 255, 0), 2, cv2.LINE_AA)

        # Draw the ROI box - only consider angle validity
        x, y, w, h = edge.roi
        is_valid = edge.valid_angle_for_mtf  # Only consider angle
        box_color = (0, 255, 0) if is_valid else (0, 0, 255)
        cv2.rectangle(result_img, (x, y), (x+w, y+h),
                      box_color, 2, cv2.LINE_AA)

        # Add label with detected angle
        cv2.putText(result_img, f"{edge.angle:.1f}°",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    # Add validation overlay - only check angles
    result_img = add_angle_validation_overlay(
        result_img, edges, expected_angle, detailed=True, debug=True)

    # Save result
    output_path = input_image_path.replace('.png', '_analyzed.png')
    cv2.imwrite(output_path, result_img)

    # Print summary
    valid_for_mtf = any(edge.valid_angle_for_mtf for edge in edges)
    print(f"\nAnalysis of {input_image_path}:")
    print(f"Valid angle for MTF: {'YES' if valid_for_mtf else 'NO'}")
    print(f"Detected {len(edges)} edges")

    for i, edge in enumerate(edges):
        print(f"  Edge {i+1}:")
        print(
            f"    Angle: {edge.angle:.2f}° ({'VALID' if edge.valid_angle_for_mtf else 'INVALID'})")
        if edge.contrast_ratio:
            print(
                f"    Contrast: {edge.contrast_ratio:.2f}:1 (informational only)")
        print(f"    Reason: {edge.mtf_validity_reason}")

    print(f"Result saved to: {output_path}")

    return edges


def batch_test_edge_detection(input_dir="mtf_test_targets"):
    """Test edge detection on all images in a directory, focusing on angle validity"""
    os.makedirs("mtf_results", exist_ok=True)

    # Find all PNG files in the directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

    results = []
    for image_file in image_files:
        file_path = os.path.join(input_dir, image_file)
        print(f"\nProcessing {image_file}...")

        # Extract expected angle from filename if available
        expected_angle = None
        if "angle" in image_file:
            try:
                angle_part = image_file.split("angle")[1].split("_")[0]
                expected_angle = float(angle_part)
                print(f"Extracted expected angle: {expected_angle}°")
            except:
                pass

        # Detect edges
        edges = test_edge_detection(file_path, expected_angle)

        # Store results - only consider angle validity
        valid_angle_for_mtf = any(
            edge.valid_angle_for_mtf for edge in edges) if edges else False
        results.append({
            "filename": image_file,
            "valid_angle_for_mtf": valid_angle_for_mtf,
            "expected_valid": "valid" in image_file.lower() and "invalid" not in image_file.lower(),
            "num_edges": len(edges) if edges else 0,
            "edges": edges
        })

    # Analyze results
    correct_classifications = 0
    false_positives = 0
    false_negatives = 0

    for result in results:
        if result["valid_angle_for_mtf"] == result["expected_valid"]:
            correct_classifications += 1
        elif result["valid_angle_for_mtf"] and not result["expected_valid"]:
            false_positives += 1
        elif not result["valid_angle_for_mtf"] and result["expected_valid"]:
            false_negatives += 1

    print("\n--- BATCH TEST RESULTS ---")
    print(f"Total images analyzed: {len(results)}")
    print(
        f"Correct classifications: {correct_classifications} ({correct_classifications/len(results)*100:.1f}%)")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")

    return results


if __name__ == "__main__":
    # Generate test targets
    from improved_target_generation import generate_test_set

    # Generate a diverse set of test targets
    generate_test_set(num_targets=10, output_dir="mtf_test_targets")

    # Test edge detection on all generated targets
    batch_test_edge_detection("mtf_test_targets")
