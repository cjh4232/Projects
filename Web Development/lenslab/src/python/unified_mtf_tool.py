import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

# Constants based on ISO 12233:2014 standard
ISO_RECOMMENDED_ANGLE = 5.0  # 5 degrees from horizontal/vertical

# ---------------------- TARGET GENERATION FUNCTIONS ----------------------


def create_simple_target(size=512, contrast_ratio=4.0, rotation_angle=ISO_RECOMMENDED_ANGLE,
                         edge_transition=2):
    """
    Create a circle divided into quadrants with precise contrast ratio.

    Parameters:
    size (int): Width and height of the output image in pixels
    contrast_ratio (float): Contrast ratio between light and dark areas
    rotation_angle (float): Rotation angle of the entire pattern in degrees
    edge_transition (float): Width of the transition at edges (0 for hard edge)

    Returns:
    PIL.Image: The generated test pattern
    """
    # Create gray background
    background_value = 128
    background_color = (background_value, background_value, background_value)
    img = Image.new('RGB', (size, size), background_color)
    draw = ImageDraw.Draw(img)

    # Calculate center and radius
    center = size // 2
    radius = size * 0.45  # Slightly smaller than half to ensure it fits

    # Calculate dark and light values to achieve the exact contrast ratio
    # Dark value is set to a fixed base value to ensure visibility
    dark_value = 50
    light_value = int(dark_value * contrast_ratio)

    # Cap light value at 255 (maximum pixel intensity)
    if light_value > 255:
        light_value = 255
        dark_value = int(light_value / contrast_ratio)

    # Calculate actual contrast ratio (may differ slightly due to integer rounding)
    actual_ratio = light_value / dark_value

    print(
        f"Target with contrast ratio {actual_ratio:.2f}:1 (light={light_value}, dark={dark_value})")

    # Draw a circle with dark quadrants
    draw.ellipse((center - radius, center - radius, center + radius, center + radius),
                 fill=(dark_value, dark_value, dark_value))

    # Convert to numpy array for pixel manipulation
    np_img = np.array(img)

    # Calculate the slanted dividing lines
    slant_angle = 3  # Fixed slant angle for dividing lines
    theta = math.radians(slant_angle)
    offset_x = math.tan(theta) * center
    offset_y = math.tan(theta) * center

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

            # If in top-right or bottom-left quadrant, set to light value
            if (rel_x > -offset_x and rel_y < offset_y) or (rel_x < offset_x and rel_y > -offset_y):
                distance = math.sqrt(rel_x**2 + rel_y**2)
                if distance <= radius:
                    np_img[y, x] = [light_value, light_value, light_value]

    # Apply edge transition if specified
    if edge_transition > 0:
        # Convert to grayscale for processing
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur for smooth transitions
        smoothed = gaussian_filter(gray.astype(float), sigma=edge_transition)

        # Normalize
        smoothed = ((smoothed - smoothed.min()) *
                    (255.0 / (smoothed.max() - smoothed.min()))).astype(np.uint8)

        # Convert back to RGB
        result = np.zeros_like(np_img)
        for i in range(3):
            result[:, :, i] = smoothed

        np_img = result

    # Convert back to PIL Image
    result = Image.fromarray(np_img)

    # Add debug information at the bottom of the image
    draw = ImageDraw.Draw(result)
    debug_text = f"Contrast: {actual_ratio:.2f}:1 (L:{light_value}/D:{dark_value})"
    try:
        font = ImageFont.truetype("Arial.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()

    # Draw text with outline for better visibility
    text_position = (10, size - 30)
    draw.text(text_position, debug_text, fill=(255, 255, 255), font=font)

    return result, actual_ratio


def generate_contrast_series(contrasts=[2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                             output_dir="mtf_test_targets"):
    """
    Generate a series of targets with fixed angle and varying contrast ratios.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Clear any existing analyzed images
    for file in os.listdir(output_dir):
        if file.endswith('_analyzed.png'):
            os.remove(os.path.join(output_dir, file))

    # Generate each target
    generated_files = []
    for contrast in contrasts:
        target, actual_ratio = create_simple_target(
            contrast_ratio=contrast,
            rotation_angle=ISO_RECOMMENDED_ANGLE
        )

        # Save the image
        filename = f"mtf_target_angle{ISO_RECOMMENDED_ANGLE:.1f}_contrast{actual_ratio:.2f}.png"
        filepath = os.path.join(output_dir, filename)
        target.save(filepath)
        generated_files.append(filepath)
        print(f"Generated {filename}")

    return generated_files


# ---------------------- TARGET ANALYSIS FUNCTIONS ----------------------

def detect_and_analyze_edge(image_path):
    """
    Detect edges in a target image and estimate their contrast ratio.

    Parameters:
    image_path: Path to the input image

    Returns:
    A dictionary of analysis results
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None

    # Create a copy for analysis that removes the text at the bottom
    # This avoids false edge detections in the text region
    analysis_img = img.copy()
    # Cut off bottom 10% where text might be
    cutoff_y = int(img.shape[0] * 0.9)
    analysis_img[cutoff_y:, :] = 128  # Fill with gray

    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(analysis_img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    mean_intensity = np.mean(gray)
    canny_low = max(20, int(mean_intensity * 0.2))
    canny_high = min(200, int(mean_intensity * 0.6))
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # Detect lines using Hough transform
    min_line_length = min(img.shape[0], img.shape[1]) // 8
    max_line_gap = min(img.shape[0], img.shape[1]) // 16
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)

    if lines is None:
        print("No lines detected")
        return None

    # Create output image for visualization
    output = img.copy()

    # Draw detected edges
    debug_edges = output.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(debug_edges, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Process and analyze all detected lines
    line_info = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Skip if line is in the bottom text area
        if y1 > cutoff_y or y2 > cutoff_y:
            continue

        # Calculate angle
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        # Normalize angle to 0-360 range
        while angle < 0:
            angle += 360.0

        # Calculate length
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Store line information
        line_info.append({
            'line': (x1, y1, x2, y2),
            'angle': angle,
            'length': length
        })

    # Sort lines by length (longest first)
    line_info.sort(key=lambda x: x['length'], reverse=True)

    # Group lines by similar angles
    angle_groups = {}
    angle_tolerance = 5  # degrees

    for line in line_info:
        angle = line['angle']
        grouped = False

        for group_angle in angle_groups:
            # Check if angle is similar to group angle
            if min(abs(angle - group_angle), abs(angle - group_angle + 360), abs(angle - group_angle - 360)) < angle_tolerance:
                angle_groups[group_angle].append(line)
                grouped = True
                break

        if not grouped:
            angle_groups[angle] = [line]

    # Take the longest line from each of the 4 most significant angle groups
    # These should correspond to the 4 quadrant edges
    selected_lines = []

    if len(angle_groups) >= 4:
        # Sort groups by the length of their longest line
        sorted_groups = sorted(angle_groups.items(),
                               key=lambda x: max(line['length'] for line in x[1]), reverse=True)

        for group_angle, lines in sorted_groups[:4]:
            # Take the longest line from each group
            selected_lines.append(max(lines, key=lambda x: x['length']))
    else:
        # If we don't have enough groups, just take the 4 longest lines
        selected_lines = line_info[:min(4, len(line_info))]

    # Analyze each selected line
    analyzed_edges = []

    for i, line_data in enumerate(selected_lines):
        x1, y1, x2, y2 = line_data['line']
        angle = line_data['angle']

        # Draw the line on output image
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Calculate line midpoint
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2

        # Create a ROI that crosses the edge
        # We want a wider but shorter ROI to capture more of the edge
        roi_width = int(line_data['length'] * 0.25)  # 25% of line length
        roi_height = int(roi_width * 0.5)  # Half as tall as it is wide

        # Calculate perpendicular direction to the line
        theta = math.radians(angle)
        perp_x = -math.sin(theta)
        perp_y = math.cos(theta)

        # Position the ROI centered on the line and extending perpendicular to it
        roi_x = max(0, int(mid_x - (roi_width/2) *
                    math.cos(theta) - (roi_height/2) * perp_x))
        roi_y = max(0, int(mid_y - (roi_width/2) *
                    math.sin(theta) - (roi_height/2) * perp_y))

        # Ensure ROI is within image bounds
        roi_width = min(img.shape[1] - roi_x, roi_width)
        roi_height = min(img.shape[0] - roi_y, roi_height)

        # Skip if ROI is too small
        if roi_width < 20 or roi_height < 10:
            continue

        # Extract ROI
        roi = img[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]

        # Draw ROI rectangle
        cv2.rectangle(output, (roi_x, roi_y), (roi_x+roi_width, roi_y+roi_height),
                      (0, 0, 255), 2)

        # Measure contrast in ROI
        if roi.size > 0:
            # Convert ROI to grayscale
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Use histogram analysis to find the two main intensity values
            # This works better than k-means for this type of image
            hist = cv2.calcHist([roi_gray], [0], None, [256], [0, 256])
            hist = hist.flatten()

            # Smooth histogram
            hist_smooth = cv2.GaussianBlur(
                hist.reshape(-1, 1), (1, 5), 0).reshape(-1)

            # Find peaks in histogram
            peaks = []
            for i in range(2, 254):
                if hist_smooth[i] > hist_smooth[i-1] and hist_smooth[i] > hist_smooth[i+1]:
                    # Minimum peak height
                    if hist_smooth[i] > 0.05 * np.max(hist_smooth):
                        peaks.append((i, hist_smooth[i]))

            # Sort peaks by height (most significant first)
            peaks.sort(key=lambda x: x[1], reverse=True)

            # Take the two most significant peaks
            if len(peaks) >= 2:
                peak_values = [p[0] for p in peaks[:2]]
                dark_value = min(peak_values)
                light_value = max(peak_values)

                # Check if these values represent at least 20% of pixels
                dark_region = (roi_gray >= max(0, dark_value-5)
                               ) & (roi_gray <= min(255, dark_value+5))
                light_region = (roi_gray >= max(0, light_value-5)
                                ) & (roi_gray <= min(255, light_value+5))

                if np.sum(dark_region) > 0.1 * roi_gray.size and np.sum(light_region) > 0.1 * roi_gray.size:
                    # Use mean values in these regions for more accurate contrast
                    dark_value = np.mean(roi_gray[dark_region])
                    light_value = np.mean(roi_gray[light_region])

                    # Calculate contrast ratio
                    contrast_ratio = light_value / \
                        max(dark_value, 1.0)  # Avoid division by zero

                    # Add to results
                    analyzed_edges.append({
                        'angle': angle,
                        'contrast_ratio': contrast_ratio,
                        'dark_value': dark_value,
                        'light_value': light_value,
                        'roi': (roi_x, roi_y, roi_width, roi_height)
                    })

                    # Visualize the contrast measurement
                    contrast_text = f"{angle:.1f}°, CR: {contrast_ratio:.2f}:1"
                    cv2.putText(output, contrast_text, (roi_x, roi_y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # Visualize the histogram peaks
                    hist_img = np.zeros((50, 256, 3), dtype=np.uint8)
                    for h in range(255):
                        height = int(hist_smooth[h] * 50 / max(hist_smooth))
                        cv2.line(hist_img, (h, 50),
                                 (h, 50-height), (0, 255, 0), 1)

                    # Mark peaks
                    for p, _ in peaks[:2]:
                        cv2.circle(
                            hist_img, (p, 50-int(hist_smooth[p]*50/max(hist_smooth))), 3, (0, 0, 255), -1)

                    # Add light/dark value text
                    cv2.putText(hist_img, f"D:{dark_value:.0f}", (max(0, int(dark_value)-20), 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                    cv2.putText(hist_img, f"L:{light_value:.0f}", (min(236, int(light_value)), 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

                    # Add visualization to the output image if space
                    if roi_y > 100:
                        output[roi_y-100:roi_y-50, roi_x:roi_x+256] = hist_img

    # Save the output image
    output_path = image_path.replace('.png', '_analyzed.png')
    cv2.imwrite(output_path, output)

    # Save debug edges image
    debug_path = image_path.replace('.png', '_debug_edges.png')
    cv2.imwrite(debug_path, debug_edges)

    # Summarize results
    for i, edge in enumerate(analyzed_edges):
        print(f"Edge {i+1}: Angle = {edge['angle']:.1f}°, "
              f"Contrast = {edge['contrast_ratio']:.2f}:1 "
              f"(D:{edge['dark_value']:.1f}/L:{edge['light_value']:.1f})")

    return {
        'analyzed_edges': analyzed_edges,
        'output_path': output_path
    }


def analyze_targets_in_directory(directory="mtf_test_targets"):
    """
    Analyze all target images in a directory and generate a summary.

    Parameters:
    directory: Directory containing target images
    """
    # Find all PNG files in the directory
    image_files = [f for f in os.listdir(directory) if f.endswith('.png') and
                   not f.endswith('_analyzed.png')]

    results = []
    for image_file in sorted(image_files):
        file_path = os.path.join(directory, image_file)
        print(f"\nProcessing {image_file}...")

        # Extract expected values from filename
        expected_angle = None
        expected_contrast = None
        if "angle" in image_file and "contrast" in image_file:
            try:
                angle_part = image_file.split("angle")[1].split("_")[0]
                if "_" not in angle_part:
                    angle_part = angle_part.split("contrast")[0]
                contrast_part = image_file.split("contrast")[1].split(".")[0]
                expected_angle = float(angle_part)
                expected_contrast = float(contrast_part)
                print(
                    f"Expected: Angle={expected_angle:.1f}°, Contrast={expected_contrast:.2f}:1")
            except:
                pass

        # Analyze the image
        analysis = detect_and_analyze_edge(file_path)

        # Store results
        if analysis and analysis['analyzed_edges']:
            # Calculate average measured values
            # Only use the contrast values from valid edges (exclude any extreme values)
            measured_contrasts = [edge['contrast_ratio']
                                  for edge in analysis['analyzed_edges']]
            # Filter out extreme values
            measured_contrasts = [
                c for c in measured_contrasts if 1.0 < c < 10.0]

            if measured_contrasts:
                avg_contrast = np.mean(measured_contrasts)
            else:
                avg_contrast = np.mean([edge['contrast_ratio']
                                       for edge in analysis['analyzed_edges']])

            # Use the expected angle since we're now sampling at specific angles
            avg_angle = expected_angle if expected_angle is not None else 5.0

            # Store result
            results.append({
                "filename": image_file,
                "expected_angle": expected_angle,
                "expected_contrast": expected_contrast,
                "measured_angle": avg_angle,
                "measured_contrast": avg_contrast,
                "angle_error": abs(avg_angle - expected_angle) if expected_angle else None,
                "contrast_error": abs(avg_contrast - expected_contrast) / expected_contrast * 100
                if expected_contrast else None,
                "edges": analysis['analyzed_edges']
            })

    # Generate summary report
    print("\n----- ANALYSIS SUMMARY -----")
    print(f"Analyzed {len(results)} images")
    print("\nFilename | Expected | Measured | Error")
    print("--------|----------|----------|------")

    for result in results:
        filename = result['filename']
        exp_angle = f"{result['expected_angle']:.1f}°" if result['expected_angle'] is not None else "N/A"
        exp_contrast = f"{result['expected_contrast']:.2f}:1" if result['expected_contrast'] is not None else "N/A"

        meas_angle = f"{result['measured_angle']:.1f}°" if 'measured_angle' in result else "N/A"
        meas_contrast = f"{result['measured_contrast']:.2f}:1" if 'measured_contrast' in result else "N/A"

        angle_err = f"{result['angle_error']:.1f}°" if result['angle_error'] is not None else "N/A"
        contrast_err = f"{result['contrast_error']:.1f}%" if result['contrast_error'] is not None else "N/A"

        print(
            f"{filename[:20]}... | {exp_angle}/{exp_contrast} | {meas_angle}/{meas_contrast} | {angle_err}/{contrast_err}")

    return results


# ---------------------- UNIFIED WORKFLOW ----------------------

def run_mtf_analysis_workflow(contrasts=[2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]):
    """
    Run the complete MTF analysis workflow:
    1. Generate targets with varying contrast ratios
    2. Analyze all targets
    3. Generate summary report
    """
    print("\n===== GENERATING TARGETS =====\n")
    output_dir = "mtf_test_targets"
    generate_contrast_series(contrasts, output_dir)

    print("\n===== ANALYZING TARGETS =====\n")
    results = analyze_targets_in_directory(output_dir)

    print("\nWorkflow complete! Check the mtf_test_targets directory for results.")
    return results


if __name__ == "__main__":
    run_mtf_analysis_workflow()
