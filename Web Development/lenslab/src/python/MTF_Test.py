import numpy as np
from scipy.fftpack import fft2, fftshift
import matplotlib.pyplot as plt
from scipy.special import j1

def slanted_edge_mthod(image):
    # Step 0A: Check if the image is a 2D array
    if len(image.shape) != 2:
        raise ValueError("The input image must be a 2D array")
    # Step 0B: Check if the image is a grayscale image
    if image.dtype != np.uint8:
        raise ValueError("The input image must be a grayscale image")
    # Step 1: Compute the gradient of the image
    gradient = np.gradient(image)
    # Step 2: Compute the edge spread function
    edge_spread_function = np.fft.fft2(gradient)
    # Step 3: Compute the line spread function
    line_spread_function = np.fft.ifft2(edge_spread_function)
    # Step 4: Compute the modulation transfer function
    modulation_transfer_function = np.abs(fftshift(fft2(line_spread_function)))
    # Step 5: Compute the point spread function
    point_spread_function = np.fft.ifft2(modulation_transfer_function)
    # Step 6: Compute the contrast transfer function
    contrast_transfer_function = np.abs(fftshift(fft2(point_spread_function)))
    # Step 7: Compute the noise power spectrum
    noise_power_spectrum = np.fft.fft2(image)
    # Step 8: Compute the detective quantum efficiency
    detective_quantum_efficiency = np.abs(fftshift(noise_power_spectrum))
    # Step 9: Return the results
    return gradient, edge_spread_function, line_spread_function, modulation_transfer_function, point_spread_function, contrast_transfer_function, noise_power_spectrum, detective_quantum_efficiency


def calculate_mtf(image):
    # Step 0A: Check if the image is a 2D array
    if len(image.shape) != 2:
        raise ValueError("The input image must be a 2D array")
    # Step 0B: Check if the image is a grayscale image
    if image.dtype != np.uint8:
        raise ValueError("The input image must be a grayscale image")
    # Step 1: Compute the point spread function
    point_spread_function = np.fft.fft2(image)
    # Step 1A: Normalize the point spread function
    point_spread_function /= np.max(point_spread_function)
    # Step 2: Computer optical transfer function
    optical_transfer_function = fft2(point_spread_function)
    optical_transfer_function_shifted = fftshift(optical_transfer_function)
    # Step 3: Compute the modulation transfer function
    modulation_transfer_function = np.abs(optical_transfer_function_shifted)
    # Step 4: Compute the spatial frequency
    rows, cols = image.shape
    u = np.arange(-cols // 2, cols // 2)
    v = np.arange(-rows // 2, rows // 2)
    u, v = np.meshgrid(u, v)
    u = u / cols
    v = v / rows
    spatial_frequency = np.sqrt(u ** 2 + v ** 2)
    # Step 5: Return the results
    return modulation_transfer_function, spatial_frequency

def airy_disk(size, wavelength, aperture_diameter, scale_factor=1):
    """
    Generate a perfect Airy disk image in grayscale.
    
    Parameters:
    - size: The size of the image (width and height in pixels).
    - wavelength: Wavelength of the light (arbitrary units, used for scaling).
    - aperture_diameter: Diameter of the aperture (arbitrary units, used for scaling).
    - scale_factor: Scaling factor to adjust the disk size.
    
    Returns:
    - A 2D numpy array representing the Airy disk image.
    """
    # Create a grid of points
    y, x = np.indices((size, size))
    center = size // 2
    x = x - center
    y = y - center
    r = np.sqrt(x**2 + y**2) * scale_factor

    # Calculate the scaling factor k
    k = (2 * np.pi / wavelength) * (aperture_diameter / 2)

    # Calculate the Airy disk intensity
    with np.errstate(divide='ignore', invalid='ignore'):
        airy_pattern = (2 * j1(k * r) / (k * r))**2
        airy_pattern[r == 0] = 1  # Correct the value at the center

    # Normalize the intensity to the range [0, 1]
    airy_pattern /= np.max(airy_pattern)

    return airy_pattern

# Parameters
size = 256
wavelength = 550e-9  # Example wavelength in meters
aperture_diameter = 2.5e-3  # Example aperture diameter in meters
scale_factor = 1e6  # Adjust the scale to fit the image size

# Generate the Airy disk image
airy_disk_image = airy_disk(size, wavelength, aperture_diameter, scale_factor)

# Plot the Airy disk image
plt.imshow(airy_disk_image, cmap='gray')
plt.title('Perfect Airy Disk Image')
plt.xlabel('X (pixels)')
plt.ylabel('Y (pixels)')
plt.colorbar(label='Intensity')
plt.show()

# Parameters

def main():
    radius = 50
    size = 256
    wavelength = 550e-9  # Example wavelength in meters
    aperture_diameter = 2.5e-3  # Example aperture diameter in meters

    # Generate the Airy disk image
    airy_disk_image = airy_disk(radius, size, wavelength, aperture_diameter)

    # Display the Airy disk image
    plt.figure(figsize=(8, 8))

    # Plot the Airy disk image
    plt.imshow(airy_disk_image, cmap='gray')
    plt.title('Perfect Airy Disk Image')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.colorbar(label='Intensity')
    plt.show()

if __name__ == "__main__":
    main()
